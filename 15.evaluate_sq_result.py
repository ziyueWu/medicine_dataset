# -*- coding: UTF-8 -*-

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import call_gpt5


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "13.build_sq_dataset"
RESPONSE_DIR = ROOT / "14.test_sq_dataset"
OUTPUT_DIR = ROOT / "15.evaluate_sq_result"
EVALUATOR_MODELS = ["deepseek-v4-pro", "gpt-5.4", "claude-sonnet-4-6"]

MYSQL_USER = "root"
MYSQL_PASSWORD = "12345678"
MYSQL_HOST = "localhost"
MYSQL_DATABASE = "mydb"
MYSQL_TABLE_NAME = "herb_zhongyao"

PROMPT_TEMPLATE = """假设你是一个主观题打分工具，目的是给题目对应的回答打分。请阅读以下内容，按照以下要求进行打分，并按最后的格式要求返回得分结果。
题目：{question}
回答：{response_text}
参考：{diff_component_sql_list}
得分点：1.在基础方剂的基础上，提出了具体的加味药物名称，2分；2.解释了所加药物的作用，需要与数据库中查找的功效匹配，2分；3.加入药物后，需要解释新的方剂可能的治疗病症变化，需要与数据库中查找到的主治匹配，2分。
请阅读题目和回答，结合参考和得分点对该回答打分，并给出该分数的置信度，如果你十分有依据，置信度为1，如果你非常没有把握，则置信度为0.1。
请将分数结果整理为如下所示的json格式，仅仅返回该json，不需要输出额外的内容，不需要输出额外的内容，不需要输出额外的内容：
{{"score_1":2, "score_2":1, "score_3":1, "socre_total": 4, "confidence": 0.8}}
"""


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_question_index(questions: List[dict]) -> Dict[int, dict]:
    return {item["id"]: item for item in questions}


def safe_parse_response_json(text: Optional[str]) -> Optional[dict]:
    return safe_parse_json_text(text)


def normalize_diff_component(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    result.append(stripped)
        return result
    if isinstance(value, str):
        parts = re.split(r"[、，,；;\s]+", value)
        return [item.strip() for item in parts if item.strip()]
    return []


def query_component_info(component: str) -> Optional[dict]:
    escaped_component = component.replace("'", "''")
    sql = (
        f"SELECT `药名`, `功效`, `主治` "
        f"FROM {MYSQL_DATABASE}.{MYSQL_TABLE_NAME} "
        f"WHERE `药名` = '{escaped_component}';"
    )
    command = [
        "mysql",
        f"--host={MYSQL_HOST}",
        f"--user={MYSQL_USER}",
        f"--password={MYSQL_PASSWORD}",
        "--batch",
        "--raw",
        "--skip-column-names",
        "-e",
        sql,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except Exception as exc:
        print(f"查询药材失败：component={component} error={exc}")
        return None

    if completed.returncode != 0:
        print(f"查询药材失败：component={component} stderr={completed.stderr.strip()}")
        return None

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    first = lines[0].split("\t")
    name = first[0].strip() if len(first) > 0 else ""
    effect = first[1].strip() if len(first) > 1 else ""
    cure = first[2].strip() if len(first) > 2 else ""
    return {"name": name, "effect": effect, "cure": cure}


def build_diff_component_sql_list(diff_components: List[str]) -> List[dict]:
    results = []
    for component in diff_components:
        row = query_component_info(component)
        if row:
            results.append(row)
        else:
            results.append({"name": component, "effect": "", "cure": ""})
    return results


def build_evaluate_prompt(question: str, response_text: Optional[str], diff_component_sql_list: List[dict]) -> str:
    return PROMPT_TEMPLATE.format(
        question=question,
        response_text=response_text or "",
        diff_component_sql_list=json.dumps(diff_component_sql_list, ensure_ascii=False),
    )


def safe_parse_json_text(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        return None

    candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def extract_score_total(evaluation_json: Optional[dict]) -> float:
    if not isinstance(evaluation_json, dict):
        return 0.0

    for key in ["score_total", "socre_total"]:
        value = evaluation_json.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    return 0.0


def extract_confidence(evaluation_json: Optional[dict]) -> float:
    if not isinstance(evaluation_json, dict):
        return 0.0

    value = evaluation_json.get("confidence")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def call_evaluator(evaluate_prompt: str, model_name: str) -> Optional[str]:
    return call_gpt5.call_gpt5(evaluate_prompt, model_name=model_name)


def calculate_score_llm_all(model_evaluations: Dict[str, dict]) -> float:
    kimi = model_evaluations.get("deepseek-v4-pro", {})
    claude = model_evaluations.get("claude-sonnet-4-6", {})
    gpt = model_evaluations.get("gpt-5.4", {})
    return (
        float(kimi.get("score_total", 0.0)) * float(kimi.get("confidence", 0.0))
        + float(claude.get("score_total", 0.0)) * float(claude.get("confidence", 0.0))
        + float(gpt.get("score_total", 0.0)) * float(gpt.get("confidence", 0.0))
    )


def modify_result() -> List[Path]:
    ensure_output_dir()
    updated_files: List[Path] = []
    evaluation_files = sorted(OUTPUT_DIR.glob("*.json"))

    print(f"共发现 {len(evaluation_files)} 个评价结果文件待修正。")

    for evaluation_file in evaluation_files:
        payload = load_json(evaluation_file)
        if not isinstance(payload, dict):
            print(f"跳过非字典结构文件：{evaluation_file.name}")
            continue

        results = payload.get("results", [])
        if not isinstance(results, list):
            print(f"跳过 results 非列表文件：{evaluation_file.name}")
            continue

        modified_count = 0
        for row in results:
            if not isinstance(row, dict):
                continue
            model_evaluations = row.get("model_evaluations", {})
            if not isinstance(model_evaluations, dict):
                continue

            new_score_llm_all = calculate_score_llm_all(model_evaluations)
            old_score_llm_all = row.get("score_llm_all")
            row["score_llm_all"] = new_score_llm_all
            if old_score_llm_all != new_score_llm_all:
                modified_count += 1

        write_json(evaluation_file, payload)
        updated_files.append(evaluation_file)
        print(f"已修正：{evaluation_file.name} | 更新 {modified_count} 条记录")

    return updated_files


def load_existing_results(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    data = load_json(path)
    rows = data.get("results", []) if isinstance(data, dict) else []
    indexed: Dict[int, dict] = {}
    for row in rows:
        row_id = row.get("id")
        if isinstance(row_id, int):
            indexed[row_id] = row
    return indexed


def build_output_path(response_file: Path) -> Path:
    output_name = response_file.name.replace("_responses.json", "_evaluations.json")
    return OUTPUT_DIR / output_name


def is_model_evaluation_complete(model_evaluation: Any) -> bool:
    if not isinstance(model_evaluation, dict):
        return False
    return model_evaluation.get("evaluation_text") is not None and model_evaluation.get("evaluation_json") is not None


def is_result_complete(row: dict) -> bool:
    model_evaluations = row.get("model_evaluations")
    if not isinstance(model_evaluations, dict):
        return False
    return all(is_model_evaluation_complete(model_evaluations.get(model_name)) for model_name in EVALUATOR_MODELS)


def evaluate_single_file(
    response_file: Path,
    question_index: Dict[int, dict],
    sleep_seconds: float = 0.0,
    overwrite: bool = False,
) -> Path:
    response_payload = load_json(response_file)
    response_rows = response_payload.get("results", []) if isinstance(response_payload, dict) else []
    output_path = build_output_path(response_file)

    existing_by_id = {} if overwrite else load_existing_results(output_path)
    results_by_id: Dict[int, dict] = dict(existing_by_id)

    total = len(response_rows)
    for index, row in enumerate(response_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, int):
            print(f"[{response_file.name}] {index}/{total} | 跳过无效 id")
            continue

        existing_row = existing_by_id.get(row_id, {}) if isinstance(existing_by_id.get(row_id), dict) else {}

        if existing_row and is_result_complete(existing_row):
            print(f"[{response_file.name}] {index}/{total} | 跳过已完成 id={row_id}")
            continue

        question_item = question_index.get(row_id)
        if not question_item:
            print(f"[{response_file.name}] {index}/{total} | 未找到题目 id={row_id}，跳过")
            continue

        response_json = safe_parse_response_json(row.get("response_text"))
        question_text = row.get("question") or question_item.get("question", "")
        formula_name = row.get("formula_name") or question_item.get("formula_name", "")
        modified_name = ""
        component_effect = ""
        modified_cure = ""

        if isinstance(response_json, dict):
            modified_name = response_json.get("modified_name") or ""
            component_effect = response_json.get("component_effect") or ""
            modified_cure = response_json.get("modified_cure") or ""

        diff_component = normalize_diff_component(
            (response_json or {}).get("diff_component") if isinstance(response_json, dict) else row.get("diff_component")
        )
        diff_component_sql_list = build_diff_component_sql_list(diff_component)

        evaluate_prompt = build_evaluate_prompt(
            question=question_text,
            response_text=row.get("response_text"),
            diff_component_sql_list=diff_component_sql_list,
        )

        print(f"[{response_file.name}] {index}/{total} | 开始评价 id={row_id}")
        existing_model_evaluations = existing_row.get("model_evaluations", {}) if isinstance(existing_row, dict) else {}
        model_evaluations = {}
        for evaluator_model in EVALUATOR_MODELS:
            existing_evaluation = existing_model_evaluations.get(evaluator_model)
            if is_model_evaluation_complete(existing_evaluation):
                print(f"[{response_file.name}] {index}/{total} | 复用已有 {evaluator_model} 评价 id={row_id}")
                model_evaluations[evaluator_model] = existing_evaluation
                continue

            print(f"[{response_file.name}] {index}/{total} | 调用 {evaluator_model} 评价 id={row_id}")
            evaluation_text = call_evaluator(evaluate_prompt, model_name=evaluator_model)
            evaluation_json = safe_parse_json_text(evaluation_text)
            model_evaluations[evaluator_model] = {
                "evaluation_text": evaluation_text,
                "evaluation_json": evaluation_json,
                "score_total": extract_score_total(evaluation_json),
                "confidence": extract_confidence(evaluation_json),
            }
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        score_llm_all = calculate_score_llm_all(model_evaluations)

        results_by_id[row_id] = {
            "id": row_id,
            "question": question_text,
            "response_model_name": row.get("model_name"),
            "evaluator_model_names": EVALUATOR_MODELS,
            "formula_name": formula_name,
            "modified_name": modified_name,
            "diff_component": diff_component,
            "diff_component_sql_list": diff_component_sql_list,
            "component_effect": component_effect,
            "modified_cure": modified_cure,
            "response_text": row.get("response_text"),
            "response_json": response_json,
            "evaluate_prompt": evaluate_prompt,
            "model_evaluations": model_evaluations,
            "score_llm_all": score_llm_all,
        }

        ordered_results = [results_by_id[item.get("id")] for item in response_rows if isinstance(item.get("id"), int) and item.get("id") in results_by_id]
        payload = {
            "dataset": response_payload.get("dataset", "dataset1"),
            "response_file": response_file.name,
            "response_model_name": response_payload.get("model_name"),
            "evaluator_model_names": EVALUATOR_MODELS,
            "total_questions": total,
            "completed_questions": len(ordered_results),
            "results": ordered_results,
        }
        write_json(output_path, payload)

    ordered_results = [results_by_id[item.get("id")] for item in response_rows if isinstance(item.get("id"), int) and item.get("id") in results_by_id]
    payload = {
        "dataset": response_payload.get("dataset", "dataset1"),
        "response_file": response_file.name,
        "response_model_name": response_payload.get("model_name"),
        "evaluator_model_names": EVALUATOR_MODELS,
        "total_questions": total,
        "completed_questions": len(ordered_results),
        "results": ordered_results,
    }
    write_json(output_path, payload)
    return output_path


def get_response_files(dataset_name: str = "dataset2") -> List[Path]:
    return sorted(RESPONSE_DIR.glob(f"{dataset_name}_*_responses.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评价 14.test_sq_dataset 中的主观题回答结果")
    parser.add_argument(
        "--dataset",
        default="dataset2",
        choices=["dataset1", "dataset2"],
        help="待评价的数据集名称，当前默认 dataset1",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="每次评价请求后的等待秒数，默认 0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="是否覆盖已有评价结果重新运行；默认断点续跑",
    )
    parser.add_argument(
        "--modify-result",
        action="store_true",
        help="单独运行结果修正流程，重算 15.evaluate_sq_result 目录下所有 JSON 的 score_llm_all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.modify_result:
        output_paths = modify_result()
        print("\n=== 全部结果修正完成 ===")
        for output_path in output_paths:
            print(f"- {output_path}")
        return

    ensure_output_dir()
    question_path = DATASET_DIR / f"{args.dataset}_questions.json"
    questions = load_json(question_path)
    question_index = build_question_index(questions)
    response_files = get_response_files(args.dataset)

    print(f"共发现 {len(response_files)} 个待评价结果文件。")
    print(f"评价模型：{', '.join(EVALUATOR_MODELS)}")
    output_paths: List[Path] = []

    for response_file in response_files:
        print(f"\n=== 开始评价：{response_file.name} ===")
        output_path = evaluate_single_file(
            response_file=response_file,
            question_index=question_index,
            sleep_seconds=args.sleep_seconds,
            overwrite=args.overwrite,
        )
        output_paths.append(output_path)
        print(f"=== 评价完成：{output_path.name} ===")

    print("\n=== 全部评价完成 ===")
    for output_path in output_paths:
        print(f"- {output_path}")


if __name__ == "__main__":
    main()