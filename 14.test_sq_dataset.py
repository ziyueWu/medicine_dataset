# -*- coding: UTF-8 -*-

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import call_baichuan
import call_gpt5


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "13.build_sq_dataset"
OUTPUT_DIR = ROOT / "14.test_sq_dataset"

SUPPORTED_MODELS = [
    "gpt-5.4",
    "gemini-3.1-pro-preview",
    "claude-sonnet-4-6",
    "qwen3.5-plus",
    "o3",
    "kimi-k2.5",
    "deepseek-v3.2",
    "baichuan-m3-plus"
]

DEFAULT_RETRY_TIMES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 2.0
MIN_VALID_RESPONSE_LENGTH = 80
INVALID_RESPONSE_PATTERNS = [
    "我无法回答这个问题",
    "不能提供医疗建议",
    "建议查阅权威的中医药教材和文献",
    "咨询有资质的中医师",
    "如果你有软件开发、编程或其他技术方面的问题",
]
DATASET_PROMPT_PREFIX = {
    "dataset1": "",
    "dataset2": "请完成关于中医加减方的测试，所谓加减方，是如麻黄汤（麻黄9克、桂枝6克、杏仁6克、炙甘草3克）和麻黄加术汤（麻黄汤原方加白术12g）这样的方剂。\n",
}

DATASET_PROMPT_SUFFIX = {
    "dataset1": "",
    "dataset2": '\n请不要回答多余的内容，仅用json的形式返回加味后的方剂名称、增加的药物名称、增加药物的作用、加味后方剂的治疗病症，如{"modified_name":"麻黄加术加茯苓汤", "diff_component": "茯苓", "component_effect":"渗湿利水；健脾和胃；宁心安神", "modified_cure":"脾虚食少；心悸不安；"}',
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


def build_output_path(dataset_name: str, model_name: str) -> Path:
    return OUTPUT_DIR / f"{dataset_name}_{slugify_model_name(model_name)}_responses.json"


def read_existing_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    data = load_json(path)
    return data.get("results", []) if isinstance(data, dict) else []


def index_rows_by_id(rows: List[dict]) -> Dict[int, dict]:
    indexed: Dict[int, dict] = {}
    for row in rows:
        row_id = row.get("id")
        if isinstance(row_id, int):
            indexed[row_id] = row
    return indexed


def is_valid_response_text(response_text: Optional[str]) -> bool:
    if response_text is None:
        return False
    if not isinstance(response_text, str):
        return False
    compact = response_text.strip()
    if not compact:
        return False
    if len(compact) < MIN_VALID_RESPONSE_LENGTH:
        return False
    for pattern in INVALID_RESPONSE_PATTERNS:
        if pattern in compact:
            return False
    return True


def call_model(question: str, model_name: str) -> Optional[str]:
    if model_name == "baichuan-m3-plus":
        return call_baichuan.call_bc(question)
    else:
        return call_gpt5.call_gpt5(question, model_name=model_name)


def build_prompt(dataset_name: str, question_text: str) -> str:
    prefix = DATASET_PROMPT_PREFIX.get(dataset_name, "")
    suffix = DATASET_PROMPT_SUFFIX.get(dataset_name, "")
    return f"{prefix}{question_text}{suffix}"


def call_model_with_retry(
    question: str,
    model_name: str,
    retry_times: int = DEFAULT_RETRY_TIMES,
    retry_sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
) -> Optional[str]:
    last_response: Optional[str] = None

    for attempt in range(1, retry_times + 1):
        print(f"[{model_name}] 第 {attempt}/{retry_times} 次调用")
        last_response = call_model(question, model_name=model_name)
        if is_valid_response_text(last_response):
            return last_response

        print(
            f"[{model_name}] 返回结果无效，"
            f"长度={len((last_response or '').strip()) if isinstance(last_response, str) or last_response is None else 'unknown'}"
        )
        if attempt < retry_times and retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds)

    return last_response


def run_single_dataset(
    dataset_name: str,
    questions: List[dict],
    model_name: str,
    sleep_seconds: float = 0.0,
    overwrite: bool = False,
    retry_times: int = DEFAULT_RETRY_TIMES,
    retry_sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(dataset_name, model_name)

    existing_rows = [] if overwrite else read_existing_rows(output_path)
    existing_by_id = index_rows_by_id(existing_rows)
    results_by_id: Dict[int, dict] = dict(existing_by_id)

    total = len(questions)
    for index, question_item in enumerate(questions, start=1):
        question_id = question_item["id"]
        question_text = question_item["question"]
        prompt_text = build_prompt(dataset_name, question_text)
        existing_row = existing_by_id.get(question_id)

        if existing_row and is_valid_response_text(existing_row.get("response_text")):
            print(f"[{model_name}] {dataset_name} {index}/{total} | 跳过已完成题目 id={question_id}")
            continue

        if existing_row:
            print(f"[{model_name}] {dataset_name} {index}/{total} | 发现无效历史结果，重新请求 id={question_id}")

        print(f"[{model_name}] {dataset_name} {index}/{total} | 开始请求 id={question_id}")
        response_text = call_model_with_retry(
            prompt_text,
            model_name=model_name,
            retry_times=retry_times,
            retry_sleep_seconds=retry_sleep_seconds,
        )

        row = {
            "id": question_id,
            "type": question_item.get("type"),
            "question": question_text,
            "prompt": prompt_text,
            "base_formula_name": question_item.get("base_formula_name"),
            "formula_name": question_item.get("formula_name"),
            "formula": question_item.get("formula"),
            "modified_formula_name": question_item.get("modified_formula_name"),
            "diff_component": question_item.get("diff_component", []),
            "diff_component_text": question_item.get("diff_component_text"),
            "model_name": model_name,
            "response_text": response_text,
        }
        results_by_id[question_id] = row

        ordered_rows = [results_by_id[item["id"]] for item in questions if item["id"] in results_by_id]
        payload = {
            "dataset": dataset_name,
            "model_name": model_name,
            "total_questions": total,
            "completed_questions": len(ordered_rows),
            "results": ordered_rows,
        }
        write_json(output_path, payload)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    final_rows = [results_by_id[item["id"]] for item in questions if item["id"] in results_by_id]
    final_payload = {
        "dataset": dataset_name,
        "model_name": model_name,
        "total_questions": total,
        "completed_questions": len(final_rows),
        "results": final_rows,
    }
    write_json(output_path, final_payload)
    return output_path


def run_dataset_for_models(
    dataset_name: str,
    model_names: List[str],
    sleep_seconds: float = 0.0,
    overwrite: bool = False,
    retry_times: int = DEFAULT_RETRY_TIMES,
    retry_sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
) -> List[Path]:
    dataset_path = DATASET_DIR / f"{dataset_name}_questions.json"
    questions = load_json(dataset_path)
    output_paths: List[Path] = []

    for model_name in model_names:
        print(f"\n=== 开始测试模型：{model_name} | 数据集：{dataset_name} ===")
        output_path = run_single_dataset(
            dataset_name=dataset_name,
            questions=questions,
            model_name=model_name,
            sleep_seconds=sleep_seconds,
            overwrite=overwrite,
            retry_times=retry_times,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        output_paths.append(output_path)
        print(f"=== 测试完成：{output_path} ===")

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 13.build_sq_dataset 中的主观题问题")
    parser.add_argument(
        "--dataset",
        default="dataset1",
        choices=["dataset1", "dataset2"],
        help="待测试的数据集名称，当前支持 dataset1 和 dataset2",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=SUPPORTED_MODELS,
        choices=SUPPORTED_MODELS,
        help="待测试的模型列表，默认跑全部模型",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="每次请求后的等待秒数，默认 0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="是否覆盖已有结果重新运行；默认断点续跑",
    )
    parser.add_argument(
        "--retry-times",
        type=int,
        default=DEFAULT_RETRY_TIMES,
        help="单题请求失败或返回过短时的最大重试次数，默认 3",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=DEFAULT_RETRY_SLEEP_SECONDS,
        help="重试之间的等待秒数，默认 2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = run_dataset_for_models(
        dataset_name=args.dataset,
        model_names=args.models,
        sleep_seconds=args.sleep_seconds,
        overwrite=args.overwrite,
        retry_times=args.retry_times,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )

    print("\n=== 全部任务完成 ===")
    for output_path in output_paths:
        print(f"- {output_path}")


if __name__ == "__main__":
    main()