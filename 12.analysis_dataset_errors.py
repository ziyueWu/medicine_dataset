# -*- coding: UTF-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "11.run_normal_llm_dataset"
OUTPUT_DIR = ROOT / "12.analysis_dataset_errors"

MODEL_NAMES = [
    "gpt-5.4",
    "o3",
    "kimi-k2.5",
    "deepseek-v3.2",
    "gemini-3.1-pro-preview",
    "claude-sonnet-4-6",
    "qwen3.5-plus",
    "zhongjing",
]

MODEL_DISPLAY_NAMES = {
    "gpt-5.4": "GPT-5.4",
    "o3": "O3",
    "kimi-k2.5": "Kimi-K2.5",
    "deepseek-v3.2": "DeepSeek-V3.2",
    "gemini-3.1-pro-preview": "Gemini-3.1-Pro-Preview",
    "claude-sonnet-4-6": "Claude-Sonnet-4.6",
    "qwen3.5-plus": "Qwen3.5-Plus",
    "zhongjing": "仲景",
}


def slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def option_text(options: Dict[str, str], choice: str) -> str:
    if not choice:
        return ""
    return options.get(choice, "")


def normalize_detail_row(row: Dict[str, Any]) -> Dict[str, Any]:
    options = row.get("options", {})
    gold_answer = row.get("gold_answer", row.get("ground_truth"))
    predicted_answer = row.get("predicted_answer", row.get("prediction"))
    is_correct = row.get("is_correct", row.get("correct"))
    gold_answer_text = row.get("gold_answer_text") or option_text(options, gold_answer)

    normalized = {
        "id": row["id"],
        "type": row.get("type", ""),
        "question": row.get("question", ""),
        "options": options,
        "gold_answer": gold_answer,
        "gold_answer_text": gold_answer_text,
        "predicted_answer": predicted_answer,
        "predicted_answer_text": option_text(options, predicted_answer),
        "model_response": row.get("model_response", row.get("model_output")),
        "is_correct": bool(is_correct),
    }

    for extra_key in ["formula_name", "standard_name", "masked_index"]:
        if extra_key in row:
            normalized[extra_key] = row[extra_key]

    return normalized


def load_dataset_model_rows(dataset_name: str, model_name: str) -> List[Dict[str, Any]]:
    model_slug = slugify_model_name(model_name)
    path = INPUT_DIR / f"{dataset_name}_{model_slug}_detail.json"
    if not path.exists():
        raise FileNotFoundError(f"未找到结果文件：{path}")

    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("details"), list):
        rows = payload["details"]
    else:
        raise ValueError(f"无法识别的 detail 文件结构：{path}")

    return [normalize_detail_row(row) for row in rows]


def build_wrong_entry(model_name: str, row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": model_name,
        "model_display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
        "predicted_answer": row["predicted_answer"],
        "predicted_answer_text": row["predicted_answer_text"],
        "gold_answer": row["gold_answer"],
        "gold_answer_text": row["gold_answer_text"],
        "model_response": row["model_response"],
    }


def build_dataset_report(dataset_name: str) -> Dict[str, Any]:
    all_rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
    question_error_map: Dict[int, Dict[str, Any]] = {}
    model_error_breakdown: Dict[str, List[Dict[str, Any]]] = {}
    total_questions = 0

    for model_name in MODEL_NAMES:
        rows = load_dataset_model_rows(dataset_name, model_name)
        all_rows_by_model[model_name] = rows
        model_error_breakdown[model_name] = []
        total_questions = max(total_questions, len(rows))

        for row in rows:
            if row["is_correct"]:
                continue

            wrong_entry = build_wrong_entry(model_name, row)
            question_id = row["id"]
            question_bucket = question_error_map.setdefault(
                question_id,
                {
                    "id": question_id,
                    "type": row["type"],
                    "question": row["question"],
                    "options": row["options"],
                    "gold_answer": row["gold_answer"],
                    "gold_answer_text": row["gold_answer_text"],
                    "error_count": 0,
                    "error_models": [],
                },
            )
            question_bucket["error_count"] += 1
            question_bucket["error_models"].append(wrong_entry)

            model_error_breakdown[model_name].append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "gold_answer_text": row["gold_answer_text"],
                    "predicted_answer": row["predicted_answer"],
                    "predicted_answer_text": row["predicted_answer_text"],
                    "model_response": row["model_response"],
                }
            )

    ranked_questions = sorted(
        question_error_map.values(),
        key=lambda item: (-item["error_count"], item["id"]),
    )
    top10_questions = ranked_questions[:10]

    model_error_counts = {
        model_name: len(model_error_breakdown[model_name])
        for model_name in MODEL_NAMES
    }

    return {
        "dataset": dataset_name,
        "total_questions": total_questions,
        "model_count": len(MODEL_NAMES),
        "model_error_counts": model_error_counts,
        "top10_error_questions": top10_questions,
        "all_ranked_error_questions": ranked_questions,
        "model_error_breakdown": model_error_breakdown,
    }


def write_top10_csv(path: Path, top10_questions: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "question_id",
                "type",
                "question",
                "gold_answer",
                "gold_answer_text",
                "error_count",
                "wrong_models",
            ]
        )
        for item in top10_questions:
            wrong_models = " | ".join(
                f"{detail['model_display_name']}:{detail['predicted_answer']}->{detail['predicted_answer_text']}"
                for detail in item["error_models"]
            )
            writer.writerow(
                [
                    item["id"],
                    item["type"],
                    item["question"],
                    item["gold_answer"],
                    item["gold_answer_text"],
                    item["error_count"],
                    wrong_models,
                ]
            )


def write_model_errors_csv(path: Path, model_error_breakdown: Dict[str, List[Dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "model_display_name",
                "question_id",
                "type",
                "question",
                "gold_answer",
                "gold_answer_text",
                "predicted_answer",
                "predicted_answer_text",
                "model_response",
            ]
        )
        for model_name in MODEL_NAMES:
            for item in model_error_breakdown[model_name]:
                writer.writerow(
                    [
                        model_name,
                        MODEL_DISPLAY_NAMES.get(model_name, model_name),
                        item["id"],
                        item["type"],
                        item["question"],
                        item["gold_answer"],
                        item["gold_answer_text"],
                        item["predicted_answer"],
                        item["predicted_answer_text"],
                        item["model_response"],
                    ]
                )


def build_dataset_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    dataset_name = report["dataset"]
    total_questions = report["total_questions"]
    model_count = report["model_count"]

    lines.append(f"# {dataset_name} 错题分析")
    lines.append("")
    lines.append(f"- 题目总数：{total_questions}")
    lines.append(f"- 模型数量：{model_count}")
    lines.append("")

    lines.append("## 各模型错题数")
    lines.append("")
    lines.append("| 模型 | 错题数 |")
    lines.append("| --- | ---: |")
    for model_name in MODEL_NAMES:
        lines.append(
            f"| {MODEL_DISPLAY_NAMES.get(model_name, model_name)} | {report['model_error_counts'][model_name]} |"
        )
    lines.append("")

    lines.append("## 错误次数最多的前 10 道题")
    lines.append("")
    for index, item in enumerate(report["top10_error_questions"], start=1):
        lines.append(f"### TOP {index}｜题号 {item['id']}｜错误次数 {item['error_count']}/{model_count}")
        lines.append("")
        lines.append(f"- 类型：{item['type']}")
        lines.append(f"- 题目：{item['question']}")
        lines.append(f"- 正确答案：{item['gold_answer']}（{item['gold_answer_text']}）")
        lines.append("- 错误模型：")
        for detail in item["error_models"]:
            pred_label = detail["predicted_answer"] or "未识别"
            pred_text = detail["predicted_answer_text"] or ""
            lines.append(
                f"  - {detail['model_display_name']}：预测 {pred_label}（{pred_text}）"
            )
        lines.append("")

    lines.append("## 各模型错在了哪里")
    lines.append("")
    for model_name in MODEL_NAMES:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        errors = report["model_error_breakdown"][model_name]
        lines.append(f"### {display_name}（共 {len(errors)} 题）")
        lines.append("")
        if not errors:
            lines.append("- 无错题")
            lines.append("")
            continue

        for item in errors:
            pred_label = item["predicted_answer"] or "未识别"
            pred_text = item["predicted_answer_text"] or ""
            lines.append(
                f"- 题号 {item['id']}｜{item['type']}｜预测 {pred_label}（{pred_text}）｜正确 {item['gold_answer']}（{item['gold_answer_text']}）"
            )
        lines.append("")

    return "\n".join(lines)


def build_overall_summary(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    dataset_error_overview = []
    model_total_errors = {model_name: 0 for model_name in MODEL_NAMES}

    for report in reports:
        dataset_error_overview.append(
            {
                "dataset": report["dataset"],
                "top10_question_ids": [item["id"] for item in report["top10_error_questions"]],
                "top10_error_counts": [item["error_count"] for item in report["top10_error_questions"]],
                "model_error_counts": report["model_error_counts"],
            }
        )
        for model_name, error_count in report["model_error_counts"].items():
            model_total_errors[model_name] += error_count

    return {
        "datasets": dataset_error_overview,
        "model_total_errors": model_total_errors,
    }


def build_overall_markdown(summary: Dict[str, Any]) -> str:
    lines = ["# 三套数据集错题总览", ""]
    lines.append("## 各模型总错题数")
    lines.append("")
    lines.append("| 模型 | 总错题数 |")
    lines.append("| --- | ---: |")
    for model_name in MODEL_NAMES:
        lines.append(
            f"| {MODEL_DISPLAY_NAMES.get(model_name, model_name)} | {summary['model_total_errors'][model_name]} |"
        )
    lines.append("")

    lines.append("## 各数据集 Top10 错题概览")
    lines.append("")
    for item in summary["datasets"]:
        lines.append(f"### {item['dataset']}")
        lines.append("")
        lines.append("| 排名 | 题号 | 错误次数 |")
        lines.append("| --- | ---: | ---: |")
        for index, (question_id, error_count) in enumerate(
            zip(item["top10_question_ids"], item["top10_error_counts"]),
            start=1,
        ):
            lines.append(f"| {index} | {question_id} | {error_count} |")
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 11.run_normal_llm_dataset 中 8 个大模型在 3 套数据集上的错题情况")
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="指定要分析的数据集编号，例如 --datasets 1 3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    for dataset_idx in args.datasets:
        dataset_name = f"dataset{dataset_idx}"
        report = build_dataset_report(dataset_name)
        reports.append(report)

        write_json(OUTPUT_DIR / f"{dataset_name}_error_report.json", report)
        write_text(OUTPUT_DIR / f"{dataset_name}_error_report.md", build_dataset_markdown(report))
        write_top10_csv(OUTPUT_DIR / f"{dataset_name}_top10_errors.csv", report["top10_error_questions"])
        write_model_errors_csv(OUTPUT_DIR / f"{dataset_name}_model_errors.csv", report["model_error_breakdown"])

        print(f"[完成] {dataset_name} 错题分析已输出到 {OUTPUT_DIR}")

    overall_summary = build_overall_summary(reports)
    write_json(OUTPUT_DIR / "overall_error_summary.json", overall_summary)
    write_text(OUTPUT_DIR / "overall_error_summary.md", build_overall_markdown(overall_summary))
    print(f"[完成] 总览已输出到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()