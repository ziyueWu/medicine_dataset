# -*- coding: UTF-8 -*-

import json
import re
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "15.evaluate_sq_result"
OUTPUT_DIR = ROOT / "16.statistics_sq_result"
OUTPUT_JSON_PATH = OUTPUT_DIR / "sq_result_statistics.json"
ERROR_OUTPUT_DIR = OUTPUT_DIR / "errors"

MODEL_DISPLAY_NAMES = {
    "gpt_5_4": "GPT-5.4",
    "o3": "O3",
    "kimi_k2_5": "Kimi-K2.5",
    "deepseek_v3_2": "DeepSeek-V3.2",
    "gemini_3_1_pro_preview": "Gemini-3.1-Pro-Preview",
    "claude_sonnet_4_6": "Claude-Sonnet-4.6",
    "qwen3_5_plus": "Qwen3.5-Plus",
    "zhongjing": "仲景",
    "baichuan_m3_plus": "百川医疗大模型",
}

MODEL_ORDER = [
    "gpt_5_4",
    "o3",
    "kimi_k2_5",
    "deepseek_v3_2",
    "gemini_3_1_pro_preview",
    "claude_sonnet_4_6",
    "qwen3_5_plus",
    "zhongjing",
    "baichuan_m3_plus"
]

DATASET_ORDER = ["dataset1", "dataset2"]

PLOT_SPECS = [
    ("average", "大模型中医主观题能力评测结果", "sq_average_total_scores.png"),
    ("dataset1", "大模型加减方解释能力评测结果", "dataset1_total_scores.png"),
    ("dataset2", "大模型方剂生成能力评测结果", "dataset2_total_scores.png"),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    if plt is None:
        return
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Arial Unicode MS",
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def round_half_up(value: float, digits: int = 2) -> float:
    quantize_exp = "1." + "0" * digits
    return float(Decimal(str(value)).quantize(Decimal(quantize_exp), rounding=ROUND_HALF_UP))


def parse_dataset_and_model(file_path: Path) -> Optional[Tuple[str, str]]:
    match = re.fullmatch(r"(dataset[12])_(.+)_evaluations\.json", file_path.name)
    if not match:
        return None
    dataset_name = match.group(1)
    model_slug = match.group(2)
    return dataset_name, model_slug


def normalize_question_score(score_llm_all: float) -> float:
    return round_half_up(float(score_llm_all) / 18.0, 2)


def calculate_total_score(rows: List[dict]) -> float:
    total_score = 0.0
    for row in rows:
        total_score += normalize_question_score(row.get("score_llm_all", 0.0))
    return round_half_up(total_score, 2)


def extract_evaluation_reason(evaluation_json: Optional[dict]) -> str:
    """尽量从评测 JSON 中提取原因字段；当前评测文件通常只含分数，可能为空。"""
    if not isinstance(evaluation_json, dict):
        return ""

    for key in ("reason", "reasons", "explanation", "analysis", "comment", "comments", "评价原因", "原因"):
        value = evaluation_json.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    return ""


def build_evaluator_details(model_evaluations: dict) -> Dict[str, dict]:
    evaluator_details: Dict[str, dict] = {}
    if not isinstance(model_evaluations, dict):
        return evaluator_details

    for evaluator_name, evaluation in model_evaluations.items():
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        evaluation_json = evaluation.get("evaluation_json")
        evaluator_details[evaluator_name] = {
            "score_total": float(evaluation.get("score_total", 0.0) or 0.0),
            "confidence": float(evaluation.get("confidence", 0.0) or 0.0),
            "score_items": evaluation_json,
            "reason": extract_evaluation_reason(evaluation_json),
            "evaluation_text": evaluation.get("evaluation_text", ""),
        }

    return evaluator_details


def collect_low_score_questions(top_n: int = 10) -> Dict[str, Any]:
    """
    统计每套主观题中平均得分最低的前 top_n 道题。

    排序口径：对同一 dataset、同一 question id，汇总所有回答模型的 score_llm_all，
    先按每个模型的 score_llm_all/18 归一化为 0-1 得分，再计算跨模型平均分；平均分越低，
    表示该题在各大模型上的整体表现越差。输出中保留每个回答模型的原始回答、总分、
    各评测模型评分明细及可提取到的评分原因。
    """
    grouped: Dict[str, Dict[str, dict]] = {dataset_name: {} for dataset_name in DATASET_ORDER}

    for file_path in sorted(INPUT_DIR.glob("dataset*_evaluations.json")):
        parsed = parse_dataset_and_model(file_path)
        if not parsed:
            continue

        dataset_name, model_slug = parsed
        payload = load_json(file_path)
        rows = payload.get("results", []) if isinstance(payload, dict) else []
        response_model_name = payload.get("response_model_name", format_model_label(model_slug)) if isinstance(payload, dict) else format_model_label(model_slug)

        for row in rows:
            question_id = str(row.get("id", ""))
            if not question_id:
                continue

            question_record = grouped.setdefault(dataset_name, {}).setdefault(
                question_id,
                {
                    "id": row.get("id"),
                    "question": row.get("question", ""),
                    "base_formula_name": row.get("base_formula_name", ""),
                    "modified_formula_name": row.get("modified_formula_name", ""),
                    "base_formula_cure": row.get("base_formula_cure", ""),
                    "modified_formula_cure": row.get("modified_formula_cure", ""),
                    "model_results": {},
                },
            )

            raw_score = float(row.get("score_llm_all", 0.0) or 0.0)
            question_record["model_results"][model_slug] = {
                "model_display_name": format_model_label(model_slug),
                "response_model_name": row.get("response_model_name", response_model_name),
                "score_llm_all": round_half_up(raw_score, 4),
                "normalized_score": normalize_question_score(raw_score),
                "response_text": row.get("response_text", ""),
                "evaluator_scores": build_evaluator_details(row.get("model_evaluations", {})),
            }

    result: Dict[str, Any] = {}
    for dataset_name in DATASET_ORDER:
        dataset_records = []
        for question_record in grouped.get(dataset_name, {}).values():
            model_results = question_record.get("model_results", {})
            normalized_scores = [item.get("normalized_score", 0.0) for item in model_results.values()]
            raw_scores = [item.get("score_llm_all", 0.0) for item in model_results.values()]
            if not normalized_scores:
                continue

            sorted_model_results = {
                model_name: model_results[model_name]
                for model_name in sort_model_names({model_name: 0 for model_name in model_results.keys()})
            }
            question_record["model_results"] = sorted_model_results
            question_record["model_count"] = len(normalized_scores)
            question_record["average_normalized_score"] = round_half_up(sum(normalized_scores) / len(normalized_scores), 4)
            question_record["average_score_llm_all"] = round_half_up(sum(raw_scores) / len(raw_scores), 4)
            question_record["min_normalized_score"] = round_half_up(min(normalized_scores), 4)
            question_record["min_score_llm_all"] = round_half_up(min(raw_scores), 4)
            dataset_records.append(question_record)

        dataset_records.sort(key=lambda item: (item["average_normalized_score"], item["min_normalized_score"], item.get("id", 0)))
        result[dataset_name] = dataset_records[:top_n]

    return {
        "description": "每套主观题按题目聚合所有大模型结果，以各模型 score_llm_all/18 的平均值从低到高排序，取最低前10题；每题记录各大模型回答、评分结果、评测模型明细及可提取到的评分原因。",
        "note": "当前评测文件的 evaluation_json 多数仅包含分数与置信度，未显式包含原因；若无原因字段，reason 为空，并保留原始 evaluation_text 便于追溯。",
        "datasets": result,
    }


def write_low_score_csv(low_score_statistics: dict) -> List[Path]:
    csv_paths: List[Path] = []
    fieldnames = [
        "dataset",
        "rank",
        "question_id",
        "question",
        "average_normalized_score",
        "average_score_llm_all",
        "min_normalized_score",
        "model",
        "model_display_name",
        "model_normalized_score",
        "model_score_llm_all",
        "evaluator",
        "evaluator_score_total",
        "evaluator_confidence",
        "reason",
        "evaluation_text",
        "response_text",
    ]

    for dataset_name, records in low_score_statistics.get("datasets", {}).items():
        csv_path = ERROR_OUTPUT_DIR / f"{dataset_name}_lowest_top10_questions.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rank, record in enumerate(records, start=1):
                for model_name, model_result in record.get("model_results", {}).items():
                    evaluator_scores = model_result.get("evaluator_scores", {})
                    if not evaluator_scores:
                        writer.writerow(
                            {
                                "dataset": dataset_name,
                                "rank": rank,
                                "question_id": record.get("id", ""),
                                "question": record.get("question", ""),
                                "average_normalized_score": record.get("average_normalized_score", ""),
                                "average_score_llm_all": record.get("average_score_llm_all", ""),
                                "min_normalized_score": record.get("min_normalized_score", ""),
                                "model": model_name,
                                "model_display_name": model_result.get("model_display_name", ""),
                                "model_normalized_score": model_result.get("normalized_score", ""),
                                "model_score_llm_all": model_result.get("score_llm_all", ""),
                                "response_text": model_result.get("response_text", ""),
                            }
                        )
                        continue

                    for evaluator_name, evaluator_result in evaluator_scores.items():
                        writer.writerow(
                            {
                                "dataset": dataset_name,
                                "rank": rank,
                                "question_id": record.get("id", ""),
                                "question": record.get("question", ""),
                                "average_normalized_score": record.get("average_normalized_score", ""),
                                "average_score_llm_all": record.get("average_score_llm_all", ""),
                                "min_normalized_score": record.get("min_normalized_score", ""),
                                "model": model_name,
                                "model_display_name": model_result.get("model_display_name", ""),
                                "model_normalized_score": model_result.get("normalized_score", ""),
                                "model_score_llm_all": model_result.get("score_llm_all", ""),
                                "evaluator": evaluator_name,
                                "evaluator_score_total": evaluator_result.get("score_total", ""),
                                "evaluator_confidence": evaluator_result.get("confidence", ""),
                                "reason": evaluator_result.get("reason", ""),
                                "evaluation_text": evaluator_result.get("evaluation_text", ""),
                                "response_text": model_result.get("response_text", ""),
                            }
                        )
        csv_paths.append(csv_path)

    return csv_paths


def generate_low_score_error_reports() -> List[Path]:
    low_score_statistics = collect_low_score_questions(top_n=10)
    json_path = ERROR_OUTPUT_DIR / "lowest_top10_questions.json"
    write_json(json_path, low_score_statistics)
    csv_paths = write_low_score_csv(low_score_statistics)
    return [json_path] + csv_paths


def collect_dataset_scores() -> Dict[str, Dict[str, float]]:
    dataset_scores: Dict[str, Dict[str, float]] = {dataset_name: {} for dataset_name in DATASET_ORDER}

    for file_path in sorted(INPUT_DIR.glob("dataset*_evaluations.json")):
        parsed = parse_dataset_and_model(file_path)
        if not parsed:
            continue

        dataset_name, model_slug = parsed
        payload = load_json(file_path)
        rows = payload.get("results", []) if isinstance(payload, dict) else []
        dataset_scores.setdefault(dataset_name, {})[model_slug] = calculate_total_score(rows)

    return dataset_scores


def build_average_scores(dataset_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    average_scores: Dict[str, float] = {}
    model_names = sorted({model for scores in dataset_scores.values() for model in scores.keys()})

    for model_name in model_names:
        values = [dataset_scores[dataset_name][model_name] for dataset_name in DATASET_ORDER if model_name in dataset_scores.get(dataset_name, {})]
        if values:
            average_scores[model_name] = round_half_up(sum(values) / len(values), 2)

    return average_scores


def sort_model_names(score_map: Dict[str, float]) -> List[str]:
    remaining = [name for name in score_map.keys() if name not in MODEL_ORDER]
    return [name for name in MODEL_ORDER if name in score_map] + sorted(remaining)


def format_model_label(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def calculate_axis_bounds(values: List[float], max_value: float, tick_step: int) -> Tuple[float, float, List[float]]:
    if not values:
        ticks = list(range(0, int(max_value) + 1, tick_step))
        return 0.0, float(max_value), [float(tick) for tick in ticks]

    min_value = min(values)
    axis_min = 0.0
    if min_value >= tick_step:
        axis_min = float((int(min_value) // tick_step) * tick_step)

    axis_max = float(max_value)
    if axis_min >= axis_max:
        axis_min = max(0.0, axis_max - tick_step)

    ticks = list(range(int(axis_min), int(axis_max) + 1, tick_step))
    if not ticks or ticks[-1] != int(axis_max):
        ticks.append(int(axis_max))
    return axis_min, axis_max, [float(tick) for tick in ticks]


def plot_bar_chart(title: str, score_map: Dict[str, float], save_path: Path) -> Path:
    if plt is None:
        return plot_bar_chart_svg(title, score_map, save_path.with_suffix(".svg"))

    configure_matplotlib()
    model_names = sort_model_names(score_map)
    labels = [format_model_label(name) for name in model_names]
    values = [score_map[name] for name in model_names]
    axis_min, axis_max, _ = calculate_axis_bounds(values, max_value=50, tick_step=5)

    width = max(12, len(labels) * 1.4)
    fig, ax = plt.subplots(figsize=(width, 8))
    bars = ax.bar(labels, values, color="#4C78A8", width=0.62)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("模型名称", fontsize=12)
    ax.set_ylabel("总分", fontsize=12)
    ax.set_ylim(axis_min, axis_max)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", labelsize=11)
    plt.xticks(rotation=20, ha="right")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.5, axis_max - 0.3),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_bar_chart_svg(title: str, score_map: Dict[str, float], save_path: Path) -> Path:
    model_names = sort_model_names(score_map)
    labels = [format_model_label(name) for name in model_names]
    values = [score_map[name] for name in model_names]
    axis_min, axis_max, y_ticks = calculate_axis_bounds(values, max_value=50, tick_step=5)
    axis_span = max(axis_max - axis_min, 1)

    width = 1280
    height = 760
    margin_left = 90
    margin_right = 40
    margin_top = 80
    margin_bottom = 170
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    baseline_y = margin_top + plot_height
    bar_width = plot_width / max(len(labels) * 1.8, 1)
    step = plot_width / max(len(labels), 1)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: PingFang SC, Hiragino Sans GB, Heiti SC, Arial Unicode MS, SimHei, Microsoft YaHei, sans-serif; }</style>',
        f'<text x="{width / 2}" y="40" text-anchor="middle" font-size="26" font-weight="bold">{title}</text>',
        f'<text x="30" y="{margin_top + plot_height / 2}" text-anchor="middle" font-size="18" transform="rotate(-90 30 {margin_top + plot_height / 2})">总分</text>',
        f'<text x="{margin_left + plot_width / 2}" y="{height - 30}" text-anchor="middle" font-size="18">模型名称</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{baseline_y}" stroke="#333" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{baseline_y}" x2="{margin_left + plot_width}" y2="{baseline_y}" stroke="#333" stroke-width="2"/>',
    ]

    for tick in y_ticks:
        y = baseline_y - ((tick - axis_min) / axis_span) * plot_height
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#D9D9D9" stroke-dasharray="4 4"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 12}" y="{y + 6}" text-anchor="end" font-size="14" fill="#333">{tick:.0f}</text>'
        )

    for idx, (label, value) in enumerate(zip(labels, values)):
        center_x = margin_left + step * (idx + 0.5)
        x = center_x - bar_width / 2
        bar_height = max(0.0, (value - axis_min) / axis_span) * plot_height
        y = baseline_y - bar_height

        svg_parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#4C78A8" rx="4"/>'
        )
        svg_parts.append(
            f'<text x="{center_x:.2f}" y="{max(y - 10, margin_top + 15):.2f}" text-anchor="middle" font-size="15" fill="#333">{value:.2f}</text>'
        )
        svg_parts.append(
            f'<text x="{center_x:.2f}" y="{baseline_y + 28}" text-anchor="middle" font-size="14" fill="#333">{label}</text>'
        )

    svg_parts.append("</svg>")
    save_path.write_text("\n".join(svg_parts), encoding="utf-8")
    return save_path


def generate_statistics() -> dict:
    dataset_scores = collect_dataset_scores()
    average_scores = build_average_scores(dataset_scores)

    result = {
        "description": "每题分数按 score_llm_all/18 归一化后四舍五入保留两位小数，再累加得到每个数据集下每个模型的总分（满分 50）",
        "datasets": dataset_scores,
        "average": average_scores,
    }
    write_json(OUTPUT_JSON_PATH, result)
    return result


def generate_charts(statistics: dict) -> List[Path]:
    generated_paths: List[Path] = []
    score_sources = {
        "average": statistics.get("average", {}),
        "dataset1": statistics.get("datasets", {}).get("dataset1", {}),
        "dataset2": statistics.get("datasets", {}).get("dataset2", {}),
    }

    for score_key, title, filename in PLOT_SPECS:
        score_map = score_sources.get(score_key, {})
        if not score_map:
            continue
        generated_paths.append(plot_bar_chart(title=title, score_map=score_map, save_path=OUTPUT_DIR / filename))

    return generated_paths


def main() -> None:
    ensure_output_dir()
    statistics = generate_statistics()
    chart_paths = generate_charts(statistics)
    error_report_paths = generate_low_score_error_reports()

    print("统计完成：")
    print(f"- JSON: {OUTPUT_JSON_PATH}")
    for chart_path in chart_paths:
        print(f"- 图表: {chart_path}")
    for error_report_path in error_report_paths:
        print(f"- 低分题统计: {error_report_path}")


if __name__ == "__main__":
    main()