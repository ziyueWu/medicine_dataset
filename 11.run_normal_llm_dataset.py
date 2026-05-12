# -*- coding: UTF-8 -*-

import argparse
import html
import json
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None

import call_gpt5
import call_baichuan


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "10.build_tcmllm_dataset"
OUTPUT_DIR = ROOT / "11.run_normal_llm_dataset"

SUPPORTED_MODELS = {
    "gpt-5.4": "ready",
    "o3": "ready",
    "kimi-k2.5": "ready",
    "deepseek-v3.2": "ready",
    "gemini-3.1-pro-preview": "ready",
    "claude-sonnet-4-6": "ready",
    "qwen3.5-plus": "ready",
    "baichuan-m3-plus": "call_bc"
}

MODEL_DISPLAY_NAMES = {
    "gpt-5.4": "GPT-5.4",
    "o3": "O3",
    "kimi-k2.5": "Kimi-K2.5",
    "deepseek-v3.2": "DeepSeek-V3.2",
    "gemini-3.1-pro-preview": "Gemini-3.1-Pro-Preview",
    "claude-sonnet-4-6": "Claude-Sonnet-4.6",
    "qwen3.5-plus": "Qwen3.5-Plus",
    "zhongjing": "仲景",
    "baichuan-m3-plus": "百川"
}

PLOT_MODEL_ORDER = list(SUPPORTED_MODELS.keys()) + ["zhongjing"]

MODEL_CATEGORY_LABELS = {
    "gpt-5.4": "通用大模型",
    "o3": "通用大模型",
    "kimi-k2.5": "通用大模型",
    "deepseek-v3.2": "通用大模型",
    "gemini-3.1-pro-preview": "通用大模型",
    "claude-sonnet-4-6": "通用大模型",
    "qwen3.5-plus": "通用大模型",
    "zhongjing": "中医药大模型",
    "baichuan-m3-plus": "医疗大模型"
}

PLOT_SPECS = [
    ("dataset1", "大模型知识匹配能力测评结果"),
    ("dataset2", "大模型方剂组成补全能力测评结果"),
    ("dataset3", "大模型中药名称别名识别能力测评结果"),
    ("overall", "大模型中医能力评测结果"),
]


def load_json(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


def build_prompt(question_item: dict) -> str:
    options = question_item["options"]
    return f"""你是一名中医药选择题答题助手。请认真阅读题目并作答。

要求：
1. 你只能从 A、B、C、D 四个选项中选择一个最合适的答案。
2. 请仅输出一个大写字母，例如：A
3. 不要输出解释、理由、标点、引号、前后缀或其它任何内容。

题型：{question_item['type']}
题目：{question_item['question']}

A. {options['A']}
B. {options['B']}
C. {options['C']}
D. {options['D']}
"""


def extract_choice(response_text: Optional[str]) -> Optional[str]:
    if not response_text:
        return None

    text = str(response_text).strip()
    if not text:
        return None

    compact = text.replace("`", "").replace("*", "").strip()
    if compact in {"A", "B", "C", "D"}:
        return compact

    patterns = [
        r"(?:答案|选择|选项)\s*[:：]?\s*([ABCD])\b",
        r"^\s*([ABCD])\b",
        r"\b([ABCD])\b",
        r"([ABCD])",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def call_model(model_name: str, prompt: str) -> Optional[str]:
    status = SUPPORTED_MODELS.get(model_name)
    if status is None:
        raise ValueError(f"不支持的模型：{model_name}")
    if status != "ready":
        if status == "call_bc":
            return call_baichuan.call_bc(prompt)
        else:
            raise NotImplementedError(f"模型 {model_name} 的调用逻辑暂未接入")

    return call_gpt5.call_gpt5(prompt, model_name=model_name)


def run_single_dataset(
    model_name: str,
    dataset_name: str,
    questions: List[dict],
    answers: List[dict],
    sleep_seconds: float = 0.0,
) -> Tuple[dict, List[dict]]:
    answer_map = {item["id"]: item for item in answers}
    result_rows: List[dict] = []
    correct_count = 0

    for index, question in enumerate(questions, start=1):
        answer = answer_map[question["id"]]
        prompt = build_prompt(question)
        raw_response = call_model(model_name, prompt)
        pred_choice = extract_choice(raw_response)
        is_correct = pred_choice == answer["answer"]
        if is_correct:
            correct_count += 1

        row = {
            "id": question["id"],
            "type": question["type"],
            "question": question["question"],
            "options": question["options"],
            "gold_answer": answer["answer"],
            "gold_answer_text": answer["answer_text"],
            "model_response": raw_response,
            "predicted_answer": pred_choice,
            "is_correct": is_correct,
        }

        for extra_key in ["formula_name", "standard_name", "masked_index"]:
            if extra_key in answer:
                row[extra_key] = answer[extra_key]

        result_rows.append(row)
        print(
            f"[{model_name}] {dataset_name} {index}/{len(questions)} | "
            f"预测={pred_choice} | 标准={answer['answer']} | 正确={is_correct}"
        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    total = len(questions)
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "total": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total else 0.0,
    }
    return summary, result_rows


def build_overall_summary_from_saved_files(model_name: str, dataset_ids: List[int]) -> dict:
    model_slug = slugify_model_name(model_name)
    dataset_summaries = []
    all_total = 0
    all_correct = 0

    for dataset_idx in dataset_ids:
        dataset_name = f"dataset{dataset_idx}"
        summary_path = OUTPUT_DIR / f"{dataset_name}_{model_slug}_summary.json"
        summary = read_json_if_exists(summary_path)
        if not summary:
            continue
        dataset_summaries.append(summary)
        all_total += summary["total"]
        all_correct += summary["correct"]

    aggregate = {
        "model": model_name,
        "datasets": dataset_summaries,
        "overall": {
            "total": all_total,
            "correct": all_correct,
            "accuracy": all_correct / all_total if all_total else 0.0,
        },
    }
    write_json(OUTPUT_DIR / f"overall_{model_slug}_summary.json", aggregate)
    return aggregate


def load_accuracy_from_summary(result_type: str, model_name: str) -> float:
    model_slug = slugify_model_name(model_name)
    summary_path = OUTPUT_DIR / f"{result_type}_{model_slug}_summary.json"
    summary = read_json_if_exists(summary_path)
    if model_name == "zhongjing":
        overall_summary = summary if summary and "overall_accuracy" in summary else read_json_if_exists(OUTPUT_DIR / "overall_zhongjing_summary.json")
        if overall_summary:
            if result_type == "overall":
                return float(overall_summary["overall_accuracy"])
            metric_key = f"{result_type}_accuracy"
            if metric_key in overall_summary:
                return float(overall_summary[metric_key])

    if not summary:
        raise FileNotFoundError(f"未找到结果文件：{summary_path}")

    if result_type == "overall":
        return float(summary["overall"]["accuracy"])
    return float(summary["accuracy"])


def format_model_label(model_name: str) -> str:
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    if model_name == "gemini-3.1-pro-preview":
        display_name = "Gemini-3.1-Pro\nPreview"
    category_label = MODEL_CATEGORY_LABELS.get(model_name)
    return f"{display_name}\n{category_label}" if category_label else display_name


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


def plot_accuracy_bar_chart(title: str, model_names: List[str], accuracies: List[float], save_path: Path) -> None:
    if plt is None:
        return plot_accuracy_bar_chart_svg(title, model_names, accuracies, save_path.with_suffix(".svg"))

    configure_matplotlib()

    labels = [format_model_label(name) for name in model_names]
    values = [acc * 100 for acc in accuracies]
    colors = ["#4C78A8" if MODEL_CATEGORY_LABELS.get(name) == "通用大模型" else "#E45756" for name in model_names]
    axis_min, axis_max, _ = calculate_axis_bounds(values, max_value=100, tick_step=10)

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(labels, values, color=colors[: len(labels)], width=0.6)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("大模型名称", fontsize=12)
    ax.set_ylabel("准确率（%）", fontsize=12)
    ax.set_ylim(axis_min, axis_max)
    ax.tick_params(axis="x", labelsize=11)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 1, axis_max - 0.8),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#4C78A8", label="通用大模型"),
        Patch(facecolor="#E45756", label="中医药大模型"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_bar_chart_svg(title: str, model_names: List[str], accuracies: List[float], save_path: Path) -> None:
    labels = [format_model_label(name) for name in model_names]
    values = [acc * 100 for acc in accuracies]
    colors = ["#4C78A8" if MODEL_CATEGORY_LABELS.get(name) == "通用大模型" else "#E45756" for name in model_names]
    axis_min, axis_max, y_ticks = calculate_axis_bounds(values, max_value=100, tick_step=10)
    axis_span = max(axis_max - axis_min, 1)

    width = 1280
    height = 760
    margin_left = 90
    margin_right = 40
    margin_top = 80
    margin_bottom = 190
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    baseline_y = margin_top + plot_height
    bar_width = plot_width / max(len(labels) * 1.8, 1)
    step = plot_width / max(len(labels), 1)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: PingFang SC, Hiragino Sans GB, Heiti SC, Arial Unicode MS, SimHei, Microsoft YaHei, sans-serif; }</style>',
        f'<text x="{width / 2}" y="40" text-anchor="middle" font-size="26" font-weight="bold">{html.escape(title)}</text>',
        f'<text x="28" y="{margin_top + plot_height / 2}" text-anchor="middle" font-size="18" transform="rotate(-90 28 {margin_top + plot_height / 2})">准确率（%）</text>',
        f'<text x="{margin_left + plot_width / 2}" y="{height - 30}" text-anchor="middle" font-size="18">大模型名称</text>',
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
        color = colors[idx % len(colors)]

        svg_parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" rx="4"/>'
        )
        svg_parts.append(
            f'<text x="{center_x:.2f}" y="{max(y - 10, margin_top + 15):.2f}" text-anchor="middle" font-size="15" fill="#333">{value:.1f}%</text>'
        )

        label_lines = label.split("\n")
        label_y = baseline_y + 28
        svg_parts.append(f'<text x="{center_x:.2f}" y="{label_y}" text-anchor="middle" font-size="14" fill="#333">')
        for line_idx, line in enumerate(label_lines):
            dy = 0 if line_idx == 0 else 20
            svg_parts.append(f'<tspan x="{center_x:.2f}" dy="{dy}">{html.escape(line)}</tspan>')
        svg_parts.append("</text>")

    legend_x = width - 220
    legend_y = 70
    legend_items = [("#4C78A8", "通用大模型"), ("#E45756", "中医药大模型")]
    for idx, (legend_color, legend_label) in enumerate(legend_items):
        y = legend_y + idx * 30
        svg_parts.append(f'<rect x="{legend_x}" y="{y - 12}" width="18" height="18" fill="{legend_color}" rx="3"/>')
        svg_parts.append(
            f'<text x="{legend_x + 28}" y="{y + 2}" text-anchor="start" font-size="15" fill="#333">{legend_label}</text>'
        )

    svg_parts.append("</svg>")
    save_path.write_text("\n".join(svg_parts), encoding="utf-8")


def generate_comparison_charts() -> List[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_paths: List[Path] = []
    model_names = PLOT_MODEL_ORDER
    for result_type, chart_title in PLOT_SPECS:
        accuracies = [load_accuracy_from_summary(result_type, model_name) for model_name in model_names]
        save_path = OUTPUT_DIR / f"{chart_title}.png"
        plot_accuracy_bar_chart(
            title=chart_title,
            model_names=model_names,
            accuracies=accuracies,
            save_path=save_path,
        )
        generated_paths.append(save_path if plt is not None else save_path.with_suffix(".svg"))

    return generated_paths


def run_all_datasets(model_name: str, dataset_ids: List[int], sleep_seconds: float = 0.0) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = slugify_model_name(model_name)

    for dataset_idx in dataset_ids:
        dataset_name = f"dataset{dataset_idx}"
        questions = load_json(DATASET_DIR / f"{dataset_name}_questions.json")
        answers = load_json(DATASET_DIR / f"{dataset_name}_answers.json")

        summary, rows = run_single_dataset(
            model_name=model_name,
            dataset_name=dataset_name,
            questions=questions,
            answers=answers,
            sleep_seconds=sleep_seconds,
        )

        write_json(OUTPUT_DIR / f"{dataset_name}_{model_slug}_detail.json", rows)
        write_json(OUTPUT_DIR / f"{dataset_name}_{model_slug}_summary.json", summary)

    return build_overall_summary_from_saved_files(model_name=model_name, dataset_ids=[1, 2, 3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测常规大模型在 TCM LLM 数据集上的选择题表现")
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        choices=list(SUPPORTED_MODELS.keys()),
        help="待评测的模型名称",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="每次请求后的等待秒数，默认 0",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="指定运行哪些数据集，例如 --datasets 3 表示只跑 dataset3",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="只读取已有 summary 结果并生成对比柱状图，不重新调用模型",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        chart_paths = generate_comparison_charts()
        print("\n=== 柱状图生成完成 ===")
        for chart_path in chart_paths:
            print(f"- {chart_path}")
        return

    aggregate = run_all_datasets(
        model_name=args.model,
        dataset_ids=args.datasets,
        sleep_seconds=args.sleep_seconds,
    )

    print("\n=== 评测完成 ===")
    for dataset_summary in aggregate["datasets"]:
        print(
            f"- {dataset_summary['dataset']}: "
            f"{dataset_summary['correct']}/{dataset_summary['total']} "
            f"= {dataset_summary['accuracy']:.2%}"
        )
    print(
        f"- overall: {aggregate['overall']['correct']}/{aggregate['overall']['total']} "
        f"= {aggregate['overall']['accuracy']:.2%}"
    )

    chart_paths = generate_comparison_charts()
    for chart_path in chart_paths:
        print(f"- 图表输出: {chart_path}")
    print(f"- 输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()