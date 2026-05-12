import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


RANDOM_SEED = 20260318
ROOT = Path(__file__).resolve().parent
MEDICINE_PATH = ROOT / "8.simi_add_delete" / "medicine_std_1.json"
SYNONYM_PATH = ROOT / "8.simi_add_delete" / "synonym_db.json"
OUTPUT_DIR = ROOT / "10.build_tcmllm_dataset"


def prefer_text(item: dict) -> str:
    return (item.get("value") or item.get("text_value") or "").strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def split_synonyms(text: str) -> List[str]:
    parts = re.split(r"[、,，;；/\s]+", text or "")
    cleaned = []
    seen = set()
    for part in parts:
        value = part.strip(" 、,，;；/\t\n\r")
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            cleaned.append(value)
    return cleaned


def shuffle_options(correct_answer: str, wrong_answers: Sequence[str], rng: random.Random) -> Tuple[Dict[str, str], str]:
    labels = ["A", "B", "C", "D"]
    options = [correct_answer, *wrong_answers]
    rng.shuffle(options)
    option_map = {label: option for label, option in zip(labels, options)}
    answer = next(label for label, option in option_map.items() if option == correct_answer)
    return option_map, answer


def sample_distinct(pool: Sequence[str], count: int, exclude: Optional[Iterable[str]] = None, rng: Optional[random.Random] = None) -> List[str]:
    rng = rng or random
    excluded = set(exclude or [])
    candidates = [item for item in pool if item not in excluded]
    if len(candidates) < count:
        raise ValueError(f"候选项不足，期望 {count} 个，实际 {len(candidates)} 个")
    return rng.sample(candidates, count)


def load_formula_records() -> List[dict]:
    with MEDICINE_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    grouped: Dict[str, dict] = {}
    for item in raw:
        name = ""
        cure = ""
        components: List[str] = []
        for block in item.get("reformat_result", []) or []:
            key = block.get("key")
            if key == "name" and not name:
                name = normalize_whitespace(prefer_text(block))
            elif key == "cure" and not cure:
                cure = normalize_whitespace(prefer_text(block))
            elif key == "component":
                for component in block.get("components", []) or []:
                    standard_name = normalize_whitespace(component.get("standard_name") or "")
                    if standard_name:
                        components.append(standard_name)

        if not name:
            continue

        components = list(dict.fromkeys(components))
        entry = grouped.setdefault(name, {"name": name, "cures": set(), "components": set()})
        if cure:
            entry["cures"].add(cure)
        if components:
            entry["components"].add(tuple(components))

    formulas = []
    for name, entry in grouped.items():
        if len(entry["cures"]) == 1:
            cure = next(iter(entry["cures"]))
        else:
            cure = None
        if len(entry["components"]) == 1:
            components = list(next(iter(entry["components"])))
        else:
            components = None

        formulas.append(
            {
                "name": name,
                "cure": cure,
                "components": components,
            }
        )
    return formulas


def load_synonym_data() -> Tuple[List[str], List[dict], List[str]]:
    with SYNONYM_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    standard_names: List[str] = []
    alias_pairs: List[dict] = []
    alias_pool: List[str] = []
    seen_standard = set()
    seen_alias_pool = set()

    for row in raw:
        if not isinstance(row, dict):
            continue
        standard = normalize_whitespace(row.get("术语名称") or "")
        alias_text = row.get("同义词") or ""
        if not standard or standard == "术语名称":
            continue

        if standard not in seen_standard:
            seen_standard.add(standard)
            standard_names.append(standard)

        aliases = split_synonyms(alias_text)
        for alias in aliases:
            if alias == standard:
                continue
            if standard in alias or alias in standard:
                continue
            alias_pairs.append({"standard_name": standard, "alias": alias})
            if alias not in seen_alias_pool:
                seen_alias_pool.add(alias)
                alias_pool.append(alias)

    all_name_pool = list(dict.fromkeys([*standard_names, *alias_pool]))
    return standard_names, alias_pairs, all_name_pool


def build_dataset1(formulas: List[dict], rng: random.Random) -> Tuple[List[dict], List[dict]]:
    component_formulas = [item for item in formulas if item["components"]]
    cure_formulas = [item for item in formulas if item["cure"]]

    if len(component_formulas) < 50 or len(cure_formulas) < 50:
        raise ValueError("可用于生成匹配题的方剂数量不足")

    component_pool = list(dict.fromkeys("、".join(item["components"]) for item in component_formulas))
    cure_pool = list(dict.fromkeys(item["cure"] for item in cure_formulas))

    questions: List[dict] = []
    answers: List[dict] = []

    selected_component_formulas = rng.sample(component_formulas, 50)
    selected_cure_formulas = rng.sample(cure_formulas, 50)

    for item in selected_component_formulas:
        correct = "、".join(item["components"])
        wrong = sample_distinct(component_pool, 3, exclude={correct}, rng=rng)
        option_map, answer = shuffle_options(correct, wrong, rng)
        qid = len(questions) + 1
        questions.append(
            {
                "id": qid,
                "type": "方剂名匹配组成",
                "question": f"方剂“{item['name']}”的正确组成是下列哪一项？",
                "options": option_map,
            }
        )
        answers.append(
            {
                "id": qid,
                "answer": answer,
                "answer_text": correct,
                "formula_name": item["name"],
            }
        )

    for item in selected_cure_formulas:
        correct = item["cure"]
        wrong = sample_distinct(cure_pool, 3, exclude={correct}, rng=rng)
        option_map, answer = shuffle_options(correct, wrong, rng)
        qid = len(questions) + 1
        questions.append(
            {
                "id": qid,
                "type": "方剂名匹配主治病症",
                "question": f"方剂“{item['name']}”对应的治疗病症是下列哪一项？",
                "options": option_map,
            }
        )
        answers.append(
            {
                "id": qid,
                "answer": answer,
                "answer_text": correct,
                "formula_name": item["name"],
            }
        )

    return questions, answers


def build_dataset2(formulas: List[dict], standard_names: List[str], rng: random.Random) -> Tuple[List[dict], List[dict]]:
    maskable_formulas = [item for item in formulas if item["components"] and len(item["components"]) >= 2]
    if len(maskable_formulas) < 100:
        raise ValueError("可用于生成补全题的方剂数量不足")

    questions: List[dict] = []
    answers: List[dict] = []
    selected = rng.sample(maskable_formulas, 100)

    for item in selected:
        components = list(item["components"])
        masked_index = rng.randrange(len(components))
        correct = components[masked_index]
        display_components = components[:]
        display_components[masked_index] = "____"
        wrong = sample_distinct(standard_names, 3, exclude=set(components), rng=rng)
        option_map, answer = shuffle_options(correct, wrong, rng)
        qid = len(questions) + 1
        questions.append(
            {
                "id": qid,
                "type": "方剂组成补全",
                "question": f"请补全方剂“{item['name']}”缺失的药材：{'、'.join(display_components)}",
                "options": option_map,
            }
        )
        answers.append(
            {
                "id": qid,
                "answer": answer,
                "answer_text": correct,
                "formula_name": item["name"],
                "masked_index": masked_index,
            }
        )

    return questions, answers


def build_dataset3(alias_pairs: List[dict], all_name_pool: List[str], rng: random.Random) -> Tuple[List[dict], List[dict]]:
    unique_pairs = []
    seen = set()
    for item in alias_pairs:
        key = (item["standard_name"], item["alias"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(item)

    if len(unique_pairs) < 100:
        raise ValueError("可用于生成别名识别题的样本不足")

    questions: List[dict] = []
    answers: List[dict] = []

    for item in rng.sample(unique_pairs, 100):
        correct = item["alias"]
        wrong = sample_distinct(all_name_pool, 3, exclude={item["standard_name"], correct}, rng=rng)
        option_map, answer = shuffle_options(correct, wrong, rng)
        qid = len(questions) + 1
        questions.append(
            {
                "id": qid,
                "type": "药材别名识别",
                "question": f"下列哪一项是中药“{item['standard_name']}”的别名？",
                "options": option_map,
            }
        )
        answers.append(
            {
                "id": qid,
                "answer": answer,
                "answer_text": correct,
                "standard_name": item["standard_name"],
            }
        )

    return questions, answers


def validate_dataset(name: str, questions: List[dict], answers: List[dict]) -> None:
    if len(questions) != 100 or len(answers) != 100:
        raise ValueError(f"{name} 题目或答案数量不是 100")
    for question, answer in zip(questions, answers):
        if question["id"] != answer["id"]:
            raise ValueError(f"{name} 题目与答案编号不一致")
        options = question["options"]
        if set(options.keys()) != {"A", "B", "C", "D"}:
            raise ValueError(f"{name} 第 {question['id']} 题选项标签错误")
        if len(set(options.values())) != 4:
            raise ValueError(f"{name} 第 {question['id']} 题存在重复选项")
        if answer["answer"] not in options:
            raise ValueError(f"{name} 第 {question['id']} 题答案标签无效")
        if options[answer["answer"]] != answer["answer_text"]:
            raise ValueError(f"{name} 第 {question['id']} 题答案内容不匹配")


def write_json(path: Path, data: List[dict]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    formulas = load_formula_records()
    standard_names, alias_pairs, all_name_pool = load_synonym_data()

    dataset1_q, dataset1_a = build_dataset1(formulas, rng)
    dataset2_q, dataset2_a = build_dataset2(formulas, standard_names, rng)
    dataset3_q, dataset3_a = build_dataset3(alias_pairs, all_name_pool, rng)

    validate_dataset("dataset1", dataset1_q, dataset1_a)
    validate_dataset("dataset2", dataset2_q, dataset2_a)
    validate_dataset("dataset3", dataset3_q, dataset3_a)

    write_json(OUTPUT_DIR / "dataset1_questions.json", dataset1_q)
    write_json(OUTPUT_DIR / "dataset1_answers.json", dataset1_a)
    write_json(OUTPUT_DIR / "dataset2_questions.json", dataset2_q)
    write_json(OUTPUT_DIR / "dataset2_answers.json", dataset2_a)
    write_json(OUTPUT_DIR / "dataset3_questions.json", dataset3_q)
    write_json(OUTPUT_DIR / "dataset3_answers.json", dataset3_a)

    print("数据集生成完成：")
    for file_name in [
        "dataset1_questions.json",
        "dataset1_answers.json",
        "dataset2_questions.json",
        "dataset2_answers.json",
        "dataset3_questions.json",
        "dataset3_answers.json",
    ]:
        print(f"- {OUTPUT_DIR / file_name}")


if __name__ == "__main__":
    main()
