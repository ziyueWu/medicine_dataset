import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence


RANDOM_SEED = 20260328
QUESTION_COUNT = 50
MAX_DATASET1_COMPONENT_COUNT_DIFF = 3

ROOT = Path(__file__).resolve().parent
STD_PATH = ROOT / "8.simi_add_delete" / "medicine_std_1.json"
ADD_DELETE_PATH = ROOT / "8.simi_add_delete" / "medicine_add_delete_1.json"
OUTPUT_DIR = ROOT / "13.build_sq_dataset"

FORMULA_LOOKUP_PATH = OUTPUT_DIR / "formula_name_component_cure.json"
MODIFIED_FORMULA_PATH = OUTPUT_DIR / "modified_formula.json"
DATASET1_QUESTIONS_PATH = OUTPUT_DIR / "dataset1_questions.json"
DATASET1_ANSWERS_PATH = OUTPUT_DIR / "dataset1_answers.json"
DATASET2_QUESTIONS_PATH = OUTPUT_DIR / "dataset2_questions.json"


def normalize_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def ensure_list_unique(items: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        value = normalize_text(item)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def join_components(items: Sequence[str]) -> str:
    return "、".join(ensure_list_unique(items))


def choose_better_text(old_value: str, new_value: str) -> str:
    old_value = normalize_text(old_value)
    new_value = normalize_text(new_value)
    if not old_value:
        return new_value
    if not new_value:
        return old_value
    return new_value if len(new_value) > len(old_value) else old_value


def build_formula_lookup() -> Dict[str, dict]:
    raw = json.loads(STD_PATH.read_text(encoding="utf-8"))
    lookup: Dict[str, dict] = {}

    for item in raw:
        reformat_result = item.get("reformat_result") or []
        name = ""
        component_block = None
        cure_block = None

        for block in reformat_result:
            key = block.get("key")
            if key == "name" and not name:
                name = normalize_text(block.get("value") or block.get("text_value") or "")
            elif key == "component" and component_block is None:
                component_block = block
            elif key == "cure" and cure_block is None:
                cure_block = block

        if not name:
            continue

        component_names: List[str] = []
        component_value = ""
        component_text_value = ""
        if component_block:
            component_value = normalize_text(component_block.get("value") or "")
            component_text_value = normalize_text(component_block.get("text_value") or "")
            for component in component_block.get("components") or []:
                standard_name = normalize_text(component.get("standard_name") or component.get("text_name") or "")
                if standard_name:
                    component_names.append(standard_name)

        cure_value = ""
        cure_text_value = ""
        if cure_block:
            cure_value = normalize_text(cure_block.get("value") or "")
            cure_text_value = normalize_text(cure_block.get("text_value") or "")

        current = lookup.get(name)
        payload = {
            "name": name,
            "component": {
                "components": ensure_list_unique(component_names),
                "value": component_value,
                "text_value": component_text_value,
            },
            "cure": {
                "value": cure_value,
                "text_value": cure_text_value,
            },
        }

        if current is None:
            lookup[name] = payload
            continue

        current_component_names = current["component"].get("components") or []
        new_component_names = payload["component"].get("components") or []
        if len(new_component_names) > len(current_component_names):
            current["component"]["components"] = new_component_names

        current["component"]["value"] = choose_better_text(current["component"].get("value", ""), component_value)
        current["component"]["text_value"] = choose_better_text(
            current["component"].get("text_value", ""), component_text_value
        )
        current["cure"]["value"] = choose_better_text(current["cure"].get("value", ""), cure_value)
        current["cure"]["text_value"] = choose_better_text(current["cure"].get("text_value", ""), cure_text_value)

    return lookup


def build_formula_pairs(formula_lookup: Dict[str, dict]) -> List[dict]:
    raw = json.loads(ADD_DELETE_PATH.read_text(encoding="utf-8"))
    pairs: List[dict] = []

    for item in raw:
        base_formula = item.get("base_formula") or {}
        base_name = normalize_text(base_formula.get("name") or "")
        base_components = ensure_list_unique(base_formula.get("components") or [])
        if not base_name:
            continue
        if len(base_components) <= 1:
            continue

        for modified_formula in item.get("modified_formula_list") or []:
            modified_name = normalize_text(modified_formula.get("name") or "")
            modified_components = ensure_list_unique(modified_formula.get("components") or [])
            if not modified_name:
                continue

            component_count_diff = len(modified_components) - len(base_components)
            if component_count_diff <= 0 or component_count_diff > MAX_DATASET1_COMPONENT_COUNT_DIFF:
                continue

            diff_component = [name for name in modified_components if name not in set(base_components)]
            if not diff_component:
                continue

            if base_name not in formula_lookup or modified_name not in formula_lookup:
                continue

            base_lookup = formula_lookup[base_name]
            modified_lookup = formula_lookup[modified_name]
            if not base_lookup["component"].get("value") and not base_lookup["component"].get("text_value"):
                continue
            if not modified_lookup["component"].get("value") and not modified_lookup["component"].get("text_value"):
                continue
            if not base_lookup["cure"].get("value") and not base_lookup["cure"].get("text_value"):
                continue
            if not modified_lookup["cure"].get("value") and not modified_lookup["cure"].get("text_value"):
                continue

            pairs.append(
                {
                    "base_formula_name": base_name,
                    "modified_formula_name": modified_name,
                    "base_formula_components": base_components,
                    "modified_formula_components": modified_components,
                    "diff_component": diff_component,
                }
            )

    return pairs


def build_modified_formula_lookup() -> Dict[str, dict]:
    raw = json.loads(ADD_DELETE_PATH.read_text(encoding="utf-8"))
    lookup: Dict[str, dict] = {}

    for item in raw:
        for modified_formula in item.get("modified_formula_list") or []:
            modified_name = normalize_text(modified_formula.get("name") or "")
            modified_components = ensure_list_unique(modified_formula.get("components") or [])
            if not modified_name:
                continue

            current = lookup.get(modified_name)
            if current is None or len(modified_components) > len(current.get("components") or []):
                lookup[modified_name] = {
                    "name": modified_name,
                    "components": modified_components,
                }

    return lookup


def build_dataset1(formula_lookup: Dict[str, dict], rng: random.Random) -> tuple[list, list]:
    pairs = build_formula_pairs(formula_lookup)
    if len(pairs) < QUESTION_COUNT:
        raise ValueError(f"可用方剂对不足 {QUESTION_COUNT} 个，当前仅 {len(pairs)} 个")

    selected_pairs = rng.sample(pairs, QUESTION_COUNT)
    questions: List[dict] = []
    answers: List[dict] = []

    for idx, pair in enumerate(selected_pairs, start=1):
        base_name = pair["base_formula_name"]
        modified_name = pair["modified_formula_name"]
        if len(pair["base_formula_components"]) <= 1:
            raise ValueError(f"基础方剂 {base_name} 仅 1 味药，不应进入题库")
        diff_component = pair["diff_component"]
        diff_component_text = join_components(diff_component)

        question_text = (
            f"用一段话比较{base_name}与{modified_name}的区别，"
            f"说明{diff_component_text}的作用以及增加{diff_component_text}后方剂主治病症有何变化"
        )

        questions.append(
            {
                "id": idx,
                "type": "方剂加味比较主观题",
                "question": question_text,
                "base_formula_name": base_name,
                "modified_formula_name": modified_name,
                "diff_component": diff_component,
                "diff_component_text": diff_component_text,
            }
        )

        answers.append(
            {
                "id": idx,
                "base_formula": {
                    "name": base_name,
                    "component": {
                        "value": formula_lookup[base_name]["component"].get("value", ""),
                        "text_value": formula_lookup[base_name]["component"].get("text_value", ""),
                    },
                    "cure": {
                        "value": formula_lookup[base_name]["cure"].get("value", ""),
                        "text_value": formula_lookup[base_name]["cure"].get("text_value", ""),
                    },
                },
                "modified_formula": {
                    "name": modified_name,
                    "component": {
                        "value": formula_lookup[modified_name]["component"].get("value", ""),
                        "text_value": formula_lookup[modified_name]["component"].get("text_value", ""),
                    },
                    "cure": {
                        "value": formula_lookup[modified_name]["cure"].get("value", ""),
                        "text_value": formula_lookup[modified_name]["cure"].get("text_value", ""),
                    },
                },
                "diff_component": diff_component,
                "diff_component_text": diff_component_text,
            }
        )

    return questions, answers


def build_dataset2(formula_lookup: Dict[str, dict], rng: random.Random) -> List[dict]:
    available_formulas = []
    for name, info in formula_lookup.items():
        component = info.get("component") or {}
        cure = info.get("cure") or {}
        if not name:
            continue
        if not (component.get("value") or component.get("text_value")):
            continue
        if not (cure.get("value") or cure.get("text_value")):
            continue
        available_formulas.append(info)

    if len(available_formulas) < QUESTION_COUNT:
        raise ValueError(f"可用方剂不足 {QUESTION_COUNT} 个，当前仅 {len(available_formulas)} 个")

    selected_formulas = rng.sample(available_formulas, QUESTION_COUNT)
    questions: List[dict] = []

    for idx, formula in enumerate(selected_formulas, start=1):
        formula_name = formula["name"]
        question_text = (
            f"根据你对中药的理解基于{formula_name}生成一个全新的加味方剂，"
            f"解释新增药物的作用，并说明预期功效。"
        )
        questions.append(
            {
                "id": idx,
                "type": "方剂创新加味主观题",
                "question": question_text,
                "formula_name": formula_name,
                "formula": {
                    "name": formula_name,
                    "component": {
                        "components": formula["component"].get("components", []),
                        "value": formula["component"].get("value", ""),
                        "text_value": formula["component"].get("text_value", ""),
                    },
                    "cure": {
                        "value": formula["cure"].get("value", ""),
                        "text_value": formula["cure"].get("text_value", ""),
                    },
                },
            }
        )

    return questions


def validate_dataset1(questions: List[dict], answers: List[dict]) -> None:
    if len(questions) != QUESTION_COUNT or len(answers) != QUESTION_COUNT:
        raise ValueError("题目或评分参考数量不正确")

    for question, answer in zip(questions, answers):
        if question["id"] != answer["id"]:
            raise ValueError(f"第 {question['id']} 题题目与评分参考编号不一致")
        if not question.get("question"):
            raise ValueError(f"第 {question['id']} 题题干为空")
        if not question.get("diff_component"):
            raise ValueError(f"第 {question['id']} 题缺少加味药物")

        for formula_key in ["base_formula", "modified_formula"]:
            formula = answer.get(formula_key) or {}
            if not formula.get("name"):
                raise ValueError(f"第 {question['id']} 题 {formula_key} 名称为空")
            component = formula.get("component") or {}
            cure = formula.get("cure") or {}
            if not (component.get("value") or component.get("text_value")):
                raise ValueError(f"第 {question['id']} 题 {formula_key} 组成为空")
            if not (cure.get("value") or cure.get("text_value")):
                raise ValueError(f"第 {question['id']} 题 {formula_key} 主治为空")


def validate_dataset2(questions: List[dict]) -> None:
    if len(questions) != QUESTION_COUNT:
        raise ValueError("第二个题库题目数量不正确")

    for question in questions:
        if not question.get("question"):
            raise ValueError(f"第 {question.get('id')} 题题干为空")
        formula = question.get("formula") or {}
        if not formula.get("name"):
            raise ValueError(f"第 {question.get('id')} 题方剂名称为空")
        component = formula.get("component") or {}
        cure = formula.get("cure") or {}
        if not (component.get("value") or component.get("text_value")):
            raise ValueError(f"第 {question.get('id')} 题方剂组成为空")
        if not (cure.get("value") or cure.get("text_value")):
            raise ValueError(f"第 {question.get('id')} 题方剂主治为空")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    formula_lookup = build_formula_lookup()
    modified_formula_lookup = build_modified_formula_lookup()
    dataset1_questions, dataset1_answers = build_dataset1(formula_lookup, rng)
    dataset2_questions = build_dataset2(formula_lookup, rng)
    validate_dataset1(dataset1_questions, dataset1_answers)
    validate_dataset2(dataset2_questions)

    write_json(FORMULA_LOOKUP_PATH, formula_lookup)
    write_json(MODIFIED_FORMULA_PATH, modified_formula_lookup)
    write_json(DATASET1_QUESTIONS_PATH, dataset1_questions)
    write_json(DATASET1_ANSWERS_PATH, dataset1_answers)
    write_json(DATASET2_QUESTIONS_PATH, dataset2_questions)

    print("主观题题库生成完成：")
    print(f"- {FORMULA_LOOKUP_PATH}")
    print(f"- {MODIFIED_FORMULA_PATH}")
    print(f"- {DATASET1_QUESTIONS_PATH}")
    print(f"- {DATASET1_ANSWERS_PATH}")
    print(f"- {DATASET2_QUESTIONS_PATH}")


if __name__ == "__main__":
    main()