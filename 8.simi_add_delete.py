# -*- coding: UTF-8 -*-

# 根据数据集研究加减方剂
# 1.打印方剂名称及组成，观察数据

import json
import io
import os
import re
import subprocess
import argparse
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
from collections import Counter

import zlib

# 源
path_reformat_dataset_result_m1 = '7.reformat_result/medicine_dataset_1.json'

# 目的
path_synonym_db = '8.simi_add_delete/synonym_db.json'  # 有手动增加，小心别被覆盖！
path_std_dataset_result_m1 = '8.simi_add_delete/medicine_std_1.json'
path_add_delete_result_m1 = '8.simi_add_delete/medicine_add_delete_1.json'

# 相似病症 -> 共现药物
path_simi_cure_cooccur_result_m1 = '8.simi_add_delete/medicine_simi_cure_cooccur_1.json'

# mysql
user = 'root'
password = '12345678'
host = 'localhost'
database = 'mydb'
table_name = '中药组成'

# 查找词典
standard_dict = {}
synonym_dict = {}


# 从json文件中读取中药同义词库，在内存中构建查找词典
def read_synonym_dict():
    dict_list = read_file(path_synonym_db)
    for item in dict_list:
        std = item["术语名称"]
        if std == "术语名称":
            continue
        standard_dict[std] = std

        if item["同义词"]:
            syns = item["同义词"].split("、")
            for s in syns:
                synonym_dict[s] = std


def read_file(path):
    with io.open(path, 'r', encoding='utf-8') as file:
        data_list = file.readlines()
        total_string = ''.join(data_list)
    # print(total_string[:10])

    dict_list = json.loads(total_string)
    return dict_list


def save_file(path, content):
    # ensure parent dir exists
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False, indent=2))
        f.write("\n")


def backup_file_if_exists(path: str) -> Optional[str]:
    """Backup existing file to `path + .bak.<timestamp>`.

    Returns backup path if backed up, else None.
    """
    if not os.path.exists(path):
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak_path = f"{path}.bak.{ts}"
    # copy bytes to preserve original exactly
    with open(path, "rb") as rf, open(bak_path, "wb") as wf:
        wf.write(rf.read())
    return bak_path


def export_synonym_db_to_json(backup: bool = True):
    """Export the whole MySQL table into a JSON file at `path_synonym_db`.

    Uses local `mysql` CLI to avoid extra Python dependencies.
    Output format: List[Dict[column_name, value]]
    """

    sql = f"SELECT * FROM `{table_name}`;"
    cmd = [
        "mysql",
        "--default-character-set=utf8mb4",
        "--raw",
        "-h",
        host,
        "-u",
        user,
        f"-p{password}",
        database,
        "-B",  # batch, tab-separated, includes header row by default
        "-e",
        sql,
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "未找到 mysql 命令行工具。请先安装/确保可在PATH中使用 mysql。"
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"执行mysql导出失败：{stderr.strip()}") from e

    stdout = (res.stdout or b"").decode("utf-8", errors="replace")
    lines = [ln for ln in stdout.splitlines() if ln.strip() != ""]
    if not lines:
        save_file(path_synonym_db, [])
        return

    header = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        cols = ln.split("\t")
        # pad to header length
        if len(cols) < len(header):
            cols = cols + [""] * (len(header) - len(cols))
        row = {}
        for k, v in zip(header, cols):
            if v == "NULL":
                row[k] = None
            else:
                row[k] = v
        rows.append(row)

    bak_path = backup_file_if_exists(path_synonym_db) if backup else None
    save_file(path_synonym_db, rows)
    if bak_path:
        print(f"Backed up existing file -> {bak_path}")
    print(f"Exported {len(rows)} rows -> {path_synonym_db}")


def show_name_components():
    item_list = read_file(path_reformat_dataset_result_m1)
    for item in item_list:
        print(item["name"])
        print(item["component"])
        print(item["cure"])
        print("------")


# 去掉药物前缀
def preprocess_name(name):
    remove_prefix = ["生", "制", "炮", "炒", "炙"]
    # 去掉括号及括号内的文字（中英文括号）
    # e.g. "当归(酒制)" -> "当归"; "当归（酒制）" -> "当归"
    name = re.sub(r"\([^)]*\)|（[^）]*）", "", name).strip()
    # 特例："生地" 作为固定药名时不去掉“生”
    if name == "生地" or name == "生姜":
        return name
    for p in remove_prefix:
        if name.startswith(p):
            name = name[len(p):]
    return name


# 规范化药物名称
def normalize_components():
    read_synonym_dict()
    manual_check = []
    data_list = read_file(path_reformat_dataset_result_m1)

    for i in range(len(data_list)):
        item = data_list[i]
        reformat_item_list = item.get("reformat_result")
        for reformat_item in reformat_item_list:
            if reformat_item["key"] == "component":
                for comp in reformat_item["components"]:
                    name = comp["text_name"]
                    name = preprocess_name(name)
                    # 1 标准名查找
                    if name in standard_dict:
                        comp["standard_name"] = standard_dict[name]
                    # 2 同义词查找
                    elif name in synonym_dict:
                        comp["standard_name"] = synonym_dict[name]
                    # 3 未找到
                    else:
                        comp["standard_name"] = ""
                        manual_check.append(str(i) + " " + name)

    save_file(path_std_dataset_result_m1, data_list)

    # 有需要人工核验的中药名，打印其序号进行人工核查
    if len(manual_check) > 0:
        print(manual_check)


def _extract_formula_name_and_components(std_item: dict):
    """Extract formula name and standardized component list from one std dataset item.

    Input item schema (from `path_std_dataset_result_m1`) is a list of records;
    each record has `reformat_result` where:
      - key=="name" provides formula name in `value`
      - key=="component" provides `components`, each has `standard_name`
    """

    name = None
    components = []
    for it in std_item.get("reformat_result", []) or []:
        if it.get("key") == "name" and name is None:
            name = (it.get("value") or it.get("text_value") or "").strip()
        elif it.get("key") == "component":
            for c in it.get("components", []) or []:
                sn = (c.get("standard_name") or "").strip()
                if sn:
                    components.append(sn)
    # de-duplicate while preserving original order for output readability
    comp_list = list(dict.fromkeys(components))
    comp_set = frozenset(comp_list)
    if not name or not comp_list:
        return None, None, None
    return name, comp_set, comp_list


def _prefer_text_value(d: dict) -> str:
    """Prefer `text_value` over `value`, return stripped string."""
    if not isinstance(d, dict):
        return ""
    return (d.get("text_value") or d.get("value") or "").strip()


def _extract_name_cure_components(std_item: dict):
    """Extract (name, cure_text, components_list) from one std dataset record."""
    name = None
    cure = None
    components = []
    for it in std_item.get("reformat_result", []) or []:
        k = it.get("key")
        if k == "name" and name is None:
            name = _prefer_text_value(it)
        elif k == "cure" and cure is None:
            cure = _prefer_text_value(it)
        elif k == "component":
            for c in it.get("components", []) or []:
                sn = (c.get("standard_name") or "").strip()
                if sn:
                    components.append(sn)

    # components 去重（保持顺序）
    components = list(dict.fromkeys(components))
    if not name or not cure or not components:
        return None
    return {
        "name": name,
        "cure": cure,
        "components": components,
        "_comp_set": frozenset(components),
    }


def _intersect_postings(postings_lists):
    """Intersect multiple postings lists (list[int]) -> set[int]."""
    if not postings_lists:
        return set()
    if any(not lst for lst in postings_lists):
        return set()
    postings_lists.sort(key=len)
    cand = set(postings_lists[0])
    for lst in postings_lists[1:]:
        cand.intersection_update(lst)
        if not cand:
            break
    return cand


def handle_add_delete_formula():
    """Find set-containment relations between formulas (add/delete modifications).

    Reads standardized dataset at `path_std_dataset_result_m1`, uses each component's
    `standard_name` to build an inverted index, and performs a set containment query:
      base_formula.components ⊂ modified_formula.components

    Output saved to `path_add_delete_result_m1`.
    """

    std_data = read_file(path_std_dataset_result_m1)

    # Build unique formula instances by (name, component_set).
    formula_list = []
    seen = set()
    for item in std_data:
        name, comp_set, comp_list = _extract_formula_name_and_components(item)
        if not name:
            continue
        key = (name, tuple(comp_list))
        if key in seen:
            continue
        seen.add(key)
        formula_list.append(
            {
                "name": name,
                "components": comp_list,
                "_set": comp_set,
            }
        )

    # Inverted index: component -> formula indices
    inv = {}
    for idx, f in enumerate(formula_list):
        for c in f["_set"]:
            inv.setdefault(c, []).append(idx)
    for c in inv:
        inv[c].sort()

    sizes = [len(f["_set"]) for f in formula_list]

    result = []
    for i, base in enumerate(formula_list):
        base_set = base["_set"]
        postings = [inv.get(c, []) for c in base_set]
        cand = _intersect_postings(postings)
        if not cand:
            continue

        # strict supersets only
        modified_idx = [
            j
            for j in cand
            if j != i and sizes[j] > sizes[i] and base_set.issubset(formula_list[j]["_set"])
        ]
        if not modified_idx:
            continue

        modified_idx.sort(key=lambda j: (sizes[j], formula_list[j]["name"]))
        modified_formula_list = [
            {
                "name": formula_list[j]["name"],
                "components": formula_list[j]["components"],
            }
            for j in modified_idx
        ]

        result.append(
            {
                "base_formula": {
                    "name": base["name"],
                    "components": base["components"],
                },
                "modified_formula_list": modified_formula_list,
            }
        )

    # Optional: sort output for reproducibility
    result.sort(key=lambda x: (len(x["base_formula"]["components"]), x["base_formula"]["name"]))
    save_file(path_add_delete_result_m1, result)
    print(f"Saved {len(result)} base formulas with modifications -> {path_add_delete_result_m1}")


def _union_find(n: int):
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    return find, union, parent


def _clean_cure_text(s: str) -> str:
    """Normalize cure text for long-Chinese-text similarity.

    目标：弱化标点、模板化说明、以及“文本未直接给出...”这类占位文本的干扰，
    让 20~100 汉字的主治/病症描述更容易比较。
    """
    s = (s or "").strip()
    if not s:
        return ""

    # 去掉明显的占位说明/说明性文本
    placeholder_patterns = [
        r"（?文本未直接给出[^）)]*[）)]?",
        r"（?段落未直接给出[^）)]*[）)]?",
        r"（?仅载方名[^）)]*[）)]?",
        r"（?仅给出方名[^）)]*[）)]?",
        r"（?仅见方名[^）)]*[）)]?",
    ]
    for p in placeholder_patterns:
        s = re.sub(p, "", s)

    # 去掉常见的方书套话，保留更核心的病症信息
    boilerplate_patterns = [
        r"此方主之",
        r"本方主之",
        r"方主之",
        r"主之",
        r"宜服此方",
        r"可用此方",
    ]
    for p in boilerplate_patterns:
        s = re.sub(p, "", s)

    # 常见前缀轻微归一化
    s = re.sub(r"^治", "", s)
    s = re.sub(r"^用于", "", s)

    # 去空白与标点
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[\u3000\t\r\n]", "", s)
    s = re.sub(r"[，。；：、（）()【】\[\]《》<>“”\"'‘’….,;:!?！？\\/\-—_~`@#$%^&*+=|]", "", s)
    return s.strip()


def _hash_ngram_vectors(texts, hash_dim: int = 32768, ngram_min: int = 2, ngram_max: int = 4):
    """Simple hashing TF-IDF over char ngrams, suitable for Chinese long text."""
    N = len(texts)
    D = int(hash_dim)
    if D <= 0:
        raise ValueError("hash_dim must be positive")
    X = np.zeros((N, D), dtype=np.float32)
    df = np.zeros(D, dtype=np.int32)

    for i, t in enumerate(texts):
        t = _clean_cure_text(t)
        if not t:
            continue
        seen_bins = set()
        L = len(t)
        for n in range(int(ngram_min), int(ngram_max) + 1):
            if n <= 0 or L < n:
                continue
            for k in range(0, L - n + 1):
                ng = t[k : k + n]
                b = zlib.crc32(ng.encode("utf-8")) % D
                X[i, b] += 1.0
                seen_bins.add(b)
        for b in seen_bins:
            df[b] += 1

    idf = np.log((N + 1.0) / (df.astype(np.float32) + 1.0)) + 1.0
    X *= idf
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-12)
    return X


def _hybrid_similarity_matrix(
    texts,
    hash_dim: int = 32768,
    ngram_min: int = 2,
    ngram_max: int = 4,
    coarse_threshold: float = 0.06,
):
    """Hybrid similarity for long Chinese cure text.

    Step 1: 用 char ngram hash-tfidf 做粗筛与语义近似。
    Step 2: 对候选对再融合 SequenceMatcher 与包含关系，减少仅靠短公共片段带来的误召回。
    """
    clean_texts = [_clean_cure_text(x) for x in texts]
    vecs = _hash_ngram_vectors(
        clean_texts,
        hash_dim=hash_dim,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
    coarse = vecs @ vecs.T
    n = len(clean_texts)
    sim = np.eye(n, dtype=np.float32)

    for i in range(n):
        a = clean_texts[i]
        for j in range(i + 1, n):
            cos = float(coarse[i, j])
            if cos < coarse_threshold:
                continue

            b = clean_texts[j]
            sm = SequenceMatcher(None, a, b).ratio()
            short = min(len(a), len(b))
            long_ = max(len(a), len(b)) or 1
            contain = (a in b or b in a)
            contain_score = (short / long_) if contain and short >= 6 else 0.0

            final = 0.55 * cos + 0.30 * sm + 0.15 * contain_score
            sim[i, j] = sim[j, i] = float(final)

    return sim


def handle_simi_cure_cooccur(
    sim_threshold: float = 0.24,
    min_cluster_size: int = 2,
    min_support_ratio: float = 0.5,
    min_support_abs: int = 2,
    top_k: int = 30,
    sim_method: str = "hybrid",
    hash_dim: int = 32768,
    ngram_min: int = 2,
    ngram_max: int = 4,
    min_cure_len: int = 20,
    max_cure_len: int = 100,
):
    """For similar cure/indication texts, find high-frequency co-occuring drugs.

    - Similarity computed on `cure.text_value` (fallback `value`).
    - Default uses a no-dependency hybrid method tailored for 20~100 Chinese chars.
    - Grouping by connected components with similarity >= `sim_threshold`.
    - Co-occur drugs: herbs appearing in many formulas within a cluster.
    """

    std_data = read_file(path_std_dataset_result_m1)

    # 1) extract + de-duplicate instances by (name, cure, components)
    #    仅保留较适合本任务比较的中长文本（默认 20~100 汉字）
    formula_list = []
    seen = set()
    for item in std_data:
        f = _extract_name_cure_components(item)
        if not f:
            continue
        clean_cure = _clean_cure_text(f["cure"])
        cure_len = len(clean_cure)
        if cure_len < int(min_cure_len) or cure_len > int(max_cure_len):
            continue
        key = (f["name"], f["cure"], tuple(f["components"]))
        if key in seen:
            continue
        seen.add(key)
        f["_clean_cure"] = clean_cure
        formula_list.append(f)

    if not formula_list:
        save_file(path_simi_cure_cooccur_result_m1, [])
        print(f"No valid formula records, saved empty -> {path_simi_cure_cooccur_result_m1}")
        return

    cure_texts = [f["cure"] for f in formula_list]

    # 2) compute similarity matrix
    method = (sim_method or "hybrid").lower().strip()
    if method == "hybrid":
        sim = _hybrid_similarity_matrix(
            cure_texts,
            hash_dim=hash_dim,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
        )
    elif method == "hash-ngram":
        vecs = _hash_ngram_vectors(
            cure_texts,
            hash_dim=hash_dim,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
        )
        sim = vecs @ vecs.T
    else:
        raise ValueError(f"Unsupported sim_method: {sim_method}")

    # 3) build clusters (connected components) by threshold
    n = sim.shape[0]
    find, union, parent = _union_find(n)
    for i in range(n):
        # skip j<=i
        row = sim[i]
        for j in range(i + 1, n):
            if float(row[j]) >= sim_threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    # 4) compute co-occur drugs per cluster
    out = []
    for _, idxs in clusters.items():
        if len(idxs) < min_cluster_size:
            continue
        cluster_size = len(idxs)

        cnt = Counter()
        for i in idxs:
            cnt.update(set(formula_list[i]["components"]))

        min_freq = max(min_support_abs, int(np.ceil(cluster_size * min_support_ratio)))
        high_freq = [
            (drug, c)
            for drug, c in cnt.items()
            if c >= min_freq
        ]
        high_freq.sort(key=lambda x: (-x[1], x[0]))
        if top_k and top_k > 0:
            high_freq = high_freq[:top_k]

        cooccur_list = [
            {"药物": d, "频次": int(c), "支持度": round(c / cluster_size, 4)}
            for d, c in high_freq
        ]

        formula_out = [
            {
                "方剂名": formula_list[i]["name"],
                "治疗病症": formula_list[i]["cure"],
                "组成": formula_list[i]["components"],
            }
            for i in idxs
        ]
        # stable ordering
        formula_out.sort(key=lambda x: x["方剂名"])

        out.append(
            {
                "共现药物列表": cooccur_list,
                "有相似治疗病症的方剂列表": formula_out,
            }
        )

    # sort clusters: larger first, then by first formula name
    def _cluster_sort_key(x):
        fl = x.get("有相似治疗病症的方剂列表") or []
        first_name = fl[0]["方剂名"] if fl else ""
        return (-len(fl), first_name)

    out.sort(key=_cluster_sort_key)
    save_file(path_simi_cure_cooccur_result_m1, out)
    print(
        f"Saved {len(out)} clusters -> {path_simi_cure_cooccur_result_m1} (sim_threshold={sim_threshold}, method={method})"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TCM dataset utilities")
    parser.add_argument(
        "--export-synonym-db",
        action="store_true",
        help="从本地MySQL导出整个‘中药组成’表到 path_synonym_db (JSON)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="导出时不备份已有 synonym_db.json（默认会备份）",
    )
    parser.add_argument(
        "--normalize-components",
        action="store_true",
        help="读取 synonym_db.json 并对数据集成分做标准化（原默认行为）",
    )
    parser.add_argument(
        "--handle-add-delete-formula",
        action="store_true",
        help="基于标准化成分做方剂包含关系（加减方）检索，并输出到 path_add_delete_result_m1",
    )

    parser.add_argument(
        "--handle-simi-cure-cooccur",
        action="store_true",
        help="基于治疗病症(cure.text_value优先)相似度聚类，并输出每簇高频共现药物",
    )
    parser.add_argument(
        "--simi-threshold",
        type=float,
        default=0.24,
        help="治疗病症文本相似度阈值（默认0.24；hybrid 方法下适用于20~100字中文长文本）",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="最小聚类大小（默认2）",
    )
    parser.add_argument(
        "--min-support-ratio",
        type=float,
        default=0.5,
        help="共现药物最小支持度比例（默认0.5，即至少出现在一半方剂中）",
    )
    parser.add_argument(
        "--min-support-abs",
        type=int,
        default=2,
        help="共现药物最小出现次数（默认2）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="每簇最多输出多少个共现药物（默认30，<=0表示不截断）",
    )

    parser.add_argument(
        "--sim-method",
        type=str,
        default="hybrid",
        choices=["hybrid", "hash-ngram"],
        help="相似度计算方法：hybrid(默认，适合较长中文文本)、hash-ngram",
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=32768,
        help="hybrid/hash-ngram 使用的哈希向量维度（默认32768）",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=2,
        help="hash-ngram 的最小n（默认2）",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=4,
        help="hybrid/hash-ngram 的最大n（默认4）",
    )
    parser.add_argument(
        "--min-cure-len",
        type=int,
        default=20,
        help="参与相似病症比较的最短病症文本长度（清洗后，默认20）",
    )
    parser.add_argument(
        "--max-cure-len",
        type=int,
        default=100,
        help="参与相似病症比较的最长病症文本长度（清洗后，默认100）",
    )

    args = parser.parse_args()

    if args.export_synonym_db:
        export_synonym_db_to_json(backup=not args.no_backup)
    elif args.handle_add_delete_formula:
        handle_add_delete_formula()
    elif args.handle_simi_cure_cooccur:
        handle_simi_cure_cooccur(
            sim_threshold=args.simi_threshold,
            min_cluster_size=args.min_cluster_size,
            min_support_ratio=args.min_support_ratio,
            min_support_abs=args.min_support_abs,
            top_k=args.top_k,
            sim_method=args.sim_method,
            hash_dim=args.hash_dim,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            min_cure_len=args.min_cure_len,
            max_cure_len=args.max_cure_len,
        )
    elif args.normalize_components or (not args.export_synonym_db):
        # 默认保持原行为：normalize
        normalize_components()
