# -*- coding: UTF-8 -*-

import io
import json
import math
import os
from collections import Counter
from itertools import combinations
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np


path_add_delete_result_m1 = '8.simi_add_delete/medicine_add_delete_1.json'
path_analysis_output_dir = '9.analysis_add_delete'

INPUT_FILE = "8.simi_add_delete/medicine_add_delete_1.json"
OUTPUT_DIR = "9.analysis_add_delete"
FILTERED_FILE = os.path.join(OUTPUT_DIR, "medicine_add_delete_1_filtered.json")
ADDED_HERB_FILE = os.path.join(OUTPUT_DIR, "added_herb.json")

INPUT_FILE_FILTERED = "9.analysis_add_delete/medicine_add_delete_1_filtered.json"
OUTPUT_DIR_EVOL = "9.analysis_add_delete/evolution_path"

PATH_JSON_FILE = os.path.join(OUTPUT_DIR_EVOL, "formula_evolution_paths.json")
EDGE_CSV_FILE = os.path.join(OUTPUT_DIR_EVOL, "formula_evolution_edges.csv")


def read_file(path):
    with io.open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def configure_matplotlib_for_chinese():
    plt.rcParams['font.sans-serif'] = [
        'Arial Unicode MS',
        'PingFang SC',
        'Hiragino Sans GB',
        'Heiti SC',
        'STHeiti',
        'Microsoft YaHei',
        'SimHei',
        'DejaVu Sans',
    ]
    plt.rcParams['axes.unicode_minus'] = False


def get_sorted_base_formula_counts(path=path_add_delete_result_m1):
    data_list = read_file(path)
    result = []
    for item in data_list:
        base_formula = item.get('base_formula') or {}
        modified_formula_list = item.get('modified_formula_list') or []
        result.append(
            {
                'base_formula_name': base_formula.get('name', ''),
                'modified_count': len(modified_formula_list),
                'modified_formula_list': modified_formula_list,
            }
        )

    result.sort(
        key=lambda x: (-x['modified_count'], x['base_formula_name'])
    )
    return result


def plot_top_modified_formula_bar_chart(
    path=path_add_delete_result_m1,
    output_dir=path_analysis_output_dir,
    top_n=15,
):
    configure_matplotlib_for_chinese()
    ensure_dir(output_dir)

    sorted_result = get_sorted_base_formula_counts(path)
    top_result = sorted_result[:top_n]
    if not top_result:
        print('没有可用于绘图的加减方数据。')
        return None

    names = [item['base_formula_name'] for item in top_result]
    counts = [item['modified_count'] for item in top_result]

    width = max(12, len(names) * 0.9)
    plt.figure(figsize=(width, 7))
    bars = plt.bar(names, counts, color='#5B8FF9')
    plt.title(f'modified_formula_list 数量 Top {len(top_result)} 的基础方剂')
    plt.xlabel('基础方剂名称')
    plt.ylabel('modified_formula_list 数量')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha='center',
            va='bottom',
            fontsize=10,
        )

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'top15_modified_formula_bar.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存柱状图 -> {save_path}')
    return save_path


def build_target_formula_subgraph_map(
    path=path_add_delete_result_m1,
    target_names=None,
):
    if target_names is None:
        target_names = ['甘草汤', '桂枝甘草汤', '桂枝汤']

    data_list = read_file(path)
    target_map = {}
    for item in data_list:
        base_formula = item.get('base_formula') or {}
        base_name = base_formula.get('name', '')
        if base_name in target_names:
            target_map[base_name] = item

    return target_map


def _compute_star_layout_positions(base_name, modified_names):
    positions = {base_name: (0.0, 0.0)}
    count = len(modified_names)
    if count == 0:
        return positions

    radius = max(2.5, 1.2 + count * 0.2)
    for i, name in enumerate(modified_names):
        angle = 2 * math.pi * i / count
        positions[name] = (radius * math.cos(angle), radius * math.sin(angle))
    return positions


def draw_single_formula_relationship_graph(base_name, modified_formula_list, ax):
    graph = nx.Graph()
    graph.add_node(base_name, node_type='base')

    modified_names = []
    for formula in modified_formula_list:
        modified_name = formula.get('name', '')
        if not modified_name:
            continue
        graph.add_node(modified_name, node_type='modified')
        graph.add_edge(base_name, modified_name)
        modified_names.append(modified_name)

    positions = _compute_star_layout_positions(base_name, modified_names)
    base_nodes = [node for node, data in graph.nodes(data=True) if data.get('node_type') == 'base']
    modified_nodes = [node for node, data in graph.nodes(data=True) if data.get('node_type') == 'modified']

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=base_nodes,
        node_color='#E8684A',
        node_size=2200,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=modified_nodes,
        node_color='#6DC8EC',
        node_size=1400,
        ax=ax,
    )
    nx.draw_networkx_edges(graph, positions, width=1.5, edge_color='#999999', ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=10, ax=ax)

    ax.set_title(f'{base_name} 与加味方剂关系图\n(加味方数量: {len(modified_nodes)})')
    ax.axis('off')


def plot_target_formula_relationship_graphs(
    path=path_add_delete_result_m1,
    output_dir=path_analysis_output_dir,
    target_names=None,
):
    configure_matplotlib_for_chinese()
    ensure_dir(output_dir)

    if target_names is None:
        target_names = ['甘草汤', '桂枝甘草汤', '桂枝汤']

    target_map = build_target_formula_subgraph_map(path=path, target_names=target_names)
    found_names = [name for name in target_names if name in target_map]
    if not found_names:
        print('未找到指定基础方剂的数据。')
        return None

    fig, axes = plt.subplots(1, len(found_names), figsize=(7 * len(found_names), 7))
    if len(found_names) == 1:
        axes = [axes]

    for ax, base_name in zip(axes, found_names):
        modified_formula_list = target_map[base_name].get('modified_formula_list') or []
        draw_single_formula_relationship_graph(base_name, modified_formula_list, ax)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'target_formula_relationship_graphs.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存关系图 -> {save_path}')
    return save_path


def build_formula_graph():
    G = nx.DiGraph()
    data_list = read_file(path_add_delete_result_m1)
    all_formulas = []

    for i in range(len(data_list)):
        data = data_list[i]
        base = data["base_formula"]
        base_name = base["name"]
        base_components = base.get("components", [])

        all_formulas.append(
            {
                "name": base_name,
                "components": base_components,
                "formula_type": "base",
            }
        )

        # 添加基础方节点
        G.add_node(base_name, type="formula")

        # 添加药物节点 + 关系
        for herb in base_components:
            G.add_node(herb, type="herb")
            G.add_edge(base_name, herb, relation="contains")

        # 处理加减方
        for mf in data["modified_formula_list"]:
            mf_name = mf["name"]
            mf_components = mf.get("components", [])

            all_formulas.append(
                {
                    "name": mf_name,
                    "components": mf_components,
                    "formula_type": "modified",
                    "base_formula_name": base_name,
                }
            )

            G.add_node(mf_name, type="formula")
            G.add_edge(base_name, mf_name, relation="modifies_to")

            for herb in mf_components:
                G.add_node(herb, type="herb")
                G.add_edge(mf_name, herb, relation="contains")

    if G.number_of_nodes() == 0:
        print('没有可用于构建图谱的数据。')
        return None

    configure_matplotlib_for_chinese()
    ensure_dir(path_analysis_output_dir)

    plt.figure(figsize=(22, 18))
    pos = nx.spring_layout(G, seed=42, k=1.2)

    formula_nodes = [
        node for node, data in G.nodes(data=True) if data.get('type') == 'formula'
    ]
    herb_nodes = [
        node for node, data in G.nodes(data=True) if data.get('type') == 'herb'
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=formula_nodes,
        node_color='#E8684A',
        node_size=900,
        alpha=0.9,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=herb_nodes,
        node_color='#6DC8EC',
        node_size=450,
        alpha=0.9,
    )

    contains_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'contains'
    ]
    modifies_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'modifies_to'
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=contains_edges,
        edge_color='#999999',
        width=0.8,
        alpha=0.5,
        arrows=True,
        arrowsize=10,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=modifies_edges,
        edge_color='#5B8FF9',
        width=1.4,
        alpha=0.8,
        arrows=True,
        arrowsize=12,
    )

    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title('基础方-加减方-药物 关系图谱')
    plt.axis('off')
    plt.tight_layout()

    save_path = os.path.join(path_analysis_output_dir, 'formula_graph.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存图谱 -> {save_path}')

    formula_nodes = [
        node for node, data in G.nodes(data=True) if data.get('type') == 'formula'
    ]
    herb_nodes = [
        node for node, data in G.nodes(data=True) if data.get('type') == 'herb'
    ]

    degree_centrality = nx.degree_centrality(G)

    herb_freq = sorted(
        [
            {
                'name': node,
                'degree_centrality': degree_centrality[node],
                'degree': G.degree(node),
                'in_degree': G.in_degree(node),
                'out_degree': G.out_degree(node),
            }
            for node in herb_nodes
        ],
        key=lambda x: (-x['degree_centrality'], -x['degree'], x['name']),
    )

    formula_freq = sorted(
        [
            {
                'name': node,
                'degree_centrality': degree_centrality[node],
                'degree': G.degree(node),
                'in_degree': G.in_degree(node),
                'out_degree': G.out_degree(node),
            }
            for node in formula_nodes
        ],
        key=lambda x: (-x['degree_centrality'], -x['degree'], x['name']),
    )

    formula_modified_freq = sorted(
        [
            {
                'name': node,
                'modified_count': sum(
                    1
                    for _, _, edge_data in G.out_edges(node, data=True)
                    if edge_data.get('relation') == 'modifies_to'
                ),
            }
            for node in formula_nodes
        ],
        key=lambda x: (-x['modified_count'], x['name']),
    )

    pair_counter = Counter()
    for formula in all_formulas:
        herbs = formula.get('components', [])
        for pair in combinations(herbs, 2):
            pair_counter[tuple(sorted(pair))] += 1

    herb_herb_freq = [
        {
            'pair': list(pair),
            'count': count,
        }
        for pair, count in pair_counter.most_common(20)
    ]

    stats = {
        'formula_count': len(formula_nodes),
        'herb_count': len(herb_nodes),
        'herb_freq': herb_freq,
        'formula_freq': formula_freq,
        'formula_modified_freq': formula_modified_freq,
        'herb_herb_freq': herb_herb_freq,
        'graph_summary': {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'formula_formula_edge_count': len(modifies_edges),
            'formula_herb_edge_count': len(contains_edges),
            'all_formula_record_count': len(all_formulas),
        },
    }

    stats_path = os.path.join(path_analysis_output_dir, 'formula_graph_statistics.json')
    with io.open(stats_path, 'w', encoding='utf-8') as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    print(f'已保存图谱统计结果 -> {stats_path}')
    return {
        'graph_path': save_path,
        'stats_path': stats_path,
    }


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_components(components):
    """
    去除空值、空格，并去重。
    """
    if not components:
        return []

    return sorted(set(
        herb.strip()
        for herb in components
        if isinstance(herb, str) and herb.strip()
    ))


def filter_add_delete_data(data):
    """
    过滤规则：
    1. 基础方剂组成成分大于1种药物
    2. 基础方剂与修改方剂成分数量差不大于3
    """
    filtered_data = []

    for item in data:
        base_formula = item.get("base_formula", {})
        base_components = normalize_components(base_formula.get("components", []))

        if len(base_components) <= 1:
            continue

        new_modified_list = []

        for modified_formula in item.get("modified_formula_list", []):
            modified_components = normalize_components(
                modified_formula.get("components", [])
            )

            if not modified_components:
                continue

            count_diff = abs(len(modified_components) - len(base_components))

            if count_diff <= 3:
                new_modified_list.append({
                    "name": modified_formula.get("name", ""),
                    "components": modified_components
                })

        if new_modified_list:
            filtered_data.append({
                "base_formula": {
                    "name": base_formula.get("name", ""),
                    "components": base_components
                },
                "modified_formula_list": new_modified_list
            })

    return filtered_data


def summarize_added_herbs(filtered_data):
    """
    统计加减过程中最常被增加的：
    1. 单味药
    2. 药物对
    """
    added_herb_counter = Counter()
    added_pair_counter = Counter()

    detail_records = []

    for item in filtered_data:
        base_formula = item.get("base_formula", {})
        base_name = base_formula.get("name", "")
        base_components = set(normalize_components(base_formula.get("components", [])))

        for modified_formula in item.get("modified_formula_list", []):
            modified_name = modified_formula.get("name", "")
            modified_components = set(
                normalize_components(modified_formula.get("components", []))
            )

            added_herbs = sorted(modified_components - base_components)

            if not added_herbs:
                continue

            # 统计新增单味药
            for herb in added_herbs:
                added_herb_counter[herb] += 1

            # 统计新增药物对
            if len(added_herbs) >= 2:
                for pair in combinations(added_herbs, 2):
                    added_pair_counter[pair] += 1

            detail_records.append({
                "base_formula": base_name,
                "modified_formula": modified_name,
                "added_herbs": added_herbs
            })

    result = {
        "added_herb_statistics": [
            {
                "herb": herb,
                "frequency": freq
            }
            for herb, freq in added_herb_counter.most_common()
        ],
        "added_herb_pair_statistics": [
            {
                "herb_pair": list(pair),
                "frequency": freq
            }
            for pair, freq in added_pair_counter.most_common()
        ],
        "added_detail_records": detail_records
    }

    return result


def handle_added():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(INPUT_FILE)

    filtered_data = filter_add_delete_data(data)

    save_json(filtered_data, FILTERED_FILE)

    added_herb_result = summarize_added_herbs(filtered_data)

    save_json(added_herb_result, ADDED_HERB_FILE)

    print("处理完成")
    print(f"原始基础方剂数量：{len(data)}")
    print(f"过滤后基础方剂数量：{len(filtered_data)}")
    print(f"过滤结果保存至：{FILTERED_FILE}")
    print(f"新增药物统计保存至：{ADDED_HERB_FILE}")


# =========================
# 绘制“加减方剂中最常被增加的药物前20”
# =========================
def plot_added_herb_top20(
    input_file="9.analysis_add_delete/added_herb.json",
    output_file="9.analysis_add_delete/added_herb_top20.png",
    top_n=20
):
    # 读取 JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取新增药物统计
    herb_stats = data.get("added_herb_statistics", [])

    # 按频次排序并取前20
    herb_stats = sorted(
        herb_stats,
        key=lambda x: x["frequency"],
        reverse=True
    )[:top_n]

    herbs = [item["herb"] for item in herb_stats]
    frequencies = [item["frequency"] for item in herb_stats]

    # 设置中文字体（Mac）
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]

    # Windows 可改为：
    # plt.rcParams["font.sans-serif"] = ["SimHei"]

    plt.rcParams["axes.unicode_minus"] = False

    # 绘图
    plt.figure(figsize=(14, 7))

    bars = plt.bar(
        herbs,
        frequencies
    )

    # 添加数值标签
    for bar, freq in zip(bars, frequencies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(freq),
            ha='center',
            va='bottom',
            fontsize=10
        )

    # 标题与坐标轴
    plt.title("加减方剂中最常被增加的药物前20", fontsize=18)
    plt.xlabel("中药名称", fontsize=14)
    plt.ylabel("被加频次", fontsize=14)

    # 旋转横坐标
    plt.xticks(rotation=45)

    # 自动布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_file, dpi=300)

    # 显示图片
    plt.show()

    print(f"图片已保存至：{output_file}")


# =========================
# 绘制“加减方剂中最常被加的药物对前20”热力图
# =========================
def plot_added_herb_pair_heatmap(
    input_file="9.analysis_add_delete/added_herb.json",
    output_file="9.analysis_add_delete/added_herb_pair_heatmap_top20.png",
    top_n=20
):
    # 读取 JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pair_stats = data.get("added_herb_pair_statistics", [])

    # 按频次排序取前20
    pair_stats = sorted(
        pair_stats,
        key=lambda x: x["frequency"],
        reverse=True
    )[:top_n]

    # 获取所有涉及药物
    herb_set = set()

    for item in pair_stats:
        pair = item["herb_pair"]

        if len(pair) != 2:
            continue

        herb_set.update(pair)

    herbs = sorted(list(herb_set))

    # 构建矩阵
    matrix = pd.DataFrame(
        np.zeros((len(herbs), len(herbs))),
        index=herbs,
        columns=herbs
    )

    # 填充频次
    for item in pair_stats:
        pair = item["herb_pair"]
        freq = item["frequency"]

        if len(pair) != 2:
            continue

        h1, h2 = pair

        matrix.loc[h1, h2] = freq
        matrix.loc[h2, h1] = freq

    # 设置中文字体（Mac）
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]

    # Windows 可改：
    # plt.rcParams["font.sans-serif"] = ["SimHei"]

    plt.rcParams["axes.unicode_minus"] = False

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(matrix.values)

    # 坐标轴标签
    ax.set_xticks(np.arange(len(herbs)))
    ax.set_yticks(np.arange(len(herbs)))

    ax.set_xticklabels(herbs, fontsize=10)
    ax.set_yticklabels(herbs, fontsize=10)

    # 旋转横坐标
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    # 添加数值
    for i in range(len(herbs)):
        for j in range(len(herbs)):
            value = int(matrix.values[i, j])

            if value > 0:
                ax.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    fontsize=9
                )

    # 标题
    plt.title("加减方剂中最常被加的药物对前20", fontsize=18)

    # 颜色条
    cbar = plt.colorbar(im)
    cbar.set_label("共现频次", fontsize=12)

    # 自动布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_file, dpi=300)

    # 显示
    plt.show()

    print(f"热力图已保存至：{output_file}")


# =========================
# 方剂演化路径分析
# =========================
# =========================
# 1. 基础工具函数
# =========================

def get_edit_operation(base_components, modified_components):
    """
    计算从基础方剂到修改方剂的编辑操作：
    - added_herbs: 新增药物
    - deleted_herbs: 删除药物
    """
    base_set = set(normalize_components(base_components))
    modified_set = set(normalize_components(modified_components))

    added_herbs = sorted(modified_set - base_set)
    deleted_herbs = sorted(base_set - modified_set)

    return added_herbs, deleted_herbs


# =========================
# 2. 构建方剂演化图
# =========================

def build_formula_evolution_graph(data):
    """
    构建有向演化图：
    base_formula -> modified_formula
    """
    G = nx.DiGraph()

    edge_records = []

    for item in data:
        base_formula = item.get("base_formula", {})
        base_name = base_formula.get("name", "")
        base_components = normalize_components(base_formula.get("components", []))

        if not base_name:
            continue

        G.add_node(
            base_name,
            formula_name=base_name,
            components=";".join(base_components),
            component_count=len(base_components)
        )

        for modified_formula in item.get("modified_formula_list", []):
            modified_name = modified_formula.get("name", "")
            modified_components = normalize_components(
                modified_formula.get("components", [])
            )

            if not modified_name:
                continue

            added_herbs, deleted_herbs = get_edit_operation(
                base_components,
                modified_components
            )

            G.add_node(
                modified_name,
                formula_name=modified_name,
                components=";".join(modified_components),
                component_count=len(modified_components)
            )

            G.add_edge(
                base_name,
                modified_name,
                added_herbs=";".join(added_herbs),
                deleted_herbs=";".join(deleted_herbs),
                added_count=len(added_herbs),
                deleted_count=len(deleted_herbs),
                edit_distance=len(added_herbs) + len(deleted_herbs)
            )

            edge_records.append({
                "base_formula": base_name,
                "modified_formula": modified_name,
                "added_herbs": added_herbs,
                "deleted_herbs": deleted_herbs,
                "added_count": len(added_herbs),
                "deleted_count": len(deleted_herbs),
                "edit_distance": len(added_herbs) + len(deleted_herbs)
            })

    return G, edge_records


# =========================
# 3. 统计某个基础方剂的所有演化路径
# =========================

def find_evolution_paths(G, root_formula, max_depth=5):
    """
    从指定根方剂出发，寻找所有演化路径。

    返回示例：
    桂枝汤 -> 桂枝加葛根汤 -> 葛根汤
    """
    if root_formula not in G:
        print(f"未找到方剂：{root_formula}")
        return []

    paths = []

    queue = deque()
    queue.append((root_formula, [root_formula]))

    while queue:
        current, path = queue.popleft()

        # 到达最大深度，停止继续扩展
        if len(path) - 1 >= max_depth:
            paths.append(path)
            continue

        successors = list(G.successors(current))

        if not successors:
            paths.append(path)
            continue

        for nxt in successors:
            # 防止环
            if nxt in path:
                continue

            queue.append((nxt, path + [nxt]))

    return paths


def summarize_all_evolution_paths(G, max_depth=5):
    """
    对所有可能作为起点的方剂统计演化路径。
    只统计出度大于0的方剂。
    """
    result = {}

    for node in G.nodes():
        if G.out_degree(node) > 0:
            paths = find_evolution_paths(G, node, max_depth=max_depth)
            result[node] = [
                {
                    "path": path,
                    "path_length": len(path) - 1
                }
                for path in paths
            ]

    return result


# =========================
# 4. 导出边信息 CSV
# =========================

def save_edge_records_csv(edge_records, output_file):
    import pandas as pd

    df = pd.DataFrame(edge_records)

    # list 转字符串，便于保存
    df["added_herbs"] = df["added_herbs"].apply(lambda x: ";".join(x))
    df["deleted_herbs"] = df["deleted_herbs"].apply(lambda x: ";".join(x))

    df.to_csv(output_file, index=False, encoding="utf-8-sig")


# =========================
# 5. 绘制某个方剂的演化图
# =========================

def draw_formula_evolution_graph(
    G,
    root_formula,
    output_file=None,
    max_depth=3,
    figsize=(14, 10)
):
    """
    绘制指定根方剂的局部演化图。
    边标签展示：
    +新增药物 / -删除药物
    """

    if root_formula not in G:
        print(f"未找到方剂：{root_formula}")
        return

    # 获取 root_formula 在 max_depth 内可达的节点
    nodes = {root_formula}
    current_layer = {root_formula}

    for _ in range(max_depth):
        next_layer = set()

        for node in current_layer:
            next_layer.update(G.successors(node))

        nodes.update(next_layer)
        current_layer = next_layer

    subG = G.subgraph(nodes).copy()

    # 中文字体
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=figsize)

    # 使用 graphviz 布局，如果没装 pygraphviz，则使用 spring_layout
    try:
        pos = nx.nx_agraph.graphviz_layout(subG, prog="dot")
    except Exception:
        pos = nx.spring_layout(subG, k=1.2, seed=42)

    # 节点大小根据出度变化
    node_sizes = [
        1500 + 300 * subG.out_degree(node)
        for node in subG.nodes()
    ]

    nx.draw_networkx_nodes(
        subG,
        pos,
        node_size=node_sizes,
        alpha=0.9
    )

    nx.draw_networkx_edges(
        subG,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        width=1.5,
        alpha=0.7
    )

    nx.draw_networkx_labels(
        subG,
        pos,
        font_size=11
    )

    # 构造边标签
    edge_labels = {}

    for u, v, data in subG.edges(data=True):
        added = data.get("added_herbs", "")
        deleted = data.get("deleted_herbs", "")

        label_parts = []

        if added:
            label_parts.append(f"+{added}")

        if deleted:
            label_parts.append(f"-{deleted}")

        edge_labels[(u, v)] = "\n".join(label_parts)

    nx.draw_networkx_edge_labels(
        subG,
        pos,
        edge_labels=edge_labels,
        font_size=9
    )

    plt.title(f"{root_formula} 方剂演化路径图", fontsize=18)
    plt.axis("off")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"演化路径图已保存至：{output_file}")

    plt.show()


# =========================
# 6. 主流程
# =========================

def handle_evol_main():
    os.makedirs(OUTPUT_DIR_EVOL, exist_ok=True)

    data = load_json(INPUT_FILE_FILTERED)

    # 构建演化图
    G, edge_records = build_formula_evolution_graph(data)

    print(f"方剂节点数量：{G.number_of_nodes()}")
    print(f"演化边数量：{G.number_of_edges()}")

    # 保存边信息
    save_edge_records_csv(edge_records, EDGE_CSV_FILE)

    # 统计所有演化路径
    evolution_paths = summarize_all_evolution_paths(G, max_depth=5)
    save_json(evolution_paths, PATH_JSON_FILE)

    # 保存图文件，方便 Gephi 查看
    nx.write_gexf(
        G,
        os.path.join(OUTPUT_DIR, "formula_evolution_graph.gexf")
    )

    print(f"演化边信息已保存：{EDGE_CSV_FILE}")
    print(f"演化路径统计已保存：{PATH_JSON_FILE}")
    print("演化图 GEXF 已保存。")

    # 示例：绘制桂枝汤的演化路径
    draw_formula_evolution_graph(
        G,
        root_formula="桂枝汤",
        output_file=os.path.join(OUTPUT_DIR, "桂枝汤_方剂演化路径图.png"),
        max_depth=3
    )


# =========================
# 运行
# =========================
def main():
    # plot_top_modified_formula_bar_chart()
    # plot_target_formula_relationship_graphs()
    # build_formula_graph()
    # handle_added()
    # plot_added_herb_top20()
    # plot_added_herb_pair_heatmap()
    handle_evol_main()


if __name__ == '__main__':
    main()
