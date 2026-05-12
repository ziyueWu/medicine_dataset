# -*- coding: UTF-8 -*-

import io
import json
import math
import os
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx


path_add_delete_result_m1 = '8.simi_add_delete/medicine_add_delete_1.json'
path_analysis_output_dir = '9.analysis_add_delete'


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


def main():
    # plot_top_modified_formula_bar_chart()
    # plot_target_formula_relationship_graphs()
    build_formula_graph()


if __name__ == '__main__':
    main()