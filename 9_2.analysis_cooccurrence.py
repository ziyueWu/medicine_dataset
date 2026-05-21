# -*- coding: UTF-8 -*-

import json
import itertools
from collections import Counter, defaultdict

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# =========================
# 1. 数据读取
# =========================

def load_json_or_jsonl(file_path):
    """
    支持两种格式：
    1. JSON list: [ {...}, {...} ]
    2. JSONL: 每行一个 JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if text.startswith("["):
        return json.loads(text)
    else:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


# =========================
# 2. 抽取方剂名称和药物组成
# =========================

def extract_formula_info(record):
    """
    从单条方剂记录中抽取：
    - 方剂名
    - 标准药物名列表
    """
    formula_name = None
    herbs = []

    for item in record.get("reformat_result", []):
        if item.get("key") == "name":
            formula_name = item.get("value") or item.get("text_value")

        elif item.get("key") == "component":
            components = item.get("components", [])
            for comp in components:
                herb = comp.get("standard_name")
                if herb:
                    herbs.append(herb.strip())

    # 去重，避免同一方剂内同一药物重复计数
    herbs = sorted(set(herbs))

    return formula_name, herbs


def build_formula_herb_table(records, min_herb_count=2):
    """
    构建方剂-药物表。
    min_herb_count=2 表示至少包含两个药物才用于共现分析。
    """
    rows = []

    for idx, record in enumerate(records):
        formula_name, herbs = extract_formula_info(record)

        if not formula_name:
            formula_name = f"unknown_formula_{idx}"

        if len(herbs) >= min_herb_count:
            rows.append({
                "formula_id": idx,
                "formula_name": formula_name,
                "herbs": herbs,
                "herb_count": len(herbs),
                "book_title": record.get("book_title"),
                "chapter_title": record.get("chapter_title"),
            })

    return pd.DataFrame(rows)


# =========================
# 3. 构建药物共现网络
# =========================

def build_cooccurrence_network(formula_df):
    """
    节点：药物
    边：两个药物共同出现在同一个方剂中
    边权重：共现次数
    """
    herb_counter = Counter()
    pair_counter = Counter()

    # 记录每个药物出现在哪些方剂中
    herb_formula_map = defaultdict(list)

    for _, row in formula_df.iterrows():
        herbs = row["herbs"]
        formula_name = row["formula_name"]

        for herb in herbs:
            herb_counter[herb] += 1
            herb_formula_map[herb].append(formula_name)

        for h1, h2 in itertools.combinations(herbs, 2):
            pair = tuple(sorted([h1, h2]))
            pair_counter[pair] += 1

    G = nx.Graph()

    # 添加节点
    for herb, freq in herb_counter.items():
        G.add_node(
            herb,
            frequency=freq,
            formulas=";".join(herb_formula_map[herb])
        )

    # 添加边
    for (h1, h2), weight in pair_counter.items():
        G.add_edge(h1, h2, weight=weight)

    return G, herb_counter, pair_counter


# =========================
# 4. 高频药对统计
# =========================

def get_high_frequency_pairs(pair_counter, top_n=50, min_weight=2):
    rows = []

    for (h1, h2), weight in pair_counter.most_common():
        if weight >= min_weight:
            rows.append({
                "herb_1": h1,
                "herb_2": h2,
                "cooccurrence_count": weight
            })

    return pd.DataFrame(rows).head(top_n)


# =========================
# 5. 核心药物分析
# =========================

def analyze_core_herbs(G):
    """
    常用核心性指标：
    - frequency: 药物出现频次
    - degree: 与多少种药物发生共现
    - weighted_degree: 加权度，考虑共现次数
    - degree_centrality: 度中心性
    - betweenness_centrality: 中介中心性
    - eigenvector_centrality: 特征向量中心性
    """
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight="weight")

    try:
        eigenvector_centrality = nx.eigenvector_centrality(
            G,
            weight="weight",
            max_iter=1000
        )
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = {node: None for node in G.nodes()}

    rows = []

    for node in G.nodes():
        weighted_degree = sum(
            data.get("weight", 1)
            for _, _, data in G.edges(node, data=True)
        )

        rows.append({
            "herb": node,
            "frequency": G.nodes[node].get("frequency", 0),
            "degree": G.degree(node),
            "weighted_degree": weighted_degree,
            "degree_centrality": degree_centrality.get(node),
            "betweenness_centrality": betweenness_centrality.get(node),
            "eigenvector_centrality": eigenvector_centrality.get(node),
        })

    df = pd.DataFrame(rows)

    return df.sort_values(
        by=["weighted_degree", "frequency", "degree"],
        ascending=False
    )


# =========================
# 6. 社区发现
# =========================

def detect_communities(G):
    """
    更稳健的社区发现：
    1. 去除自环
    2. 去除孤立节点
    3. 按连通分量分别做 greedy modularity community detection
    """

    # 拷贝，避免修改原图
    G = G.copy()

    # 去除自环
    G.remove_edges_from(nx.selfloop_edges(G))

    # 去除孤立节点
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("警告：过滤后网络为空，无法进行社区发现。")
        return pd.DataFrame(columns=["community_id", "herb", "community_size"]), []

    all_communities = []
    node_to_community = {}
    rows = []
    community_id = 0

    # 对每个连通分量分别做社区发现，更稳定
    for component_nodes in nx.connected_components(G):
        subgraph = G.subgraph(component_nodes).copy()

        if subgraph.number_of_nodes() <= 2:
            communities = [set(subgraph.nodes())]
        else:
            try:
                communities = nx.algorithms.community.greedy_modularity_communities(
                    subgraph,
                    weight="weight"
                )
            except Exception as e:
                print(f"社区发现失败，改为将该连通分量作为一个社区。错误信息：{e}")
                communities = [set(subgraph.nodes())]

        for community in communities:
            herbs = sorted(list(community))
            all_communities.append(set(herbs))

            for herb in herbs:
                node_to_community[herb] = community_id
                rows.append({
                    "community_id": community_id,
                    "herb": herb,
                    "community_size": len(herbs)
                })

            community_id += 1

    nx.set_node_attributes(G, node_to_community, "community")

    community_df = pd.DataFrame(rows)

    return community_df, all_communities


# =========================
# 7. 网络过滤
# =========================

def filter_network_by_edge_weight(G, min_weight=2):
    """
    过滤低频共现边，并清洗异常边。
    """
    H = nx.Graph()

    for u, v, data in G.edges(data=True):
        # 去除自环
        if u == v:
            continue

        weight = data.get("weight", 1)

        # 确保 weight 是数值
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            continue

        # 去除无效权重
        if weight < min_weight:
            continue

        H.add_edge(u, v, weight=weight)

        # 复制节点属性
        if u in G.nodes:
            H.nodes[u].update(G.nodes[u])
        if v in G.nodes:
            H.nodes[v].update(G.nodes[v])

    return H


# =========================
# 8. 主流程
# =========================

def main(
    input_file,
    output_dir="./output",
    top_n_pairs=50,
    min_pair_weight=2,
    min_edge_weight_for_graph=2
):
    import os
    os.makedirs(output_dir, exist_ok=True)

    records = load_json_or_jsonl(input_file)

    # 方剂-药物表
    formula_df = build_formula_herb_table(records)
    formula_df.to_csv(
        f"{output_dir}/formula_herb_table.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 构建共现网络
    G, herb_counter, pair_counter = build_cooccurrence_network(formula_df)

    print(f"方剂数量：{len(formula_df)}")
    print(f"药物节点数量：{G.number_of_nodes()}")
    print(f"共现边数量：{G.number_of_edges()}")

    # 高频药对
    pair_df = get_high_frequency_pairs(
        pair_counter,
        top_n=top_n_pairs,
        min_weight=min_pair_weight
    )
    pair_df.to_csv(
        f"{output_dir}/high_frequency_herb_pairs.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 核心药物
    core_herb_df = analyze_core_herbs(G)
    core_herb_df.to_csv(
        f"{output_dir}/core_herbs.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 过滤网络后做社区发现
    H = filter_network_by_edge_weight(G, min_weight=min_edge_weight_for_graph)

    print(f"过滤后节点数量：{H.number_of_nodes()}")
    print(f"过滤后边数量：{H.number_of_edges()}")

    community_df, communities = detect_communities(H)
    community_df.to_csv(
        f"{output_dir}/herb_communities.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 导出 Gephi 可视化文件
    if H.number_of_nodes() > 0 and H.number_of_edges() > 0:
        nx.write_gexf(H, f"{output_dir}/herb_cooccurrence_network.gexf")
    else:
        print("过滤后网络为空，未导出 GEXF 文件。")

    # 绘制核心药物频次图
    plot_core_herb_frequency(
        core_herb_csv=f"{output_dir}/core_herbs.csv",
        output_path=f"{output_dir}/core_herb_frequency_top20.png",
        top_n=20
    )

    print("分析完成，结果已导出：")
    print(f"- {output_dir}/formula_herb_table.csv")
    print(f"- {output_dir}/high_frequency_herb_pairs.csv")
    print(f"- {output_dir}/core_herbs.csv")
    print(f"- {output_dir}/herb_communities.csv")
    print(f"- {output_dir}/herb_cooccurrence_network.gexf")
    print(f"- {output_dir}/core_herb_frequency_top20.png")

    return G, H, pair_df, core_herb_df, community_df


# =========================
# 9. 绘制核心药物频次直方图
# =========================
def plot_core_herb_frequency(
    core_herb_csv,
    output_path="./output/core_herb_frequency_top20.png",
    top_n=20
):
    """
    绘制核心药物 frequency 前20柱状图
    """

    # 读取 CSV
    df = pd.read_csv(core_herb_csv)

    # 按 frequency 排序
    df = df.sort_values(by="frequency", ascending=False).head(top_n)

    herbs = df["herb"].tolist()
    frequencies = df["frequency"].tolist()

    # 设置中文字体（Mac）
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]

    # Linux 可改：
    # plt.rcParams["font.sans-serif"] = ["SimHei"]

    # 解决负号显示问题
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
    plt.title("中医方剂中最常出现的中药前20", fontsize=18)
    plt.xlabel("中药名称", fontsize=14)
    plt.ylabel("方剂中出现的频次", fontsize=14)

    # x轴标签旋转
    plt.xticks(rotation=45)

    # 自动布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=300)

    # 显示图片
    plt.show()

    print(f"核心药物频次直方图已保存：{output_path}")


if __name__ == "__main__":
    input_file = "8.simi_add_delete/medicine_std_1.json"  # 修改为你的数据集路径

    G, H, pair_df, core_herb_df, community_df = main(
        input_file=input_file,
        output_dir="9_2.analysis_cooccurrence",
        top_n_pairs=100,
        min_pair_weight=2,
        min_edge_weight_for_graph=2
    )
