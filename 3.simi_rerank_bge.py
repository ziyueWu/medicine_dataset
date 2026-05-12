# -*- coding: UTF-8 -*-

import io
import json
import numpy as np
import random

from FlagEmbedding import FlagAutoModel
from FlagEmbedding import FlagReranker

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
# from sklearn import metrics
import regex as re

path_search_result_m1 = '2.extend_search_result/medicine_1.json'
path_score_result_m1 = '3.score_result/medicine_1.json'


eb_model = FlagAutoModel.from_finetuned("model/bge-m3", local_files_only=True, use_fp16=True)
reranker = FlagReranker('model/bge-reranker-v2-m3', local_files_only=True)


def demo_doc_simi():

    np.set_printoptions(threshold=None, edgeitems=None)

    result_list = []
    item_list = read_file(path_search_result_m1)
    for i in range(len(item_list)):
        item = item_list[i]

        # 1.计算备选项之间的相似度
        doc_list = []
        source_item_list = item.get("source")
        for j in range(len(source_item_list)):
            source_item = source_item_list[j]
            doc_list.append(source_item.get("sentence_context"))
        doc_embeddings = eb_model.encode(doc_list, batch_size=12, max_length=1024)['dense_vecs']
        # embeddings_2 = eb_model.encode(doc_list)
        # similarity = embeddings_1 @ embeddings_2.T
        # print_2array(similarity)

        # 动态聚类，按0.95的相似度找到最相近的备选项——老师说不行
        # cluster_list = dynamic_cluster(similarity)

        # 1.kmeans
        kmeans = KMeans(n_clusters=5, random_state=12)
        kmeans_labels = kmeans.fit_predict(doc_embeddings)

        # 2.凝聚型层次聚类
        agg_clustering = AgglomerativeClustering(n_clusters=5)
        agg_labels = agg_clustering.fit_predict(doc_embeddings)

        # 3.谱聚类
        spectral_labels = SpectralClustering(n_clusters=5, gamma=1).fit_predict(doc_embeddings)

        component = get_string_save(item, "component")
        component = clear_component(component)
        # 2.计算结构化数据与备选项的相似度
        query = get_string_save(item, "name") + " " + component + " " + get_string_save(item, "usage")
        queries = [query]
        q_embeddings = eb_model.encode(queries, batch_size=12, max_length=1024)['dense_vecs']
        # q_embeddings = eb_model.encode_queries(queries)
        eb_scores = q_embeddings @ doc_embeddings.T
        # print_2array(eb_scores)

        rr_scores = reranker.compute_score(query_doc2pair(query, doc_list), normalize=True)
        # print_1array(rr_scores)

        # 处理数据并排序
        item = handle_results(item, kmeans_labels, agg_labels, spectral_labels, eb_scores, rr_scores)
        # item["source"] = remove_rank_4llm(item.get("source"), cluster_list)
        result_list.append(item)

    # print(result_list[0])
    save_file(path_score_result_m1, result_list)


# 整合结果
def handle_results(item, kmeans_labels, agg_labels, spectral_labels, eb_scores, rr_scores):
    source_item_list = item.get("source")
    for i in range(len(source_item_list)):
        # 1.kmeans label
        item["source"][i]["kmeans_label"] = int(kmeans_labels[i])

        # 2.agg label
        item["source"][i]["agg_label"] = int(agg_labels[i])

        # 3.spectral label
        item["source"][i]["spectral_label"] = int(spectral_labels[i])

        # 3.eb score
        item["source"][i]["embedding_score"] = float(eb_scores[0][i])

        # 4.rr score
        item["source"][i]["reranker_score"] = float(rr_scores[i])

    return item


def get_string_save(item, key):
    value = item.get(key)
    if value is None:
        value = ""
    return value


# 组合成query-doc对的列表
def query_doc2pair(query, doc_list):
    query_doc_list = []
    for i in range(len(doc_list)):
        query_doc_list.append((query, doc_list[i]))
    return query_doc_list


# 根据聚类结果，在doc_list中随机删除重复的文档
def remove_duplicate_doc(doc_list, cluster_list):
    remove_list = []
    for sub_list in cluster_list:
        sub_length = len(sub_list)
        # 随机选择保留的文档
        save_index = random.randint(0, sub_length - 1)
        for i in range(sub_length):
            if i == save_index:
                continue
            else:
                remove_list.append(sub_list[i])

    new_doc_list = [elem for i, elem in enumerate(doc_list) if i not in remove_list]
    return new_doc_list


# 动态聚类算法
# 输入为32x32的矩阵
# 输出为
# [[0],[1,2],[3,4]...]
def dynamic_cluster(input_matrix):
    result_list = []
    count = len(input_matrix)
    # 对称阵，且主对角线无意义，只需要循环上三角
    for i in range(count):
        for j in range(i+1, count):
            simi = float(input_matrix[i][j])
            # 可以认为非常相似，几乎一样，是重复的，需要去掉
            if simi >= 0.95:
                # print("i = " + str(i) + ", j = " + str(j))
                result_list.append([i, j])
    # print(result_list)
    # 合并子数组
    result_list = merge_arrays(result_list)
    # print(result_dict)
    return result_list


# 手动打印一维数组的内容
def print_1array(arr):
    print(arr.shape)
    for elem in arr:
        print(f"{elem:6.3f}", end=" ")  # 格式化输出，宽度为4


# 手动打印二维数组的内容
def print_2array(arr):
    print(arr.shape)
    for row in arr:
        for elem in row:
            print(f"{elem:6.3f}", end=" ")  # 格式化输出，宽度为4
        print()  # 换行


def clear_component(input_str):
    result = ""
    if input_str is None:
        return result
    if len(input_str) == 0:
        return result
    try:
        # 1. 删除阿拉伯数字
        result = re.sub(r'\d+', '', input_str)
        # 2. 删除括号，但保留括号中的内容
        result = re.sub(r'[()]', '', result)
        result = re.sub(r'[（）]', '', result)
        # 3. 删除“克”“g”
        result = result.replace('克', '')
        result = result.replace('g', '')
        # 4. 删除“大剂”“中剂”“小剂”
        result = result.replace('大剂', '')
        result = result.replace('中剂', '')
        result = result.replace('小剂', '')
    except Exception as e:
        print(input_str)
        print(e)
    return result


# 处理备选项，准备进入大模型
# 1.根据聚类的结果去重
# 2.保留eb score和rr
def remove_rank_4llm(source_list, cluster_list):
    # 1.去重
    source_list = remove_duplicate_doc(source_list, cluster_list)
    # 2.排序, 两个score从大到小
    source_list.sort(key=lambda k: (k.get("embedding_score", 0), k.get("reranker_score", 0)), reverse=True)
    return source_list


def demo():
    from FlagEmbedding import FlagAutoModel
    # model = FlagAutoModel.from_finetuned('BAAI/bge-large-zh-v1.5',
    #                                      query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    #                                      use_fp16=True)  # 设置use_fp16为True可以加快计算，效果会稍有下降
    model = FlagAutoModel.from_finetuned("model/bge-large-zh-v1.5", local_files_only=True)

    queries = ["风引汤 大黄 干姜 龙骨各56克 桂枝42克 甘草 牡蛎各28克 寒水石 滑石 赤石脂 白石脂 紫石英 石膏各84克"]
    docs = [
        "则胸满而短气。\n经行肌中。\n而心处胸间也。\n风引汤 除热瘫痫。\n大黄 干姜 龙骨（各四两） 桂枝（三两） 甘草 牡蛎（各二两） 寒水石 滑石 赤石脂白石脂 紫石英 石膏（各六两）上十二味。\n杵粗筛。\n以韦囊盛之。\n取三指撮。\n井花水三升。\n煮三沸。",
        "心电图正常。\n胸透阴性。\n总胆固醇252．2毫克％，三酸甘油酯130·0毫克％，血糖88毫克％，白分及尿常规均正常。\n舌红，两脉均弦细，症属阴虚火旺，虚火扰心，心肾不交，泻火安神风引汤主之。\n大黄1克，干姜6克，龙骨9克，桂枝6克，甘草6克，牡蛎15克，寒水石15克，滑石15克，赤石脂6克，紫石英15克。\n3剂。\n1987年1月13日二诊，3剂后失眠好转，口干减，大便不干，耳鸣减，原方12剂。\n1987年1月28 13三诊，诸症已愈，以原方6剂加减以善其后。\n1987年3月12日四诊，病人多次来诊，谓服上方要2付量作一付服方可生效，按上方双倍量继续服上方6剂，可以不发作。\n(《古妙方验案精选》199j：62)按语：肝为厥阴风木，下连肾水为乙癸同源，上接心火，成子母相应。"]
    q_embeddings = model.encode_queries(queries)
    d_embeddings = model.encode_corpus(docs)
    scores = q_embeddings @ d_embeddings.T
    print(scores)
    # [[0.6763 0.6157]]


def read_file(path):
    with io.open(path, 'r', encoding='utf-8') as file:
        data_list = file.readlines()
        total_string = ''.join(data_list)
    # print(total_string[:10])

    dict_list = json.loads(total_string)
    return dict_list


def save_file(path, content):
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False))


def read_test_matrix():
    with io.open("test_matrix.txt", 'r', encoding='utf-8') as file:
        data_list = file.readlines()
        matrix = []
        for line in data_list:
            line_list = line.split()
            matrix.append(line_list)
    return matrix


def merge_arrays(arr):
    # 用于存储合并后的结果
    merged = []

    # 遍历每个子数组
    for sub_arr in arr:
        # 用于标记是否需要新建一个组
        new_group = True
        # 遍历已合并的结果，检查是否有重复数字
        for i in range(len(merged)):
            # 如果当前子数组和已合并的组有交集
            if set(sub_arr) & set(merged[i]):
                # 合并当前子数组到已合并的组
                merged[i] = list(set(merged[i]) | set(sub_arr))
                new_group = False
                break
        # 如果没有找到可以合并的组，则新建一个组
        if new_group:
            merged.append(sub_arr)

    # 对合并后的结果进行排序（可选）
    merged = [sorted(group) for group in merged]

    return merged


def demo1():
    from FlagEmbedding import FlagAutoModel
    sentences_1 = ["样例数据-1", "样例数据-2"]
    sentences_2 = ["样例数据-3", "样例数据-4"]
    model = FlagAutoModel.from_finetuned('BAAI/bge-large-zh-v1.5',
                                         query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                         use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
    embeddings_1 = model.encode_corpus(sentences_1)
    embeddings_2 = model.encode_corpus(sentences_2)
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)

    # for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
    # corpus in retrieval task can still use encode_corpus(), since they don't need instruction
    queries = ['query_1', 'query_2']
    passages = ["样例文档-1", "样例文档-2"]
    q_embeddings = model.encode_queries(queries)
    p_embeddings = model.encode_corpus(passages)
    scores = q_embeddings @ p_embeddings.T
    print(scores)


def demo2():
    from FlagEmbedding import FlagModel
    sentences = ["样例数据-1", "样例数据-2"]
    model = FlagModel('BAAI/bge-large-zh-v1.5',
                      query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                      use_fp16=True)  # 设置use_fp16为True可以加快计算，效果会稍有下降
    embeddings_1 = model.encode(sentences)
    embeddings_2 = model.encode(sentences)
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)

    # 对于短查询到长文档的检索任务，请对查询使用 encode_queries() 函数，其会自动为每个查询加上指令
    # 由于候选文本不需要添加指令，检索中的候选集依然使用 encode() 或 encode_corpus() 函数
    queries = ['query_1', 'query_2']
    passages = ["样例文档-1", "样例文档-2"]
    q_embeddings = model.encode_queries(queries)
    p_embeddings = model.encode(passages)
    scores = q_embeddings @ p_embeddings.T
    print(scores)


# 实验不同的聚类方法的效果
def demo_doc_cluster():

    np.set_printoptions(threshold=None, edgeitems=None)

    item_list = read_file(path_search_result_m1)
    for i in range(10):
        item = item_list[i]

        # 1.计算备选项之间的相似度
        doc_list = []
        source_item_list = item.get("source")
        for j in range(len(source_item_list)):
            source_item = source_item_list[j]
            doc_list.append(source_item.get("sentence_context"))
        embeddings = eb_model.encode(doc_list, batch_size=12, max_length=1024)['dense_vecs']

        # 1.kmeans
        kmeans = KMeans(n_clusters=5, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        print("kmeans Cluster labels:", labels)
        print("====================================")

        # 2.层次聚类
        # linked = linkage(embeddings, method='ward')
        # print(linked) 很难利用的结果
        # print("====================================")

        # 3.DBSCAN
        dbscan = DBSCAN(eps=0.4, min_samples=2)
        labels = dbscan.fit_predict(embeddings)
        print("DBSCAN Cluster labels:", labels)


# 实验不同的聚类方法的效果
def demo_doc_cluster_2():
    np.set_printoptions(threshold=None, edgeitems=None)

    item_list = read_file(path_search_result_m1)
    for i in range(10):
        item = item_list[i]

        # 1.计算备选项之间的相似度
        doc_list = []
        source_item_list = item.get("source")
        for j in range(len(source_item_list)):
            source_item = source_item_list[j]
            doc_list.append(source_item.get("sentence_context"))
        embeddings = eb_model.encode(doc_list, batch_size=12, max_length=1024)['dense_vecs']

        # 1.凝聚型层次聚类
        clustering = AgglomerativeClustering(n_clusters=5)
        labels = clustering.fit_predict(embeddings)
        print("AgglomerativeClustering")
        print(labels)

        # sorted_indices = np.argsort(labels)
        # sorted_data = embeddings[sorted_indices]
        # print("排序后的数据索引：", sorted_data)
        print("====================================")

        # 2.谱聚类
        y_pred = SpectralClustering(n_clusters=5, gamma=1).fit_predict(embeddings)
        print("SpectralClustering")
        print(y_pred)

        # 调参
        # for index, gamma in enumerate((0.01, 0.1, 1, 10)):
        #     for inner_index, k in enumerate((3, 4, 5, 6)):
        #         y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(embeddings)
        #         print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k, "score:", metrics.calinski_harabasz_score(embeddings, y_pred))
        print("====================================")

        # 3.kmeans
        kmeans = KMeans(n_clusters=5, random_state=12)
        labels = kmeans.fit_predict(embeddings)
        print("kmeans Cluster")
        print(labels)
        print("====================================")


def demo_rerank():
    item_list = read_file(path_score_result_m1)
    for item in item_list:
        source_list = item.get("source")
        source_list.sort(key=lambda k: (k.get("embedding_score", 0), k.get("reranker_score", 0)), reverse=True)
    save_file(path_score_result_m1, item_list)


def demo_ppl():
    item_list = read_file(path_search_result_m1)
    for i in range(10):
        item = item_list[i]
        caution = get_string_save(item, "caution")
        classification = get_string_save(item, "classification")
        component = get_string_save(item, "component")
        cure = get_string_save(item, "cure")
        dosage = get_string_save(item, "dosage")
        from_book = get_string_save(item, "from_book")
        function = get_string_save(item, "function")
        link = get_string_save(item, "link")
        name = get_string_save(item, "name")
        usage = get_string_save(item, "usage")
        text_list = [caution, classification, component, cure, dosage, from_book, function, link, name, usage]
        text_embeddings = eb_model.encode(text_list, batch_size=12, max_length=1024)['dense_vecs']
        # 计算每个文本的向量范数（作为复杂度指标）
        complexity_scores_1 = [np.linalg.norm(embedding) for embedding in text_embeddings]
        # 计算向量的标准差,衡量的是嵌入分布的离散程度，也是一种复杂度的近似度量。
        complexity_scores_2 = [np.std(embedding) for embedding in text_embeddings]
        # 按复杂度从高到低排序
        sorted_results_1 = sorted(zip(text_list, complexity_scores_1), key=lambda x: x[1], reverse=True)
        sorted_results_2 = sorted(zip(text_list, complexity_scores_2), key=lambda x: x[1], reverse=True)
        # 输出结果
        print("计算每个文本的向量范数")
        for text, score in sorted_results_1:
            print(f"复杂度: {score:.4f} | 文本: {text}")
        print("计算向量的标准差")
        for text, score in sorted_results_2:
            print(f"复杂度: {score:.4f} | 文本: {text}")
        print("======================")


if __name__ == '__main__':
    # demo_doc_simi()

    demo_ppl()
    # demo_doc_cluster_2()
    # demo_doc_cluster()
    # demo_rerank()
    # demo()
    # demo1()
    # demo2()
    # print(read_test_matrix())
    # dynamic_cluster(read_test_matrix())
