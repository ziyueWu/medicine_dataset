# -*- coding: UTF-8 -*-


import io
import json


# 源
path_score_result_m1_bge1 = '3.score_result/medicine_1_bge1.5.json'
path_score_result_m1_m3 = '3.score_result/medicine_1_bgem3.json'

# 目的
path_rerank_result_m1_bge1 = '4.rerank_result/medicine_1_bge1.5.json'
path_rerank_result_m1_m3 = '4.rerank_result/medicine_1_bgem3.json'


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


def top5(label_key, source_item_list):
    label_list = [0, 1, 2, 3, 4]
    top5_list = []
    for i in range(len(source_item_list)):
        item = source_item_list[i]
        item_label = item.get(label_key)
        if item_label in label_list:
            top5_list.append(item)
            label_list.remove(item_label)
        else:
            continue
    return top5_list


def merge_top5_list(list1, list2, list3):
    from collections import Counter
    combined = list1 + list2 + list3
    counter = Counter(json.dumps(d, sort_keys=True) for d in combined)
    # print(counter)
    result_list = []
    for item_str, count in counter.items():
        # print(type(item_str))
        # print(count)
        item_dict = json.loads(item_str)
        item_dict["merge_count"] = count
        result_list.append(item_dict)
    return result_list


def handle_score(source_path, result_path):
    source_list = read_file(source_path)
    # for i in range(10):
    #     item = source_list[i]
    for item in source_list:
        source_item_list = item.get("source")
        source_item_list.sort(key=lambda k: (k.get("embedding_score", 0), k.get("reranker_score", 0)), reverse=True)
        kmeans_top5 = top5("kmeans_label", source_item_list)
        agg_top5 = top5("agg_label", source_item_list)
        spectral_top5 = top5("spectral_label", source_item_list)
        merge_top5 = merge_top5_list(kmeans_top5, agg_top5, spectral_top5)
        item["merge_top5"] = merge_top5

    save_file(result_path, source_list)


if __name__ == '__main__':
    handle_score(path_score_result_m1_bge1, path_rerank_result_m1_bge1)
    handle_score(path_score_result_m1_m3, path_rerank_result_m1_m3)
