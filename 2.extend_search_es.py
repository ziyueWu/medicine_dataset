# -*- coding: UTF-8 -*-

# 扩充查找es后的结果
# 1.找到source，若不为空且长度大于0
# 2.遍历source，根据title相同，搜索para_index-3至para_index+3的段落
# 3.仅保存找到source至extend_search_result文件夹

from elasticsearch import Elasticsearch
import io
import json

# 源
path_search_result_m1 = '1.search_result/medicine_1.json'
# path_search_result_m2 = '1.search_result/medicine_2.json'
# path_search_result_h1 = '1.search_result/herb_1.json'
# path_search_result_h2 = '1.search_result/herb_2.json'
# path_search_result_m3 = '1.search_result/medicine_3.json'

# path_search_result_m1 = '1.search_result/medicine_1_r16.json'
# path_search_result_m1 = '1.search_result/medicine_1_r64.json'
# path_search_result_m1 = '1.search_result/medicine_1_r128.json'

# 目的
path_result_m1 = '2.extend_search_result/medicine_1.json'
# path_result_m2 = '2.extend_search_result/medicine_2.json'
# path_result_h1 = '2.extend_search_result/herb_1.json'
# path_result_h2 = '2.extend_search_result/herb_2.json'
# path_result_m3 = '2.extend_search_result/medicine_3.json'

# path_result_m1 = '2.extend_search_result/medicine_1_r16.json'
# path_result_m1 = '2.extend_search_result/medicine_1_r64.json'
# path_result_m1 = '2.extend_search_result/medicine_1_r128.json'


def connect_es():
    es_client = Elasticsearch(hosts="http://localhost:9200", http_auth=('elastic', '-ECXN_*_UH*rG-htFtVP'),
                              scheme="http", port=9200)
    print(es_client.info())
    return es_client


def search_es(book_title, chapter_title, es_client, start_index, end_index):
    search_body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "book_title": {
                                "value": book_title
                            }
                        }
                    },
                    {
                        "match": {
                            "chapter_title": chapter_title
                        }
                    },
                    {
                        "range": {
                            "sentence_index": {
                                "gte": start_index,  # 大于或等于 start_index
                                "lte": end_index  # 小于或等于 end_index
                            }
                        }
                    }
                ]
            }
        }
    }
    raw_result = es_client.search(index='medicine_e00', body=search_body)
    hit_dict = raw_result.get("hits")
    hit_list = hit_dict.get("hits")
    # print(hit_list)
    source_list = []
    for i in range(len(hit_list)):
        source_item = hit_list[i].get("_source")
        source_list.append(source_item)

    # 升序排序
    source_list.sort(key=lambda x: x["sentence_index"])
    total_para_list = []
    for item in source_list:
        total_para_list.append(item.get("sentence"))
    total_para = "\n".join(total_para_list)

    return total_para


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


def handle(old_path, new_path, es_client):
    old_json_list = read_file(old_path)
    new_json_list = []
    for item in old_json_list:
        source = item.get("source")
        if source is None:
            continue
        if len(source) == 0:
            continue
        for source_item in source:
            book_title = source_item.get("book_title")
            chapter_title = source_item.get("chapter_title")
            sentence_index = source_item.get("sentence_index")
            sentence = source_item.get("sentence")
            if len(sentence) > 500:  # 如果原文足够长，则不需要补充上下文
                total_para = sentence
            else:
                total_para = search_es(book_title, chapter_title, es_client, sentence_index - 4, sentence_index + 5)
            source_item["sentence_context"] = total_para
        new_json_list.append(item)
    save_file(new_path, new_json_list)


if __name__ == '__main__':
    client = connect_es()
    handle(path_search_result_m1, path_result_m1, client)
    # handle(path_search_result_m2, path_result_m2, client)
    # handle(path_search_result_m3, path_result_m3, client)
    # handle(path_search_result_h1, path_result_h1, client)
    # handle(path_search_result_h2, path_result_h2, client)
