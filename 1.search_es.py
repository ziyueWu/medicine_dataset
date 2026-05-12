# -*- coding: UTF-8 -*-

# 查找es
# 1.通过usage、component、name查找方剂的相关内容
# 2.通过component查找食疗相关的内容
# 3.通过function查找herb1相关的内容
# 4.通过chinese_name + function查找herb2相关的内容

from elasticsearch import Elasticsearch
import io
import json
import re
import random

# from tensorflow.compiler.tf2xla.python.xla import key_value_sort

path_json_medicine_1 = '/Users/wuziyue/Desktop/BLC/研究组/中医/jsons/medicine1_back.json'
path_json_medicine_2 = '/Users/wuziyue/Desktop/BLC/研究组/中医/jsons/medicine2.json'
# path_json_herb_1 = '/Users/wuziyue/Desktop/BLC/研究组/中医/jsons/herb1.json'
# path_json_herb_2 = '/Users/wuziyue/Desktop/BLC/研究组/中医/jsons/herb2.json'
path_json_fangji = '/Users/wuziyue/Desktop/BLC/研究组/中医/jsons/medicine3.json'

# path_search_result_m1 = '1.search_result/medicine_1_r16.json'
# path_search_result_m1 = '1.search_result/medicine_1_r64.json'
# path_search_result_m1 = '1.search_result/medicine_1_r128.json'
path_search_result_m1 = '1.search_result/medicine_1.json'
path_search_result_m2 = '1.search_result/medicine_2.json'
# path_search_result_h1 = '1.search_result/herb_1.json'
# path_search_result_h2 = '1.search_result/herb_2.json'
path_search_result_m3 = '1.search_result/medicine_3.json'


def connect_es():
    try:
        client = Elasticsearch(hosts="http://localhost:9200", http_auth=('elastic', '-ECXN_*_UH*rG-htFtVP'),
                               scheme="http", port=9200)
        print(client.info())
        return client
    except Exception as e:
        print(e)
        return None


def search_es(keywords, chapter_title):
    # search_str = ' '.join(keywords)
    # print(keywords)
    # search_body = {
    #     "query": {
    #         "match": {
    #             "sentence": keywords
    #         }
    #     }
    # }
    search_body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "sentence": keywords
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "chapter_title": chapter_title
                        }
                    },
                ],
                "minimum_should_match": 0
            }
        },
        "size": 32
    }
    raw_result = client.search(index='medicine_e00', body=search_body)
    hit_dict = raw_result.get("hits")
    hit_list = hit_dict.get("hits")

    source_list = []
    for i in range(len(hit_list)):
        source_item = hit_list[i].get("_source")
        source_list.append(source_item)
    return source_list


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


def usage_empty(usage):
    if usage is None:
        return True
    if len(usage) < 1:
        return True
    return False


def component_empty(component):
    if component is None:
        return True
    if len(component) < 1:
        return True
    return False


# 删去组成中的阿拉伯数字，删去括号（保留括号中的内容），删去“克”
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


def handle():
    # client = connect_es()
    medicine_list = read_file(path_json_medicine_1)
    for medicine_item in medicine_list:
        usage = medicine_item.get("usage")
        component = medicine_item.get("component")
        name = medicine_item.get("name")

        # 优先查找组成
        if not component_empty(component):
            component = clear_component(component)
            search_keywords = name + " " + component
        else:
            search_keywords = name + " " + usage
        # print(search_keywords)
        search_result = search_es(search_keywords, name)
        medicine_item["source"] = search_result
    save_file(path_search_result_m1, medicine_list)


# 测试数量级，16/64/128
def handle_test():
    # client = connect_es()
    medicine_list = read_file(path_json_medicine_1)
    random_numbers = random.sample(range(0, 721), 100)  # 随机选择100个进行测试
    new_item_list = []
    for i in random_numbers:
        medicine_item = medicine_list[i]
        usage = medicine_item.get("usage")
        component = medicine_item.get("component")
        name = medicine_item.get("name")

        # 优先查找组成
        if not component_empty(component):
            component = clear_component(component)
            search_keywords = name + " " + component
        else:
            search_keywords = name + " " + usage
        # print(search_keywords)
        search_result = search_es(search_keywords, name)
        medicine_item["source"] = search_result
        new_item_list.append(medicine_item)
    save_file(path_search_result_m1, new_item_list)


def handle2():
    # client = connect_es()
    medicine_list = read_file(path_json_medicine_2)
    for medicine_item in medicine_list:
        name = medicine_item.get("name")
        component = medicine_item.get("component")

        component = clear_component(component)
        search_keywords = name + " " + component
        search_result = search_es(search_keywords, name)
        medicine_item["source"] = search_result
    save_file(path_search_result_m2, medicine_list)


# def handle3():
#     client = connect_es()
#     herb_list = read_file(path_json_herb_1)
#     for herb_item in herb_list:
#         function = herb_item.get("function")
#
#         1.search_result = search_es(function, client)
#         herb_item["source"] = 1.search_result
#     save_file(path_search_result_h1, herb_list)


# def handle4():
#     client = connect_es()
#     herb_list = read_file(path_json_herb_2)
#     for herb_item in herb_list:
#         chinese_name = herb_item.get("chinese_name")
#         function = herb_item.get("function")
#
#         1.search_result = search_es(chinese_name + " " + function, client)
#         herb_item["source"] = 1.search_result
#     save_file(path_search_result_h2, herb_list)


def handle5():
    # client = connect_es()
    medicine_list = read_file(path_json_fangji)
    for medicine_item in medicine_list:
        name = medicine_item.get("name")
        component = medicine_item.get("component")

        component = clear_component(component)
        search_keywords = name + " " + component
        search_result = search_es(search_keywords, name)
        medicine_item["source"] = search_result
    save_file(path_search_result_m3, medicine_list)


if __name__ == '__main__':
    client = connect_es()
    handle()
    # handle2()
    # handle3()
    # handle4()
    # handle5()
    # input_str = "这是一个测试123，包含括号(内容)，还有克。（中剂）（中剂）"
    # output_str = clear_component(input_str)
    print(output_str)
    # handle_test()
