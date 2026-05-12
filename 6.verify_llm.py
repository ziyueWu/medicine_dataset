# -*- coding: UTF-8 -*-

import requests
import json
import io
import re

# 源
path_cascade_result_m1 = '5.cascade_result/medicine_1.json'

# 目的
path_verify_result_m1 = '6.verify_result/medicine_1.json'

dify_workflow_url = "http://10.2.200.101/v1/chat-messages"

api_key = "app-Z9rRYVSyoTzBkxlFoR8g4VLx"
headers = {
    "Authorization": f"Bearer {api_key}",  # 如果需要认证
    "Content-Type": "application/json"
}


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


def send_request(data):
    try:
        resp = requests.post(url=dify_workflow_url, json=data, headers=headers)
        # resp.raise_for_status()  # 检查 HTTP 请求是否成功

        # 打印结果
        print("Workflow 调用成功，返回结果:")
        # print(resp.json())
        return resp.json()

    except requests.exceptions.RequestException as e:
        print("调用 Workflow 时发生错误: {e}")
        return {}


def demo_request():
    data = {"query": """
    （刘草窗）陈皮芍，防风白术煎丸酌。
（散，钱乙）桑皮地骨皮，甘草粳米四般宜。
〔白术（土炒）三两，白芍（酒炒）四两，陈皮（炒）半两，防风一两，或煎或丸，久泻加升麻。
（桑白皮、地骨皮各一钱，甘草五分，粳米百粒。
〕补土泻木理肝脾，（陈皮理气补脾，防、芍泻木益土）。
桑皮泻肺火，地骨退虚热，甘草补土生金，粳米和中清肺。
若作食伤医便错。
李时珍曰：此泻肺诸方之准绳也。
（吴鹤皋曰：伤食腹痛，得泻便减，今泻而痛不减，故责之土败木贼也。
）参茯知芩皆可入，（人参、茯苓、知母、黄芩听加，名加减泻白散。
                  """,
            "inputs": {},
            "response_mode": "blocking",
            "user": "medicine"}
    result = send_request(data)
    answer = result.get("answer")
    print(answer)


def handle():
    item_list = read_file(path_cascade_result_m1)
    for item in item_list:
        search_item = item.get("search_item")
        if len(search_item) == 0:
            continue
        llm_input = search_item.get("chapter_title") + "\n" + search_item.get("sentence_context")
        data = {"query": llm_input,
                "inputs": {},
                "response_mode": "blocking",
                "user": "medicine"
                }
        result = send_request(data)
        answer = result.get("answer")
        quoted_texts = re.findall(r'"(.*?)"', answer)
        if '不知道' in quoted_texts:  # 回答了不知道，说明找到的文本不对
            search_verify = False
        elif item.get("name") not in quoted_texts:  # 原方剂名称 与 检索到的结果 不匹配，说明找的不对
            search_verify = False
        else:
            search_verify = True
        item["search_verify"] = search_verify
        print(search_verify)
    save_file(path_verify_result_m1, item_list)


def test():
    verify_result = read_file(path_verify_result_m1)
    total_count = len(verify_result)
    print(total_count)  # 721
    true_pos_count = 0
    true_neg_count = 0
    false_pos_count = 0
    false_neg_count = 0
    for item in verify_result:
        if len(item.get("search_item")) == 0:  # 没有找到
            if item.get("search_verify") is False:  # 但是漏掉了
                false_neg_count += 1
            else:  # 确实没有
                false_pos_count += 1
        else:  # 找到了
            if item.get("search_verify") is False:  # 但是找错了
                true_neg_count += 1
            else:  # 并且是对的
                true_pos_count += 1
        # if item.get("search_verify") is False:
        #     total_false_count += 1
    # print(total_false_count)  # 132
    # tp=73 fp=4 fn=1 tn=22
    # p = 0.95
    # r = 0.99 1.94
    print(true_pos_count)  # 427 tp
    print(true_neg_count)  # 59 fp
    print(false_pos_count)  # 200 tn
    print(false_neg_count)  # 35 fn
    # p = tp/(tp+fp) = 0.8786
    # r = tp/(tp+fn) = 0.9242
    # p + r = 1.8028
    # f1 = 2*(p*r)/(p+r) = 0.9008


if __name__ == '__main__':
    # demo_request()
    # handle()
    test()
