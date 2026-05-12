# -*- coding: UTF-8 -*-


# 请求已构建好的dify workflow

import requests
import json
import io

from pympler.util.bottle import response

# 源
path_extend_result_m1 = '2.extend_search_result/medicine_1.json'
path_extend_result_m2 = '2.extend_search_result/medicine_2.json'
path_extend_result_m3 = '2.extend_search_result/medicine_3.json'

# 目的
path_result_m1 = 'select_result/medicine_1.json'
path_result_m2 = 'select_result/medicine_2.json'
path_result_m3 = 'select_result/medicine_3.json'

dify_workflow_url = "http://10.2.100.3:8082/v1/workflows/run"
api_key = "app-XTViZe5CnesOhv8X1wF2li5I"
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
        print(resp.json())
        return resp.json()

    except requests.exceptions.RequestException as e:
        print(f"调用 Workflow 时发生错误: {e}")
        return {}


def handle(source_file_path, target_file_path):
    data_list = read_file(source_file_path)
    result_list = []
    for item in data_list:
        item_source_list = item.pop("source", [])
        item_paragraph_list = []
        for source_item in item_source_list:
            # 处理段落长度，不大于2000
            para = source_item.get("sentence_context")
            if len(para) > 2000:
                para = para[:2000]
            item_paragraph_list.append(para)
        if len(item_paragraph_list) < 3:
            print("source 为空")
            continue
        input_dict = {"knowledge": str(item),
                      "paragraph1": item_paragraph_list[0],
                      "paragraph2": item_paragraph_list[1],
                      "paragraph3": item_paragraph_list[2]}
        data = {"inputs": input_dict,
                "response_mode": "blocking",
                "user": "medicine"}
        result = send_request(data)
        if len(result) == 0:
            continue
        res_data = result.get("data")
        outputs = res_data.get("outputs")
        new_item = item.copy()
        new_item["inputs"] = input_dict
        new_item["outputs"] = outputs
        result_list.append(new_item)
    save_file(target_file_path, result_list)


def demo_request():
    input_dict = {"knowledge": str({
        "caution": "1、一般冬季使用；\n2、青壮年多用；\n3、北方多用；\n4、有咽痛者不能使用麻黄汤。咽痛则表明有热，热证复用热药，则必然加重病情，故如此。",
        "classification": "解表剂-辛温解表",
        "component": "麻黄9克、桂枝6克、杏仁6克、炙甘草3克。",
        "cure": "外感风寒，恶寒发热，头痛身疼，无汗而喘，舌苔薄白，脉浮紧。（本方常用于感冒、流行性感冒、急性支气管炎、支气管哮喘等属风寒表实证者。） 【辨证要点】 恶寒发热，无汗而喘，脉浮紧。",
        "dosage": "",
        "from_book": "《伤寒论：辨太阳病脉证并治中》",
        "function": "发汗解表，宣肺平喘。",
        "link": "https://zhongyibaike.com/wiki/麻黄汤",
        "name": "麻黄汤",
        "usage": "上四味，以水九升，先煮麻黄，减二升，去上沫，内诸药，煮取二升半，去滓，温服八合。覆取微似汗，不须啜粥，余如桂枝法将息（现代用法：水煎服，温覆取微汗）。"
        }),
        "paragraph1": "麻黄 桂枝 杏仁 石膏 甘草（炙）\n",
        "paragraph2": "<strong>麻黄汤</strong>　麻黄　杏仁　桂枝　甘草\n",
        "paragraph3": "　　方药桂枝汤桂枝 芍药 甘草 大枣 生姜麻黄汤麻黄 桂枝 甘草 杏仁桂枝麻黄各半汤桂枝 芍药 甘草 大枣 生姜 麻黄 杏仁桂枝二越婢一汤桂枝 芍药 甘草 大枣 生姜 麻黄 石膏桂枝二麻黄一汤桂枝 芍药 甘草 大枣 生姜 麻黄 杏仁大青龙汤麻黄 桂枝 甘草 大枣 生姜 杏仁 石膏小青龙汤麻黄 桂枝 芍药 甘草 半夏 五味子 细辛 干姜白虎汤石膏 知母 甘草 粳米白虎加人参汤石膏 知母 甘草 粳米 人参五苓散茯苓 猪苓 泽泻 白术 桂枝茯苓甘草汤茯苓 桂枝 生姜 甘草文蛤散白散桔梗 贝母 巴豆桃核承气汤桃仁 桂枝 甘草 大黄 芒硝抵当汤大黄 水蛭 虻虫 桃仁抵当丸汤三之一为丸麻黄杏仁甘草石膏汤 甘草干姜汤 芍药甘草汤 新加汤桂枝 甘草 大枣 芍药 生姜 人参 于桂枝汤内加芍药 生姜（各一两） 人参（三两）\n"
    }
    data = {"inputs": input_dict,
            "response_mode": "blocking",
            "user": "medicine"}
    result = send_request(data)
    res_data = result.get("data")
    status = res_data.get("status")
    print(status)
    outputs = res_data.get("outputs")
    print(outputs)


if __name__ == '__main__':
    # demo_request()
    handle(path_extend_result_m1, path_result_m1)
    handle(path_extend_result_m2, path_result_m2)
