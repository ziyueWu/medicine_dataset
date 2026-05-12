# -*- coding: UTF-8 -*-

import io
import json
import re
from collections import Counter
import ast


# 源
path_llm_m1 = 'select_result/medicine_1.json'
path_llm_m2 = 'select_result/medicine_2.json'
path_llm_m3 = 'select_result/medicine_3.json'

# 目的
path_result_m1 = 'dataset/medicine_1.json'
path_result_m2 = 'dataset/medicine_2.json'
path_result_m3 = 'dataset/medicine_3.json'


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


def extract_1st_number(string):
    # 使用正则表达式提取第一个数字
    match = re.search(r'\d+', string)

    # 提取并打印第一个数字
    if match:
        first_number = match.group()
        return first_number
    else:
        # print("字符串中没有找到数字。")
        return "未找到"


# 首先对结果进行修正：
#   "outputs": {
#     "llm1_result": "1",
#     "llm2_result": "段落1",
#     "llm3_result": "段落2",
#     "para_count": 1,
#     "para_index": "1"
#   },
# llm1_result 也是段落1，para_count应修改为2，para_index应修改为段落1
def revise_outputs(outputs):
    llm1_result = outputs.get("llm1_result")
    llm2_result = outputs.get("llm2_result")
    llm3_result = outputs.get("llm3_result")
    if llm1_result.startswith("未找到"):
        llm1_num = "未找到"
    else:
        llm1_num = extract_1st_number(llm1_result)
    if llm2_result.startswith("未找到"):
        llm2_num = "未找到"
    else:
        llm2_num = extract_1st_number(llm2_result)
    if llm3_result.startswith("未找到"):
        llm3_num = "未找到"
    else:
        llm3_num = extract_1st_number(llm3_result)
    # 使用 Counter 统计每个字符串的出现次数
    count_dict = Counter([llm1_num, llm2_num, llm3_num])

    # 获取出现次数最多的字符串及其次数
    most_common_string, most_common_count = count_dict.most_common(1)[0]
    outputs["para_count"] = most_common_count
    outputs["para_index"] = most_common_string
    return outputs


# 然后修改json
# {"inputs":{}, 保持不变
# "outputs":"" 选择出来的段落}
def handle(from_path, to_path):
    llm_result_list = read_file(from_path)
    result_list = []
    for item in llm_result_list:
        inputs = item.get("inputs")
        knowledge = inputs.get("knowledge")
        try:
            knowledge = ast.literal_eval(knowledge)
        except (ValueError, SyntaxError) as e:
            print(f"解析失败: {e}")
        outputs = item.get("outputs")
        if outputs is None:
            continue
        # 首先判断是否有1和段落1，规整结果
        outputs = revise_outputs(outputs)

        para_count = outputs.get("para_count")
        para_index = outputs.get("para_index")

        if "未找到" in para_index:  # 最多的
            result_list.append({"inputs": knowledge,
                                "outputs": "未找到",
                                "check": False})
            continue

        final_para = "未找到"
        if para_count >= 2:
            para_index_num = extract_1st_number(para_index)
            final_para = inputs.get("paragraph" + para_index_num)

        result_list.append({"inputs": knowledge,
                            "outputs": final_para,
                            "check": False})
    save_file(to_path, result_list)


if __name__ == '__main__':
    handle(path_llm_m1, path_result_m1)
    handle(path_llm_m2, path_result_m2)
