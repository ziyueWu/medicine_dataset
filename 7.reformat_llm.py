# -*- coding: UTF-8 -*-

import json
import io
import call_gpt5

# 源
path_verify_result_m1 = '6.verify_result/medicine_1.json'

# 目的
path_reformat_result_m1 = '7.reformat_result/medicine_1.json'
# llm结果
path_reformat_llm_m1 = '7.reformat_result/medicine1/'
# 换一种整理格式
path_reformat_dataset_result_m1 = '7.reformat_result/medicine_dataset_1.json'

prompt_prefix = """在这个对话里我每次会发给你一个文本段落和一个json，需要你在文本段落中找到json中各个字段的值，我能保证这段文本和json是有关联的。
返回结果为json，list of dict格式，dict的各个key为"key"、"value"、"text_value"; 对于key==“component”,除了"value"、"text_value"外，还需要将json的方剂成分拆分成一个子list，记为components，其中每个dict为一个药材，药材dict的key分别为"text_name""text_count"表示药材的名次和剂量。中文里习惯量词后置，对于药材和剂量没有紧挨着的情况，需要向后查找最近的剂量词。
如果json的字段值为空，请按照你的理解，填充对应的"text_value"。"""


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


def handle():
    item_list = read_file(path_verify_result_m1)
    for i in range(len(item_list)):
        item = item_list[i]
        search_item = item.get("search_item")
        if len(search_item) == 0:  # 没找到的都不看
            continue
        verify_result = item.get("search_verify")
        if not verify_result:  # 找错了的也不看
            continue

        question_json = {"name": item.get("name"),
                         "component": item.get("component"),
                         "cure": item.get("cure")}
        question_text = search_item.get("sentence_context")
        q = prompt_prefix + "#文本段落\n" + question_text + "#对应json\n" + json.dumps(question_json, ensure_ascii=False)

        a = call_gpt5.call_gpt5(q)
        
        # 1. 循环内，将a保存至path_reformat_llm_m1+str(i)+".json"
        with open(path_reformat_llm_m1 + str(i) + ".json", 'w', encoding='utf-8') as f:
            json.dump(a, f, ensure_ascii=False, indent=2)
        
        # 2. a如果是json类型，item["reformat_result"]=a
        if isinstance(a, dict):
            item["reformat_result"] = a
    
    # 循环结束后save_file(path_reformat_result_m1, item_list)
    save_file(path_reformat_result_m1, item_list)


def demo_test():
    test_question = """
在这个对话里我每次会发给你一个文本段落和一个json，需要你在文本段落中找到json中各个字段的值，我能保证这段文本和json是有关联的。
返回结果为json，list of dict格式，dict的各个key为"key"、"value"、"text_value"; 对于key==“component”,除了"value"、"text_value"外，还需要将json的方剂成分拆分成一个子list，记为components，其中每个dict为一个药材，药材dict的key分别为"text_name""text_count"表示药材的名次和剂量。中文里习惯量词后置，对于药材和剂量没有紧挨着的情况，需要向后查找最近的剂量词。
如果json的字段值为空，请按照你的理解，填充对应的"text_value"。

#文本段落
续命汤治中风痱，身体不能自收，口不能言，冒昧不知痛处，或拘急不得转侧。
补中益气汤治阴虚内热，头痛口渴，表热自汗，不任风寒，脉洪大，心烦不安，四肢困倦，懒于言语，无气以动，动则气高而喘。
麻黄 桂枝 石膏 干姜 杏仁（四十枚） 川芎 当归 人参 甘草（各三两）上九味，以水一斗，煮取四升，温服一升，当小汗。
黄 人参 云术 炙甘草 陈皮 当归 升麻 柴胡上八味，加生姜三片，大枣二枚，水煎，温服。
薄覆脊，凭几坐，汗出自愈。
【集注】柯琴曰：仲景有建中、理中二法。
不汗更服。
无所禁，勿当风。
并治脉伏不得卧，咳逆上气，面目浮肿。
【集注】赵良曰：痱病者，营卫气血，不养于内外，故身体不用，机关不利，精神不治。
#对应json
{"name" : 续命汤 
"component" : 麻黄 桂枝 当归 人参 石膏 干姜 甘草各三两 川芎一两 杏仁四十枚 
"cure" :  }

"""
    answer = call_gpt5.call_gpt5(test_question)
    print("API返回的答案:")
    print(answer)


def read_reformat_llm_results():
    """
    读取path_reformat_llm_m1目录下的所有JSON文件，解析为json对象并返回
    """
    import os
    
    # 检查目录是否存在
    if not os.path.exists(path_reformat_llm_m1):
        print(f"目录 {path_reformat_llm_m1} 不存在")

    item_list = read_file(path_reformat_result_m1)
    for i in range(len(item_list)):
        item = item_list[i]
        search_item = item.get("search_item")
        if len(search_item) == 0:  # 没找到的都不看
            item["reformat_result"] = []
            continue
        verify_result = item.get("search_verify")
        if not verify_result:  # 找错了的也不看
            item["reformat_result"] = []
            continue

        file_path = path_reformat_llm_m1 + str(i) + ".json"

        try:
            with io.open(file_path, 'r', encoding='utf-8') as file:
                content = file.readlines()
                total_string = ''.join(content)
                # 去掉markdown的json标记
                total_string = total_string.replace('```json', '').replace('```', '')
                # print(total_string[:100])
                data = json.loads(total_string)
                # llm返回的是字符串，但是保存时又dump了一次，所以需要load两次！
                # 正确的做法应该，直接保存llm的结果，这边只load一次！
                real_data = json.loads(data)
                # print(type(real_data))
                item["reformat_result"] = real_data
            print(f"成功读取文件: {str(i)}")
        except json.JSONDecodeError as e:
            print(f"读取文件 {str(i)} 时发生JSON解析错误: {e}")
            # 尝试直接读取原始内容
            try:
                print(content)
                # with open(file_path, 'r', encoding='utf-8') as f:
                #     raw_content = f.read()
                #     print(f"文件 {str(i)} 的原始内容: {raw_content[:200]}...")
            except:
                pass
        except Exception as e:
            print(f"读取文件 {str(i)} 时发生错误: {e}")
    save_file(path_reformat_result_m1, item_list)


def reformat_dataset_result():
    import os

    # 检查目录是否存在
    if not os.path.exists(path_reformat_llm_m1):
        print(f"目录 {path_reformat_llm_m1} 不存在")

    dataset_list = []
    item_list = read_file(path_reformat_result_m1)
    for i in range(len(item_list)):
        item = item_list[i]
        new_item = {}
        search_item = item.get("search_item")
        if len(search_item) == 0:  # 没找到的都不看
            new_item["reformat_result"] = []
            continue
        verify_result = item.get("search_verify")
        if not verify_result:  # 找错了的也不看
            new_item["reformat_result"] = []
            continue

        file_path = path_reformat_llm_m1 + str(i) + ".json"

        try:
            with io.open(file_path, 'r', encoding='utf-8') as file:
                content = file.readlines()
                total_string = ''.join(content)
                # 去掉markdown的json标记
                total_string = total_string.replace('```json', '').replace('```', '')
                # print(total_string[:100])
                data = json.loads(total_string)
                # llm返回的是字符串，但是保存时又dump了一次，所以需要load两次！
                # 正确的做法应该，直接保存llm的结果，这边只load一次！
                real_data = json.loads(data)
                # print(type(real_data))
                new_item["reformat_result"] = real_data
                new_item["book_title"] = search_item["book_title"]
                new_item["chapter_title"] = search_item["chapter_title"]
                new_item["sentence"] = search_item["sentence"]
                new_item["sentence_context"] = search_item["sentence_context"]
                new_item["sentence_index"] = search_item["sentence_index"]
                new_item["old_data_index"] = i
                dataset_list.append(new_item)
            print(f"成功读取文件: {str(i)}")
        except json.JSONDecodeError as e:
            print(f"读取文件 {str(i)} 时发生JSON解析错误: {e}")
            # 尝试直接读取原始内容
            try:
                print(content)
                # with open(file_path, 'r', encoding='utf-8') as f:
                #     raw_content = f.read()
                #     print(f"文件 {str(i)} 的原始内容: {raw_content[:200]}...")
            except:
                pass
        except Exception as e:
            print(f"读取文件 {str(i)} 时发生错误: {e}")
    save_file(path_reformat_dataset_result_m1, dataset_list)


if __name__ == "__main__":
    # handle()
    # read_reformat_llm_results()
    reformat_dataset_result()
