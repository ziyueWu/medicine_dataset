# -*- coding: UTF-8 -*-

import requests
import json
import io
import re

# 源
path_rerank_result_m1 = '4.rerank_result/medicine_1_bgem3.json'

# 目的
path_cascade_result_m1 = '5.cascade_result/medicine_1/'

dify_workflow_url = "http://10.2.200.101/v1/workflows/run"
api_key = "app-nrLkosD1i2z94HUv46Kz3wEh"
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
        # print("Workflow 调用成功，返回结果:")
        # print(resp.json())
        return resp.json()

    except requests.exceptions.RequestException as e:
        print("调用 Workflow 时发生错误: {e}")
        return {}


def demo_request():
    input_dict = {"rank_score_ave": 0.81,
                  "llm_input": """
                  #目标json：
{
"component" : 黄柏90克 缩砂仁45克 甘草60克 
"cure" : 肾阴不足，相火妄动，夜梦遗精。 【方论】 封髓丹为固精之要药。方用黄柏为君，以其味性苦寒，苦能坚肾，肾职得坚则阴水不虞其泛溢，寒能清肃，秋令一至，则龙火不至于奋阳，水火交摄，精有不安其位者乎；佐以甘草，以甘能缓急，泻诸火与肝火之内烦，且能使水土合为一家，以妙封藏之固；若缩砂者，以其味辛性温，善能入肾，肾之所恶在燥，而润之者惟辛，缩砂通三焦，达津液能纳五脏六腑之精，而归于肾，肾家之气纳，肾中之髓自藏矣。 
"function" : 降火止遗。 
"name" : 封髓丹  
}
#备选1:
{
"book_title" : 时方歌括 
"chapter_title" : 封髓丹 
"reranker_score" : 0.9439619007479929 
"sentence_context" : 妄梦遗精封髓丹。砂仁黄柏草和丸。（砂仁一两。黄柏二两。炙甘草七钱。蜜丸每服三钱。淡盐汤送下。一本用肉苁蓉五钱。切片洗淡。酒浸一宿次日煎三四沸食。  
}
#备选2:
{
"book_title" : 时方歌括 
"chapter_title" : 封髓丹 
"reranker_score" : 0.9314444378298972 
"sentence_context" : 况肝又藏魂。神魂不摄。宜其夜卧思交。精泄之症出矣。封髓丹为固精之要药。方用黄柏为君。以其味性苦寒。苦能坚肾。肾职得坚。则阴水不虞其泛溢。  
}
#备选3:
{
"book_title" : 时方歌括 
"chapter_title" : 封髓丹 
"reranker_score" : 0.3142606487357905 
"sentence_context" : 佐以甘草。以甘能缓急。泻诸火与肝火之内烦。且能使水土合为一家。以妙封藏之固。若缩砂者。以其味辛性温。善能入肾。肾之所恶在燥。而润之者惟辛。 
}
                  """}
    data = {"inputs": input_dict,
            "response_mode": "blocking",
            "user": "medicine"}
    result = send_request(data)
    res_data = result.get("data")
    status = res_data.get("status")
    print(status)
    outputs = res_data.get("outputs")
    print(outputs)


def send2llm(rank_score_ave, llm_input):
    input_dict = {"rank_score_ave": rank_score_ave,
                  "llm_input": llm_input}
    data = {"inputs": input_dict,
            "response_mode": "blocking",
            "user": "medicine"}
    result = send_request(data)
    res_data = result.get("data")
    status = res_data.get("status")
    print(status)
    outputs = res_data.get("outputs")
    # print(outputs)
    return outputs


def handle():
    item_list = read_file(path_rerank_result_m1)
    for index in range(len(item_list)):
        item = item_list[index]
        # 第一步，计算rank_score并得到各项的平均分
        merge_top5_list = item.get("merge_top5")
        m = len(merge_top5_list)
        rank_score_list = []
        candidate_list = []
        for i in range(m):
            top5_item = merge_top5_list[i]
            embedding_score = top5_item.get("embedding_score")
            reranker_score = top5_item.get("reranker_score")
            rank_score = (1 - i/m + embedding_score + reranker_score) / 3.0
            rank_score_list.append(rank_score)
            candidate_list.append({"book_title": top5_item.get("book_title"),
                                   "chapter_title": top5_item.get("chapter_title"),
                                   "rank_score": rank_score,
                                   "sentence_context": top5_item.get("sentence_context")})
        rank_score_ave = sum(rank_score_list) / m

        # 第二步，拼接发送给llm的长文本
        target_json = {"component": item.get("component"),
                       "cure": item.get("cure"),
                       "function": item.get("function"),
                       "name": item.get("name")}
        llm_input = "目标json：\n" + json.dumps(target_json, ensure_ascii=False) + "\n"
        for i in range(len(candidate_list)):
            llm_input += "备选" + str(i) + ":\n"
            llm_input += json.dumps(candidate_list[i], ensure_ascii=False) + "\n"

        print(rank_score_ave)
        print(len(llm_input))
        outputs = send2llm(rank_score_ave, llm_input)
        save_file(path_cascade_result_m1 + str(index) + ".txt", outputs)


def sort_llm_result():
    item_list = read_file(path_rerank_result_m1)
    result_list = []
    not_fount_count = 0
    for i in range(len(item_list)):
        item = item_list[i]
        item_llm_result = read_file(path_cascade_result_m1 + str(i) + ".txt")
        llm1 = item_llm_result.get("llm1")
        llm2 = item_llm_result.get("llm2")
        if llm1 is None:
            llm_string = llm2
        else:
            llm_string = llm1
        if llm_string == "无":
            llm_string = ""
        elif len(llm_string) > 1:
            if "无" in llm_string:  # 无正确选项
                llm_string = ""
            else:  # 备选5
                match = matches = re.findall(r"备选(?:项)?(\d+)", llm_string)
                if match:
                    llm_string = matches[-1]
                else:
                    print("error!------")
                    print(llm_string)
                    print("error!------")
        if len(llm_string) == 0:
            search_item = {}
            not_fount_count += 1
        else:
            search_item = item.get("merge_top5")[int(llm_string)]
        result_item = {"name": item.get("name"),
                       "component": item.get("component"),
                       "cure": item.get("cure"),
                       "search_item": search_item}
        result_list.append(result_item)
    save_file("5.cascade_result/medicine_1.json", result_list)
    # print(not_fount_count). # 235


if __name__ == '__main__':
    demo_request()
    # handle()
    # sort_llm_result()
