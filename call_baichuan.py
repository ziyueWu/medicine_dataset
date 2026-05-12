# -*- coding: UTF-8 -*-

import requests
import json


def call_bc(question, model_name="Baichuan-M3-Plus"):
    """
    调用API回答问题

    Args:
        question (str): 用户问题
        model_name (str): 模型名称
    Returns:
        str: API返回的答案文本
    """
    # API配置
    api_url = "https://api.baichuan-ai.com/v1/chat/completions"
    api_key = "sk-471b2c6695830b0f9f025dc59bcaf6a1"  # 需要替换为实际的API密钥

    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 请求体
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": question}
        ],
        "stream": True,
        "max_tokens": 5000,
        "temperature": 0.7
    }
    json_data = json.dumps(data)

    print("开始调用API...")

    try:
        # 发送请求
        print(f"发送请求到: {api_url}")
        # print(f"问题: {question}")

        response = requests.post(api_url, headers=headers, data=json_data, timeout=60, stream=True)
        total_result = ""
        # 检查响应状态
        if response.status_code == 200:
            # print("API请求成功，收到响应")
            print("请求成功，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    line_data_str = line_str[6:]
                    line_data_dict = json.loads(line_data_str)
                    line_data_choices = line_data_dict["choices"]
                    total_result = total_result + line_data_choices[0]["delta"]["content"]
                    # print(total_result)
            return total_result

        else:
            print(f"API请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return total_result

    except requests.exceptions.RequestException as e:
        print(f"API请求异常: {str(e)}")
        return total_result
    except json.JSONDecodeError as e:
        print(f"JSON解析异常: {str(e)}")
        return total_result
    except Exception as e:
        print(f"未知异常: {str(e)}")
        return total_result


if __name__ == "__main__":
    q = "用一段话比较桂枝加桂汤与葛根加半夏汤的区别，说明葛根、麻黄、半夏的作用以及增加葛根、麻黄、半夏后方剂主治病症有何变化"
    a = call_bc(q)
    print(a)
