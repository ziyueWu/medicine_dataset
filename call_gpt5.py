# -*- coding: UTF-8 -*-

import requests
import json


def call_gpt5(question, model_name="gpt-5.4"):
    """
    调用API回答问题
    
    Args:
        question (str): 用户问题
        model_name (str): 模型名称
    Returns:
        str: API返回的答案文本
    """
    # API配置
    api_url = "https://api.chatanywhere.tech/v1/chat/completions"
    api_key = "sk-9KEzD9CtBkOm4l8JItovNiuJboN3i2hjefZoKUDrvFnJD83j"  # 需要替换为实际的API密钥
    
    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 请求体
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": question}
        ],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    if model_name == "kimi-k2.5":
        print("kimi不要思考了")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": question}
            ],
            "max_tokens": 2000,
            "temperature": 0.7,
            "thinking": {"type": "disabled"}
        }

    print("开始调用API...")
    
    try:
        # 发送请求
        print(f"发送请求到: {api_url}")
        # print(f"问题: {question}")
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        
        # 检查响应状态
        if response.status_code == 200:
            print("API请求成功，收到响应")
            
            # 解析JSON响应
            response_data = response.json()
            
            # 提取答案文本
            if 'choices' in response_data and len(response_data['choices']) > 0:
                answer = response_data['choices'][0]['message']['content']
                print(f"解析成功，答案长度: {len(answer)} 字符")
                print(f"答案预览: {answer[:100]}...")
                
                return answer
            else:
                print("API响应格式异常，未找到答案内容")
                return None
                
        else:
            print(f"API请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API请求异常: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析异常: {str(e)}")
        return None
    except Exception as e:
        print(f"未知异常: {str(e)}")
        return None


# 示例使用
if __name__ == "__main__":
    q = "什么是人工智能？"
    a = call_gpt5(q)
    
    if a:
        print("\n=== 最终答案 ===")
        print(a)
    else:
        print("\n获取答案失败")
