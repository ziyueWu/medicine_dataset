# -*- coding: UTF-8 -*-

import requests
import json
import io

# base_url = "https://api.chatanywhere.tech/v1"

chat_anywhere_url = "https://api.chatanywhere.tech/v1/chat/completions"
api_key = "sk-cuuMbKx1BzAG6ZHOwgzYXRtqzfO9nTzUzn0l2Rn3lxMkAqad"
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
        resp = requests.post(url=chat_anywhere_url, json=data, headers=headers)
        # resp.raise_for_status()  # 检查 HTTP 请求是否成功

        # 打印结果
        print("request 调用成功，返回结果:")
        print(resp.json())
        return resp.json()

    except requests.exceptions.RequestException as e:
        print("调用 request 时发生错误: {e}")
        return {}


def demo_request():
    data = {"messages": [{"role": "user",
                          "content": """
我会给你一段文本和一个json，需要你给出json中各个字段的值出现在了文本的哪里，我能保证这段文本和json是有关联的。

例如：
#文本段落：
六味地黄丸地黄（砂仁酒拌、九蒸九晒）八两 山茱肉（酒润）山药四两 茯苓（乳拌） 丹皮 泽泻三两蜜丸。
大承气汤大黄四两（酒洗） 芒硝三合 枳实五枚 浓朴半斤先煎朴、实，将熟内大黄，煮二、三沸，倾碗内和芒硝服。
此方非但治肝肾不足，实三阴并治之剂。
攻下之法，原因实症俱备，危在旦夕，失此不下，不可复救。
有熟地之腻补肾水，即有泽泻之宣泄肾浊以济之；有萸肉之温涩肝经，即有丹皮之清泻肝火以佐之；有山药收摄脾经，即有茯苓之淡渗脾湿以和之。
故用斩关夺门之法，定难于俄顷之间，仲景所以有急下存阴之训也。
药止六味，而大开大合，三阴并治，洵补方之正鹄也。
乃后人不明此义，有谓于攻下药中，兼行生津润导之法，则存阴之力更强，殊不知一用生津滋润之药，则互相牵制，而荡涤之力轻矣！此譬如寇盗当前，恣其焚掠，所过为墟，一旦聚而歼之，然后人得安居，而元气可以渐复。
附桂八味丸熟地八两 山茱肉四两 山药四两 茯苓三两 丹皮三两 泽泻三两 附子一两 肉桂一两蜜丸。
附桂八味为治命肾虚寒之正药，亦导龙归海之妙法。
#对应json：
{"name" : "六味地黄丸" 
"component" : "熟地黄24克，山萸肉12克，山药12克，泽泻9克，茯苓9克，牡丹皮9克。" 
"cure" : "用于肾阴亏损，头晕耳鸣，腰膝酸软，骨蒸潮热，盗汗遗精，五心烦热，咽干颧红。 "}

[{"key":"name",
"value":"六味地黄丸",
"text_value":"六味地黄丸"},
{"key":"component",
"value":"熟地黄24克",
"text_value_name":"地黄（砂仁酒拌、九蒸九晒）",
"text_value_count":"八两"},
{"key":"component",
"value":"山萸肉12克",
"text_value_name":"山茱肉（酒润）",
"text_value_count":"四两"},
……
{"key":"cure",
"value":"用于肾阴亏损，头晕耳鸣，腰膝酸软，骨蒸潮热，盗汗遗精，五心烦热，咽干颧红。 ",
"text_value":"有熟地之腻补肾水，即有泽泻之宣泄肾浊以济之；有萸肉之温涩肝经，即有丹皮之清泻肝火以佐之；有山药收摄脾经，即有茯苓之淡渗脾湿以和之。"]

如果你理解了，请补充上述不完整的返回结果。"""}],
            "model": "gpt-5-nano",
            "temperature": 0.2}
    result = send_request(data)
    # answer = result.get("answer")
    print(result)


if __name__ == '__main__':
    demo_request()
