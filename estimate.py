# -*- coding: UTF-8 -*-

import io
import json

path_result_m1 = 'dataset/medicine_1_check.json'


def read_file(path):
    with io.open(path, 'r', encoding='utf-8') as file:
        data_list = file.readlines()
        total_string = ''.join(data_list)
    # print(total_string[:10])

    dict_list = json.loads(total_string)
    return dict_list


def handle(file_path):
    item_list = read_file(file_path)
    found_true = 0
    found_false = 0
    not_found_true = 0
    not_found_false = 0
    for item in item_list:
        outputs = item.get("outputs")
        check = item.get("check")
        if outputs.startswith("未找到"):
            if check:
                not_found_true += 1
            else:
                not_found_false += 1
        else:
            if check:
                found_true += 1
            else:
                found_false += 1

    print("总计共 " + str(len(item_list)) + "条数据")
    print("找到对应段落的，有" + str(found_true + found_false) + "条" + ", 其中正确的" + str(
        found_true) + "条, 错误的" + str(found_false) + "条")
    print("没找到对应段落的，有" + str(not_found_true + not_found_false) + "条" + ", 其中正确的" + str(
        not_found_true) + "条, 错误的" + str(not_found_false) + "条")


"""
总计共 721条数据
找到对应段落的，有373条, 其中正确的305条, 错误的68条
没找到对应段落的，有348条, 其中正确的309条, 错误的39条
"""


def demo():
    text = """六味地黄丸地黄（砂仁酒拌、九蒸九晒）八两 山茱肉（酒润）山药四两 茯苓（乳拌） 丹皮 泽泻三两蜜丸。
大承气汤大黄四两（酒洗） 芒硝三合 枳实五枚 浓朴半斤先煎朴、实，将熟内大黄，煮二、三沸，倾碗内和芒硝服。
此方非但治肝肾不足，实三阴并治之剂。
攻下之法，原因实症俱备，危在旦夕，失此不下，不可复救。
有熟地之腻补肾水，即有泽泻之宣泄肾浊以济之；有萸肉之温涩肝经，即有丹皮之清泻肝火以佐之；有山药收摄脾经，即有茯苓之淡渗脾湿以和之。
故用斩关夺门之法，定难于俄顷之间，仲景所以有急下存阴之训也。
药止六味，而大开大合，三阴并治，洵补方之正鹄也。
乃后人不明此义，有谓于攻下药中，兼行生津润导之法，则存阴之力更强，殊不知一用生津滋润之药，则互相牵制，而荡涤之力轻矣！此譬如寇盗当前，恣其焚掠，所过为墟，一旦聚而歼之，然后人得安居，而元气可以渐复。
附桂八味丸熟地八两 山茱肉四两 山药四两 茯苓三两 丹皮三两 泽泻三两 附子一两 肉桂一两蜜丸。
附桂八味为治命肾虚寒之正药，亦导龙归海之妙法。"""
    a = text.index("六味地黄丸")
    print(a)
    start = text.index("有熟地之腻补肾水")
    print(start)
    print(len("有熟地之腻补肾水，即有泽泻之宣泄肾浊以济之；有萸肉之温涩肝经，即有丹皮之清泻肝火以佐之；有山药收摄脾经，即有茯苓之淡渗脾湿以和之。"))


if __name__ == '__main__':
    # handle(path_result_m1)
    demo()
