# -*- coding: UTF-8 -*-

# 将整理好的json文件保存到ES中以备查找

from elasticsearch import Elasticsearch
from elasticsearch import helpers

import io
import os
import re
import time

# test_file_path = '/Users/wuziyue/Desktop/BLC/研究组/中医/books/total/伤寒论白话解.txt'
path_books_fl = '/Users/wuziyue/Desktop/BLC/研究组/中医/中医E百网站中医古籍/3、方论'
path_books_shjk = '/Users/wuziyue/Desktop/BLC/研究组/中医/中医E百网站中医古籍/4、伤寒、金匮'


# 连接es并返回客户端
# es = Elasticsearch("http://localhost:9200")
def connect_es():
    try:
        client = Elasticsearch(hosts="http://localhost:9200", http_auth=('elastic', '-ECXN_*_UH*rG-htFtVP'),
                               scheme="http", port=9200)
        print(client.info())
        return client
    except Exception as e:
        print(e)
        return None


# 创建索引仅第一次需要
def create_index(client):
    index_body = {
        "mappings": {
            "properties": {
                "book_title": {
                    "type": "keyword"
                },
                "chapter_title": {
                    "type": "text"
                },
                "sentence_index": {
                    "type": "integer"
                },
                "sentence": {
                    "type": "text",
                    "analyzer": "smartcn"
                }
            }
        }
    }
    client.indices.create(index="medicine_e00", body=index_body)  # 中医E百网站中医古籍
    print(client.info())


def init_es():
    client = connect_es()
    create_index(client)


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print('共耗时约 {:.2f} 秒'.format(time.time() - start))
        return res

    return wrapper


# 读取段落列表
def safe_read_paragraph(path):
    para_list = []

    # 1.尝试用utf-8读取
    is_utf8_error = False
    try:
        with io.open(path, 'r', encoding='utf-8') as file:
            para_list = file.readlines()
    except Exception as e:
        is_utf8_error = True

    # 2.utf-8读取失败，再尝试用gbk读取
    is_gbk_error = False
    if is_utf8_error:
        try:
            with io.open(path, 'r', encoding='gbk') as file:
                para_list = file.readlines()
        except Exception as e:
            is_gbk_error = True

    # 3.gbk也读取失败，最后尝试用gb2312读取
    if is_gbk_error:
        try:
            with io.open(path, 'r', encoding='gb2312') as file:
                para_list = file.readlines()
        except Exception as e:
            para_list = []

    if len(para_list) == 0:
        # print(path)
        return para_list

    # 新的文本文件中未分段，内容太长不利于后续查找筛选，故按句号分割存储
    sentence_list = []
    for para in para_list:
        if len(para) == 0:
            continue
        para_sen_list = para.split("。")
        for para_sen in para_sen_list:
            if len(para_sen) == 0:
                continue
            sentence_list.append(para_sen + "。")

    return sentence_list

# curl -X PUT "localhost:9200/megacorp/employee/1?pretty" -H 'Content-Type: application/json' -d'
# {
#     "first_name" : "John",
#     "last_name" :  "Smith",
#     "age" :        25,
#     "about" :      "I love to go rock climbing",
#     "interests": [ "sports", "music" ]
# }
# '


# 抽取文件名中的汉字
# def extract_chinese(text):
#     pattern = re.compile(u'[\u4e00-\u9fa5]')  # 查找汉字
#     result = pattern.findall(text)
#     return ''.join(result)


# 保存方论
# 按书一本一本存
@timer
def save_txt_book2es(client, root_path):
    books_list = os.listdir(root_path)  # 遍历根目录
    for book in books_list:  # 一级目录书
        print(book)
        book_path = root_path + '/' + book
        if os.path.isfile(book_path):
            continue
        chapter_list = os.listdir(book_path)  # 遍历一级目录，二级目录章节

        """开始遍历一本书的内容，构造要存储的数据"""
        book_item_list = []
        for chapter in chapter_list:
            # print(chapter)
            chapter_path = book_path + '/' + chapter
            if os.path.isfile(chapter_path):
                continue
            txt_list = os.listdir(chapter_path)  # 遍历二级目录，txt文件
            for txt in txt_list:
                # print(txt)
                txt_path = chapter_path + '/' + txt
                sentence_list = safe_read_paragraph(txt_path)
                for i in range(len(sentence_list)):
                    item = {"book_title": book,
                            "chapter_title": chapter,
                            "sentence_index": i,
                            "sentence": sentence_list[i]}
                    book_item_list.append(item)
        # print(book_item_list[-1])
        print(len(book_item_list))

        """ 使用生成器批量写入数据 """
        action = ({
            "_index": "medicine_e00",
            "_source": item
        } for item in book_item_list)
        helpers.bulk(client, action)


def search_es(keywords, client):
    # "麻黄9克、桂枝6克、杏仁6克、炙甘草3克"
    # 麻黄 桂枝 杏仁 炙甘草
    search_str = ' '.join(keywords)
    search_body = {
        "query": {
            "match": {
                "sentence": search_str
            }
        }
    }
    print(client.search(index='medicine_e00', body=search_body))


def handle():
    client = connect_es()
    # save_txt_book2es(client, path_books_fl)  # fl共耗时约 128.05 秒
    save_txt_book2es(client, path_books_shjk)  # shjk共耗时约 48.88 秒


def search_test():
    client = connect_es()
    search_es(["麻黄", "桂枝", "杏仁", "炙甘草"], client)
    # search_es(["麻黄9克", "桂枝6克", "杏仁6克", "炙甘草3克"], client)
    # search_es(["麻黄汤", "麻黄", "桂枝", "杏仁", "炙甘草"], client)
    # search_es(["上四味，以水九升，先煮麻黄，减二升，去上沫，内诸药，煮取二升半，去滓，温服八合。覆取微似汗，不须啜粥，余如桂枝法将息（现代用法：水煎服，温覆取微汗）"], client)
    # search_es(["上八味，以水1升，先煮葛根、麻黄，减至800毫升，去白沫，纳诸药，煮取300毫升，去滓，温服100毫升，覆取微似汗。"], client)
    # search_es(["上七味，以水1升，先煮麻黄、葛根，减至800毫升，去上沫，纳诸药，再煮取300毫升，去滓，每次温服150毫升，覆取微似汗。"], client)


if __name__ == '__main__':
    # init_es()
    # handle()
    search_test()
