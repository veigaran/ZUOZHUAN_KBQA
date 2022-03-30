#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2020-06-09 13:10
@Author:Veigar
@File: 1.py
@Github:https://github.com/veigaran
"""
dic1 = {'entities': [{'end': 6, 'start': 0, 'type': 'DRU', 'word': '阿莫西林胶囊'}],
        'string': '阿莫西林胶囊怎么吃？'}
entity = dic1['entities']
print(entity, type(entity))
print('---------------')

dict2 = entity[0]
print(dict2, type(dict2))
print('---------------')

entity_type = dict2['type']
word = dict2['word']
print(entity_type, '\n', word)

t = dic1["entities"][0]['type']
print(t)
