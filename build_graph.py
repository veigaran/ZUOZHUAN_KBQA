#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2021-04-19 21:14
@Author:Veigar
@File: build_graph.py
@Github:https://github.com/veigaran
"""

from py2neo import Graph, Node, Relationship
import pandas as pd
import re
import os


class PersonGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, r'F:\OneDrive\课程学习资料\大四\0-毕业论文\数据\data.csv')
        self.graph = Graph("http://localhost:7474", username="neo4j", password="123456")

    def read_file(self):
        """
        读取文件，获得实体，实体关系
        :return:
        """

        # 实体
        person = []  # 人物
        alias = []  # 别名
        surname = []  # 姓
        country = []  # 国家
        school = []  # 学派
        rank = []  # 等级
        field = []  # 领域

        person_infos = []

        # 关系
        person_is_alias = []
        person_is_surname = []  # 人物的姓氏
        person_is_country = []
        person_is_rank = []  # 人物的等级
        person_field = []  # 人物的领域
        person_is_school = []
        person_is_father = []  # 人物的父亲
        person_is_mother = []  # 人物的母亲
        person_is_children = []  # 人物的子女
        person_is_wife = []  # 人物的妻子
        person_is_brother = []  # 人物的兄弟
        jun_cheng = []  # 君臣关系

        all_data = pd.read_csv(self.data_path, encoding='utf').loc[:, :].values
        for data in all_data:
            person_dict = {}
            person = str(data[0]).replace("...", " ").strip()
            person_dict["person"] = person

            # 别名
            alias_list = str(data[1]).strip().split() if str(data[1]) else "未知"
            for al in alias_list:
                alias.append(al)
                person_is_alias.append([person, al])

            # 姓
            surname_list = str(data[2]).strip().split() if str(data[2]) else "未知"
            for s in surname_list:
                surname.append(s)
                person_is_surname.append([person, s])

            # 国家
            country_list = str(data[3]).strip().split() if str(data[3]) else "未知"
            for c in country_list:
                country.append(c)
                person_is_country.append([person, c])

            # 等级
            rank_list = str(data[4]).strip().split() if str(data[4]) else "未知"
            for r in rank_list:
                rank.append(r)
                person_is_rank.append([person, r])
                # person_is_country.append([person, c])

            # 领域
            field_list = str(data[5]).strip().split() if str(data[5]) else "未知"
            for f in field_list:
                field.append(f)
                person_field.append([person, f])

            # 学派
            school_list = str(data[6]).strip().split() if str(data[6]) else "未知"
            for sc in school_list:
                school.append(sc)
                person_is_school.append([person, sc])

            # 父亲
            father_list = str(data[7]).strip().split() if str(data[7]) else "未知"
            for fa in father_list:
                person_is_father.append([person, fa])

            # 母亲
            mother_list = str(data[8]).strip().split() if str(data[8]) else "未知"
            for mo in mother_list:
                person_is_mother.append([person, mo])

            # 子女
            children_list = str(data[9]).strip().split() if str(data[9]) else "未知"
            for ch in children_list:
                person_is_children.append([person, ch])

            # 妻子
            wife_list = str(data[10]).strip().split() if str(data[10]) else "未知"
            for w in wife_list:
                person_is_wife.append([person, w])

            # 兄弟
            brother_list = str(data[11]).strip().split() if str(data[11]) else "未知"
            for b in brother_list:
                person_is_brother.append([person, b])

            # 君臣
            juncheng_list = str(data[12]).strip().split() if str(data[12]) else "未知"
            for jc in juncheng_list:
                jun_cheng.append([person, jc])

            # 添加每个属性
            #  氏
            shi = str(data[13]).strip()
            person_dict["shi"] = shi

            # 名
            ming = str(data[14]).strip()
            person_dict["ming"] = ming

            # 谥号
            sihao = str(data[15]).strip()
            person_dict["sihao"] = sihao

            # 生卒年
            birth_death = str(data[16]).strip()
            person_dict["birth_death"] = birth_death

            # 在位时间
            office_time = str(data[17]).strip()
            person_dict["office_time"] = office_time

            # 事件
            event = str(data[18]).strip()
            person_dict["event"] = event

            # 作品
            work = str(data[19]).strip()
            person_dict["work"] = work

            person_infos.append(person_dict)

        #  person_is_father, person_is_mother, person_is_children, \
        #         person_is_wife, person_is_brother, jun_cheng,
        return set(person), set(alias), set(surname), set(country), set(rank), set(field), set(school), person_is_alias, \
               person_is_surname, person_is_country, person_is_rank, person_field, person_is_school,person_is_father, person_is_mother, person_is_children, \
               person_is_wife, person_is_brother, jun_cheng,person_infos

    def create_node(self, label, nodes):
        """
        创建节点
        :param label: 标签
        :param nodes: 节点
        :return:
        """
        count = 0
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.graph.create(node)
            count += 1
            print(count, len(nodes))
        return

    def create_person_nodes(self, person_info):
        count = 0
        for person_dict in person_info:
            # node = Node("Person", name=person_dict["person"])
            node = Node("Person", name=person_dict["person"], shi=person_dict["shi"], ming=person_dict["ming"],
                        sihao=person_dict["sihao"], birth_death=person_dict["birth_death"],
                        office_time=person_dict["office_time"], event=person_dict["event"], work=person_dict["work"])
            self.graph.create(node)
            count += 1
            print(count)
        return

    def create_graphNodes(self):
        person, alias, surname, country, rank, field, school, person_is_alias, person_is_surname, person_is_country, \
        person_is_rank, person_field, person_is_school, person_is_father, person_is_mother, person_is_children, \
               person_is_wife, person_is_brother, jun_cheng,person_infos = self.read_file()
        self.create_person_nodes(person_infos)
        self.create_node("Person", person)
        self.create_node("alias", alias)
        self.create_node("surname", surname)
        self.create_node("country", country)
        self.create_node("rank", rank)
        self.create_node("field", field)
        self.create_node("school", school)
        return

    def create_graphRels(self):
        person, alias, surname, country, rank, field, school, rl_alias, rl_surname, rl_country, \
        rl_rank, rl_field, rl_school, person_is_father, person_is_mother, person_is_children, \
               person_is_wife, person_is_brother, jun_cheng,person_infos = self.read_file()
        # print(person_is_alias)
        self.create_relationship("Person", "alias", rl_alias, "person_is_alias", "别名是")
        self.create_relationship("Person", "surname", rl_surname, "person_is_surname", "姓为")
        self.create_relationship("Person", "country", rl_country, "person_is_country", "所属国家是")
        self.create_relationship("Person", "rank", rl_rank, "person_is_rank", "等级是")
        self.create_relationship("Person", "field", rl_field, "person_field", "领域是")
        self.create_relationship("Person", "school", rl_school, " person_is_school", "学派是")
        self.create_relationship("Person", "Person", person_is_father, "person_is_father", "父亲是")
        self.create_relationship("Person", "Person", person_is_mother, "person_is_mother", "母亲是")
        self.create_relationship("Person", "Person", person_is_children, "person_is_children", "子女是")
        self.create_relationship("Person", "Person", person_is_wife, "person_is_wife", "配偶是")
        self.create_relationship("Person", "Person", person_is_brother, " person_is_brother", "兄弟是")
        self.create_relationship("Person", "Person", jun_cheng, " jun_cheng", "君臣是")

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        """
        创建实体关系边
        :param start_node:
        :param end_node:
        :param edges:
        :param rel_type:
        :param rel_name:
        :return:
        """
        count = 0
        # 去重处理
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.graph.run(query)
                count += 1
                print(rel_type, count, all)
            except Exception as e:
                print(e)
        return


if __name__ == "__main__":
    handler = PersonGraph()
    handler.create_graphNodes()
    handler.create_graphRels()
