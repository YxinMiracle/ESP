# 数据预处理模板类，用户适配PFN Model
import json
import logging
import os
import pathlib
import random
from argparse import ArgumentParser
from typing import List, Tuple

from process_model.model.ReBaseData import ReSentBaseData

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessTrainDataTemplate:
    def __init__(self, data: List[ReSentBaseData], params: ArgumentParser):
        self.sent_data_list = data  # type: List[ReSentBaseData]
        self.params = params  # type: ArgumentParser
        self.root_dir = str(pathlib.Path(__file__).resolve().parent.parent)  # type: str
        self.data_directory_path = self.root_dir + os.path.sep + self.params.data_directory_name

    # 构建实体和关系数据，都是模型所需要的
    def save_ner_and_re_idx_json_data(self):
        """
        处理数据集中的数据，将实体和关系数据处理为idx，并保存为json文件
        :return:
        """
        logging.info("开始处理实体与关系idx数据")

        if not os.path.exists(self.data_directory_path):
            os.makedirs(self.data_directory_path)

        ner2idx_file_name = self.data_directory_path + os.path.sep + self.params.ner2idx_file_name  # type: str
        rel2idx_file_name = self.data_directory_path + os.path.sep + self.params.re2idx_file_name  # type: str
        if os.path.exists(ner2idx_file_name) and os.path.exists(rel2idx_file_name):
            # 如果两个文件都有了，那就不往下继续了
            logging.info("index文件已存在，不需要继续执行")
            return
        entity_type_set, relation_type_set = set(), set()  # type: set
        for sent_data in self.sent_data_list:
            entity_type_set.add(sent_data.head_entity.entity_type)
            entity_type_set.add(sent_data.tail_entity.entity_type)
            relation_type_set.add(sent_data.relation_type)
        ner2idx_dict = {entity_type: index for index, entity_type in enumerate(list(entity_type_set))}  # type: dict
        rel2idx_dict = {relation_type: index for index, relation_type in
                        enumerate(list(relation_type_set))}  # type: dict

        json.dump(ner2idx_dict, open(ner2idx_file_name, "w"))
        json.dump(rel2idx_dict, open(rel2idx_file_name, "w"))
        logging.info("完成idx数据预准备")

    def get_head_tail_entities_index_in_sent(self, sent_token_list: List[str],
                                             head_entity_list: List[str],
                                             tail_entity_list: List[str]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        获取头实体和尾实体在token_list中的连续位置，用于最后取出相应的嵌入进行分类任务。确保连续x
        :param sent_token_list: 句子分词列表
        :param head_entity_list: 头实体的连续词列表
        :param tail_entity_list: 尾实体的连续词列表
        :return: 头实体和尾实体的起始索引组成的元组
        """

        def find_sublist_indices(main_list: List[str], sublist: List[str]) -> List[List[int]]:
            """查找子列表在主列表中的所有连续出现位置的起始索引"""
            sublist_len = len(sublist)
            res_list = []  # type: List[List[int]]
            for i in range(len(main_list) - sublist_len + 1):
                if main_list[i:i + sublist_len] == sublist:
                    res_list.append([index for index in range(i, i + sublist_len)])
            return res_list

        head_entity_index_in_sent = find_sublist_indices(sent_token_list, head_entity_list)  # type: List[List[int]]
        tail_entity_index_in_sent = find_sublist_indices(sent_token_list, tail_entity_list)  # type: List[List[int]]

        return head_entity_index_in_sent, tail_entity_index_in_sent

    def save_sent_ner_re_json_data(self):
        logging.info("开始处理训练数据和测试数据")
        if not os.path.exists(self.data_directory_path):
            os.makedirs(self.data_directory_path)

        train_file_path = self.data_directory_path + os.path.sep + self.params.train_file_name  # type: str
        test_file_path = self.data_directory_path + os.path.sep + self.params.test_file_name  # type: str
        if os.path.exists(train_file_path) and os.path.exists(test_file_path):
            # 如果两个文件都有了，那就不往下继续了
            logging.info("训练数据和测试数据文件已存在，不需要继续执行")
            return
        total_data_list = []  # type: List[dict]
        for sent_data in self.sent_data_list:
            sent_dict = {}  # type: dict
            sent_dict["tokens"] = sent_data.fine_tuned_re_model_tokens
            head_entity_index_in_sent, tail_entity_index_in_sent = self.get_head_tail_entities_index_in_sent(
                sent_token_list=sent_data.fine_tuned_re_model_tokens,
                head_entity_list=sent_data.head_entity.entity_name_list,
                tail_entity_list=sent_data.tail_entity.entity_name_list)
            head_entity_list, tail_entity_list = [], []  # type: list
            for head_entity_index_list in head_entity_index_in_sent:
                head_entity_list.append({
                    "type": sent_data.head_entity.entity_type,
                    "start": head_entity_index_list[0],
                    "end": head_entity_index_list[-1] + 1
                })
            for tail_entity_index_list in tail_entity_index_in_sent:
                tail_entity_list.append({
                    "type": sent_data.tail_entity.entity_type,
                    "start": tail_entity_index_list[0],
                    "end": tail_entity_index_list[-1] + 1
                })
            relation_list = []  # list[dict]
            for head_index in range(len(head_entity_list)):
                for tail_index in range(len(tail_entity_list)):
                    relation_list.append({
                        "type": sent_data.relation_type,
                        "head": head_index,
                        "tail": head_index + tail_index + 1
                    })
            sent_dict["entities"] = head_entity_list + tail_entity_list
            sent_dict["relations"] = relation_list
            total_data_list.append(sent_dict)
        random.shuffle(total_data_list)
        split = int(0.15 * len(total_data_list))
        train_data = total_data_list[split:]  # type: list[dict]
        test_data = total_data_list[:split]  # type: list[dict]

        # 保存文件
        json.dump(train_data, open(train_file_path, "w"))
        json.dump(test_data, open(test_file_path, "w"))
        logging.info("训练数据已保存在{}和测试数据已保存在{}".format(train_file_path, test_file_path))

    def do_process(self):
        # 第一步，构建训练所需要的idx文件
        self.save_ner_and_re_idx_json_data()
        # 第二部，构建句子、实体、关系数据
        self.save_sent_ner_re_json_data()
        pass
