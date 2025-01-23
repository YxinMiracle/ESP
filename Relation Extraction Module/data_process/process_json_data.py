import json
import os

from huawei_data_process.config import cti_labelId_2_labelName_dict
from process_model.model.Entity import Entity
from process_model.model.ReBaseData import ReSentBaseData


# 获取文件夹下面的所有文件
def list_files(directory: str) -> list:
    files = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path):
            files.append(entry)
    return files


def is_in_single_sent(label_data_list: list):
    """
    这里是判断句子中是否存在跨句子标注的情况
    这里就是拿这个句子中出现实体存在的句子id来进行判断
    要是说两个实体所存在的句子id不一样，diff 是要不为0，那么就是存在跨句子的情况
    :param label_data_list:
    :return:
    """
    for label_data in label_data_list:
        label_item_id = label_data["id"]
        label_id_split_list = str(label_item_id).split(":")
        if len(label_id_split_list) < 11:
            sentence_start_id = label_data["sidS"]
            sentence_end_id = label_data["sidE"]
            diff = sentence_end_id - sentence_start_id
            if diff != 0:
                return False
    return True


def save_data(token_list, bio_label_list):
    with open("temp_data/train2.txt", "a+", encoding="UTF-8") as fp:
        for word_l, word_type_l in zip(token_list, bio_label_list):
            for word, word_type in zip(word_l, word_type_l):
                fp.write(word + " " + word_type + "\n")
            fp.write("\n")


entity_id_dict = {}


def process_single_sentence(label_data_list: list, token_list: list):
    """
    处理每一个单独的情报文件
    :param cti_json_data: 这个情报文件中的数据
    :return:
    """
    bio_label_list = ["O" for _ in range(len(token_list))]
    if len(label_data_list) == 0:  # 说明这个句子没有需要标注的信息
        return bio_label_list
    else:
        for label_data in label_data_list:  # 这里是一个类别的操作
            label_item_id = label_data["id"]
            label_id_split_list = str(label_item_id).split(":")
            if len(label_id_split_list) < 11:  # 这里是确保，这里只是词语的标注，并不是关系
                token_id = label_data["l"]
                if token_id == "": continue
                token_type = cti_labelId_2_labelName_dict[token_id]
                start_index, end_index = label_data["s"], label_data["e"]
                sub_word_list = []
                bio_label_list[start_index] = "B-" + token_type
                sub_word_list.append(token_list[start_index])
                if start_index != end_index:  # 说明这个类型是只有一个单词
                    for index in range(start_index + 1, end_index + 1):
                        bio_label_list[index] = "I-" + token_type
                        sub_word_list.append(token_list[index])
                # 存储构建图谱需要的数据
                entity_id_dict[label_item_id] = [token_type, " ".join(sub_word_list)]
    return bio_label_list


def process_cross_sentence(cross_sent_data_list: list):
    """
    处理跨句子问题
    先把这些句子合起来，也就是token进行相加
    这种跨句子的只会出现一个单词短语跨句子的情况
    :param cross_sent_data_list:
    :return:
    """
    new_token_list = []  # type: list
    type2index_dict = {}
    type2index_dict["single_data"] = []  # type: list
    for cross_sent_index, cross_sent in enumerate(cross_sent_data_list):
        token_list = cross_sent["tokens"]
        for label_data in cross_sent["labels"]:
            label_item_id = label_data["id"]
            label_id_split_list = str(label_item_id).split(":")
            if len(label_id_split_list) < 11:  # 这里是确保，这里只是词语的标注，并不是关系
                sid_S, sid_E = label_data["sidS"], label_data["sidE"]
                start_index, end_index = label_data["s"], label_data["e"]
                token_id = label_data["l"]
                token_type = cti_labelId_2_labelName_dict[token_id]
                if sid_S != sid_E:  # 这个实体是跨句子的
                    if cross_sent_index == 0:
                        type2index_dict["cross_data"] = {"word_type": token_type,
                                                         "index_data": [i for i in range(start_index, len(token_list))]}
                    else:
                        type2index_dict["cross_data"]["index_data"].extend(
                            [type2index_dict["cross_data"]["index_data"][-1] + i for i in range(1, 1 + end_index + 1)])
                else:  # 这个实体就这一个,同时也要考虑到前面句子的长度
                    type2index_dict["single_data"].append({"word_type": token_type,
                                                           "index_data": [len(new_token_list) + i for i in
                                                                          range(start_index, end_index + 1)]})
        new_token_list.extend(token_list)

    bio_label_list = ["O" for _ in range(len(new_token_list))]
    for idx, cross_word_index in enumerate(type2index_dict["cross_data"]["index_data"]):
        if idx == 0:
            bio_label_list[cross_word_index] = "B-" + type2index_dict["cross_data"]["word_type"]
        else:
            bio_label_list[cross_word_index] = "I-" + type2index_dict["cross_data"]["word_type"]

    for single_word_data in type2index_dict["single_data"]:
        for idx, single_word_index in enumerate(single_word_data["index_data"]):
            if idx == 0:
                bio_label_list[single_word_index] = "B-" + single_word_data["word_type"]
            else:
                bio_label_list[single_word_index] = "I-" + single_word_data["word_type"]

    return new_token_list, bio_label_list


def process_relation_id(label_data_list: list) -> set:
    relation_id_set = set()  # type: set
    for label_data in label_data_list:
        label_item_id = label_data["id"]
        label_id_split_list = str(label_item_id).split(":")
        if len(label_id_split_list) > 11:  # 这里是确保，这里拿到的数据都是关系
            relation_id_set.add(str(label_item_id))
    return relation_id_set


res_relation_list = []
each_cti_data = {}
import re


def format_apt(input_string):
    # 定义正则表达式
    pattern = r'(APT)\s+(\d+)'

    # 使用正则表达式的sub函数替换字符串
    result = re.sub(pattern, r'\1\2', input_string)

    return result


def clear_str(node_name: str):
    if node_name.endswith("."):
        node_name = node_name[:-1]
    if node_name.startswith("“"):
        node_name = node_name[1:]
    if node_name.endswith("”"):
        node_name = node_name[:-1]
    if node_name.startswith("("):
        node_name = node_name[1:]
    if node_name.endswith(")"):
        node_name = node_name[:-1]
    if node_name.endswith("),"):
        node_name = node_name[:-2]
    if node_name.endswith("’s"):
        node_name = node_name[:-2]
    # elif node_name.endswith("),"):
    #     node_name = node_name[:-2]
    if node_name.endswith(","):
        node_name = node_name[:-1]
    node_name = format_apt(node_name)
    return node_name


def build_re_task_data(s_entity_id, e_entity_id, relation_type, sentences_data):
    """
    用来构建re任务的主要数据
    :param s_entity_id: 头节点在这个cti文章中的id
    :param e_entity_id: 尾节点在这个cti文章中的id
    :param relation_type: 这两个节点之间的关系类别
    :return:
    """
    # 1. 我需要获取头节点的所在句子的id和尾节点所在句子的id
    start_node_sentence_id_in_sentence = int(s_entity_id.split(":")[2])
    end_node_sentence_id_in_sentence = int(e_entity_id.split(":")[2])

    s_entity_information = entity_id_dict[s_entity_id]
    e_entity_information = entity_id_dict[e_entity_id]

    # 2. 分两种情况进行考虑，第一种情况就是这两个实体并不在一个句子里面
    if start_node_sentence_id_in_sentence != end_node_sentence_id_in_sentence:
        start_id, end_id = start_node_sentence_id_in_sentence, end_node_sentence_id_in_sentence
        if start_node_sentence_id_in_sentence > end_node_sentence_id_in_sentence:
            start_id = end_node_sentence_id_in_sentence
            end_id = start_node_sentence_id_in_sentence

        sent_str = " ".join(sentences_data[start_id]["tokens"]) + " " + " ".join(sentences_data[end_id]["tokens"])
        token_list = sentences_data[start_id]["tokens"] + sentences_data[end_id]["tokens"]
        # print(
        #     f"句子为：{sent_str}\n需要判断的实体类型为如下:\n头节点名称:{s_entity_information[1]}头节点类型:{s_entity_information[0]}。\n尾节点名称: {e_entity_information[1]} 尾节点类型:{e_entity_information[0]}\n他们的关系类型为：{relation_type}")
        # print("====================================================================")
    else:
        sent_str = " ".join(sentences_data[start_node_sentence_id_in_sentence]["tokens"])
        token_list = sentences_data[start_node_sentence_id_in_sentence]["tokens"]
        # print(
        #     f"句子为：{sent_str}\n需要判断的实体类型为如下:\n头节点名称:{s_entity_information[1]}头节点类型:{s_entity_information[0]}。\n尾节点名称: {e_entity_information[1]} 尾节点类型:{e_entity_information[0]}\n他们的关系类型为：{relation_type}")
        # print("===================================================================")
    head_entity = Entity(entity_name=s_entity_information[1], entity_type=s_entity_information[0])
    tail_entity = Entity(entity_name=e_entity_information[1], entity_type=e_entity_information[0])
    re_base_data = ReSentBaseData(sent=sent_str, head_entity=head_entity, tail_entity=tail_entity,
                                  relation_type=relation_type, sent_token_list=token_list)
    return re_base_data


# 这是两个实体都在一个句子里面
def process_file(cti_json_data: dict, index):
    """
    处理一篇情报
    :param cti_json_data:
    :return:
    """
    global entity_id_dict
    sentences_data = cti_json_data["sentences"]
    cross_sent_index_set = set()  # type: set
    sent_list, label_list = [], []
    relation_id_set = set()  # type: set
    for sent_index in range(len(sentences_data)):
        sentences = sentences_data[sent_index]
        token_list = sentences["tokens"]  # 单词列表
        if len(token_list) == 1 and token_list[0] == "":
            # 先判断一下cross_sent_index_set有没有残留的数据, 如果有的话那就需要去处理一下之前残留的数据
            if len(cross_sent_index_set) > 0:
                cross_sent_index_list = sorted(list(cross_sent_index_set))
                cross_sent_data_list = [sentences_data[i] for i in cross_sent_index_list if i < len(sentences_data)]
                cross_sent_token_list, bio_label_list = process_cross_sentence(cross_sent_data_list)
                sent_list.append(cross_sent_token_list)
                label_list.append(bio_label_list)
                cross_sent_index_set = set()
            continue  # 如果这个是一个空句子数据的话那就跳过
        label_data_list = sentences["labels"]  # label数据
        # =============================================处理关系========================================================
        s_relation_id_set = process_relation_id(label_data_list)
        relation_id_set = relation_id_set | s_relation_id_set
        in_single = is_in_single_sent(label_data_list)
        # =============================================处理实体========================================================
        if in_single:
            # 先判断一下cross_sent_index_set有没有残留的数据, 如果有的话那就需要去处理一下之前残留的数据
            if len(cross_sent_index_set) > 0:
                cross_sent_index_list = sorted(list(cross_sent_index_set))
                cross_sent_data_list = [sentences_data[i] for i in cross_sent_index_list if i < len(sentences_data)]
                cross_sent_token_list, bio_label_list = process_cross_sentence(cross_sent_data_list)
                sent_list.append(cross_sent_token_list)
                label_list.append(bio_label_list)
                cross_sent_index_set = set()
            # ========================================================================================================
            # 说明这个句子中的所有标注都是在单独的一个句子中的
            bio_label_list = process_single_sentence(label_data_list, token_list)
            sent_list.append(token_list)
            label_list.append(bio_label_list)
        else:
            # 说明这个标注是存在跨句的情况
            cross_sent_index_set.add(sentences["id"])

    # ============================================== 构建三元组 =====================================================
    relation_id_list = list(relation_id_set)
    article_re_task_res = []
    for relation_id_str in relation_id_list:
        relation_id_str_list = str(relation_id_str).split(":")
        s_entity_id = ":".join(relation_id_str_list[2:11])
        relation_id = relation_id_str_list[0]
        e_entity_id = ":".join(relation_id_str_list[11:-1])
        try:
            s_entity_information = entity_id_dict[s_entity_id]
            e_entity_information = entity_id_dict[e_entity_id]
            relation_type = cti_labelId_2_labelName_dict[relation_id]
            res_relation_list.append(
                [s_entity_information[1], s_entity_information[0], relation_type, e_entity_information[1],
                 e_entity_information[0]])
            """
                现在我要做一个事情，那就是我要将两个有关系的实体，以及对应的句子存储起来，方便我后续进行关系抽取的任务
                1. 为了和论文的实现结构相符合，我准备这样子做
                    首先我需要获取两个实体对应的句子id
                    然后我需要去判断这个句子id是不是一样
                    要是是一样的话我就随便拿出一个出来就好
                2. 要是是不一样的话
                    我需要把这两个句子进行一个拼接，注意需要有前后的关系，前后就用句子id来进行判断就好
                    然后存放到数据库里面
            """
            re_base_data = build_re_task_data(s_entity_id, e_entity_id, relation_type, sentences_data)
            article_re_task_res.append(re_base_data)
        except:
            pass

    entity_id_dict = {}
    # return sent_list, label_list
    return article_re_task_res


def main():
    files = list_files("./huawei_json")
    for index, file_name in enumerate(files):
        with open(
                f"./huawei_json/{file_name}",
                "r", encoding="UTF-8") as fp:
            cti_json_data = json.load(fp)
            each_cti_data[str(index)] = {}
            # sent_list, label_list = process_file(cti_json_data, index)
            article_re_task_res = process_file(cti_json_data, index)
        # save_data(sent_list, label_list)

    # with open("truple_data.pkl", "wb") as fp:
    #     pickle.dump(res_relation_list, fp)


if __name__ == '__main__':
    main()
    # print(clear_str("(APT31)"))
