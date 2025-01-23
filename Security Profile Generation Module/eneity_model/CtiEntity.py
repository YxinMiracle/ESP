import spacy
from transformers import AutoTokenizer
import torch.nn as nn
import torch
import numpy as np

from backend_server.entity_server_utils import add_cti_chunk, update_cti_content_by_cti_id, get_item_type_id, add_entity_data, get_relation_type_id, \
    get_father_item_id, get_detail_entity_id, add_final_relation_data
from eneity_model.YxinMiracleModel import AttAttYxinNerModel_POS
from eneity_model.entity_config import BERT_MODEL_NAME, POS_LIST, VALID_WORD_POSITION, NER_MODEL_PATH, NER_MODEL_OPTIMIZER_PATH, ENTITY_LABEL_LIST, \
    GRAPH_RELATION_CONFIG

nlp = spacy.load('en_core_web_sm')


class CtiEntityModel:
    def __init__(self):
        self.auto_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, local_files_only=True)
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index
        self.model = AttAttYxinNerModel_POS()
        self.model.cuda()
        self.model.eval()
        self.model.load_state_dict(
            torch.load(NER_MODEL_PATH))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)
        self.optimizer.load_state_dict(
            torch.load(NER_MODEL_OPTIMIZER_PATH))

    def get_word_label_pos_index_list(self, sent_list: list):
        """
        输入是情报的句子列表，我需要对句子列表中的每一个句子进行处理
        处理的过程是
        1. 对于每一个句子中的每一个词语，转为对应在bert词表中的索引位置
        2. true_word_index记录真实的word在tokenizer列表中处于第几个位置
        3. 我们利用spcay，获取对应的每个句子中的每一个词语的词性，然后获取在词性列表中的下标位置
        :param sent_list:
        :return: 返回的数据是什么？ 返回的数据一共是三个列表，每一个列表都是一个二维数组，以每一个句子为单位
        """
        # 这里就是返回的三个列表
        result_word_index_list, result_true_word_index, result_pos_list = [], [], []  # type: list
        for sent in sent_list:  # 循环每一个句子
            # 由于这是一个句子，所以我们需要对她进行分词，分词工具用的是spacy
            word_list, pos_list = [], []  # 把sent转为一个个word存放在word_list中，做一个数据预处理，防止有空格

            word_index_list, true_word_index, true_pos_list = [], [], []
            for word_in_sent in nlp(str(sent)):
                word_list.append(str(word_in_sent).strip())
                try:
                    pos_list.append(POS_LIST.index(word_in_sent.pos_))
                except:
                    pos_list.append(0)

            # 然后使用BERT，将每一个word转为对应的tokenizer
            for word_num, word_in_sent in enumerate(word_list):
                word_tokenizer_list = self.auto_tokenizer.tokenize(word_in_sent)
                word_index_list.extend(  # 构建好对应的bert词表中下标
                    self.auto_tokenizer.convert_tokens_to_ids(word_tokenizer_list)
                )

                # 只有第一个位置是真正需要识别的，以后给pred出来的word用的东西
                true_word_index.extend([VALID_WORD_POSITION] + [self.pad_token_label_id] * (len(word_tokenizer_list) - 1))
                # 只有第一个位置是真正需要识别的，这个是pos的东西
                true_pos_list.extend([pos_list[word_num]] + [self.pad_token_label_id] * (len(word_tokenizer_list) - 1))

            result_word_index_list.append(
                [self.auto_tokenizer.cls_token_id] + word_index_list[:508] + [self.auto_tokenizer.sep_token_id]
            )
            result_true_word_index.append(
                [self.pad_token_label_id] + true_word_index[:508] + [self.pad_token_label_id]
            )
            result_pos_list.append(
                [self.pad_token_label_id] + true_pos_list[:508] + [self.pad_token_label_id]
            )
        return result_word_index_list, result_true_word_index, result_pos_list

    def predict_entity(self, sent_list: list):
        """
        进行预测
        """
        result_word_index_list, result_true_word_index, result_pos_list = self.get_word_label_pos_index_list(sent_list)
        result_word_list, result_label_list = [], []
        for sent_index, (word_index_list, true_word_index, pos_list) in enumerate(
                zip(result_word_index_list, result_true_word_index, result_pos_list)):
            pred_list = []
            with torch.no_grad():
                X_torch = torch.tensor([word_index_list], device="cuda").long()
                POS_torch = torch.tensor([pos_list], device="cuda").long()
                preds = self.model.test(X_torch, pos_data=POS_torch)
                pred_list.extend(preds.data.cpu().numpy())
                pred_list = np.concatenate(pred_list, axis=0)
                pred_list = np.argmax(pred_list, axis=1)
                pred_list = list(pred_list)
                ans_list = []
                for pred_index in pred_list:
                    pred_token = ENTITY_LABEL_LIST[pred_index]
                    ans_list.append(pred_token)
                if len(true_word_index) != len(word_index_list):
                    continue
                res_label_list = [ans_list[idx] for idx, value in enumerate(true_word_index) if value != self.pad_token_label_id]
                temp_word_list, temp_label_list = [], []
                for word, label in zip([str(word).strip() for word in nlp(sent_list[sent_index])], res_label_list):
                    temp_word_list.append(word)
                    temp_label_list.append(label)
                result_word_list.append(temp_word_list)
                result_label_list.append(temp_label_list)
        return result_word_list, result_label_list

    # 将一段文本进行分为一个个句子
    def get_cti_sent_list(self, cti_content: str):
        doc = nlp(cti_content)
        cti_sent_list = [sent.text for sent in doc.sents]
        return cti_sent_list

    # 处理文章
    def process_text(self, word_list, label_list, cti_id):
        # 初始化文章文本和实体信息列表
        entities = []
        current_offset = 0
        full_text = ""
        # 遍历每个句子及其对应的标签列表
        for sentence, tag in zip(word_list, label_list):
            # 初始化句子文本
            sentence_text = ' '.join(sentence)
            # 更新全文
            full_text += sentence_text + ' '
            # 处理实体标签
            start = None
            end = None
            label = None
            entity_text = ""  # 用于保存实体的文本内容

            for i, (word, tag) in enumerate(zip(sentence, tag)):
                if tag.startswith('B-'):
                    if label is not None and start is not None and end is not None:
                        # 如果已有一个实体在构建，则先保存这个实体
                        entities.append({"startOffset": start, "endOffset": end, "itemId": get_item_type_id(label),
                                         "sentText": entity_text.strip()})
                    # 开始一个新实体
                    start = current_offset
                    end = current_offset + len(word)
                    label = tag[2:]
                    entity_text = word  # 开始记录新实体的文本

                elif tag.startswith('I-') and label is not None:
                    # 继续构建当前实体
                    end = current_offset + len(word)
                    entity_text += " " + word  # 继续添加词到实体文本

                else:
                    # 非实体标签或实体结束
                    if label is not None and start is not None and end is not None:
                        # 保存当前构建的实体
                        entities.append({"startOffset": start, "endOffset": end, "itemId": get_item_type_id(label),
                                         "sentText": entity_text.strip()})
                        start = None
                        end = None
                        label = None
                        entity_text = ""

                # 更新当前文本偏移量
                current_offset += len(word) + 1  # 加1为单词后的空格

            # 检查是否有未结束的实体
            if label is not None and start is not None and end is not None:
                entities.append(
                    {"startOffset": start, "endOffset": end, "itemId": get_item_type_id(label),
                     "sentText": entity_text.strip()})

        update_cti_content_by_cti_id(cti_id, full_text.strip())

        ctiChunkData = []
        for i in entities:
            i["ctiId"] = cti_id
            ctiChunkData.append(i)

        add_cti_chunk(ctiChunkData)

    def create_graph(self, sent_list, label_list, cti_report_id):
        def parse_entities(sent, labels):
            entities = []
            current_entity = []
            current_label = None

            for word, label in zip(sent, labels):
                if label.startswith("B-"):
                    if current_entity:
                        entities.append((" ".join(current_entity), current_label))
                    current_entity = [word]
                    current_label = label[2:]
                elif label.startswith("I-") and current_label == label[2:]:
                    current_entity.append(word)
                else:
                    if current_entity:
                        entities.append((" ".join(current_entity), current_label))
                        current_entity = []
                        current_label = None
            if current_entity:
                entities.append((" ".join(current_entity), current_label))
            return entities

        entities_per_sentence = [parse_entities(sent, labels) for sent, labels in zip(sent_list, label_list)]
        # 这里需要先把所有的entity都存放一下
        for i in range(len(entities_per_sentence)):
            for entity in entities_per_sentence[i]:
                # 往entity表中去添加数据
                add_entity_data(entity[0], cti_report_id, get_item_type_id(entity[1]))

        graph = []
        for i in range(len(entities_per_sentence) - 1):
            for entity1 in entities_per_sentence[i]:
                for entity2 in entities_per_sentence[i + 1]:
                    key1 = f"{entity1[1]}|---|{entity2[1]}"
                    key2 = f"{entity2[1]}|---|{entity1[1]}"
                    if key1 in GRAPH_RELATION_CONFIG:
                        relation_type_id = get_relation_type_id(get_father_item_id(entity1[1]),
                                                                get_father_item_id(entity2[1]),
                                                                GRAPH_RELATION_CONFIG[key1][0])
                        start_detail_cti_chunk_id = get_detail_entity_id(cti_id=cti_report_id,
                                                                         sent_text=entity1[0],
                                                                         item_id=get_item_type_id(
                                                                             entity1[1]))  # 头实体和尾实体是需要正确的id
                        end_detail_cti_chunk_id = get_detail_entity_id(cti_id=cti_report_id,
                                                                       sent_text=entity2[0],
                                                                       item_id=get_item_type_id(
                                                                           entity2[1]))  # 头实体和尾实体是需要正确的id
                        add_final_relation_data(cti_report_id,
                                                      start_detail_cti_chunk_id,
                                                      end_detail_cti_chunk_id,
                                                      relation_type_id)

                    if key2 in GRAPH_RELATION_CONFIG:
                        relation_type_id = get_relation_type_id(
                            get_father_item_id(entity2[1]),
                            get_father_item_id(entity1[1]),
                            GRAPH_RELATION_CONFIG[key2][0])
                        start_detail_cti_chunk_id = get_detail_entity_id(cti_id=cti_report_id,
                                                                         sent_text=entity2[0],
                                                                         item_id=get_item_type_id(
                                                                             entity2[1]))  # 头实体和尾实体是需要正确的id
                        end_detail_cti_chunk_id = get_detail_entity_id(cti_id=cti_report_id,
                                                                       sent_text=entity1[0],
                                                                       item_id=get_item_type_id(
                                                                           entity1[1]))  # 头实体和尾实体是需要正确的id
                        add_final_relation_data(cti_report_id,
                                                      start_detail_cti_chunk_id,
                                                      end_detail_cti_chunk_id,
                                                      relation_type_id)

