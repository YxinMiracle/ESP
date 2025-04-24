import json
import logging
import os
import pathlib
from typing import List

import h5py
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel

from model.dualgate import DualGate
from process_model.knn.KnnResultVo import IdentifiedReSentBaseData
from process_model.model.ReBaseData import ReSentBaseData

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class KnnRetrievalTemplate:
    def __init__(self, source_data_list: List[ReSentBaseData], params):
        self.source_data_list = source_data_list  # type:List[ReSentBaseData]
        self.params = params
        self.tokenizer = AutoTokenizer.from_pretrained(self.params.bert_model_name)
        self.bert = AutoModel.from_pretrained(self.params.bert_model_name)
        self.sent_str_list = []
        self.final_embedding_list = []
        self.root_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent)
        self.model_path = f"{self.root_dir}/save/{params.data_directory_name}/model.pt"
        self.model = self._init_model()
        self.model.eval()
        self.knn_model_data = self.root_dir + os.path.sep + self.params.data_directory_name + os.path.sep + self.params.final_embedding_file_name
        self.knn_num = self.params.knn_num
        self.knn_model, self.stored_sentences = self._init_knn_model()

    def _init_knn_model(self):
        if not os.path.exists(
                self.knn_model_data):
            return None, None
        logger.info("正在加载knn所需数据....")
        # 加载数据
        with h5py.File(
                self.knn_model_data,
                'r') as f:
            stored_sentences = f['sentences'][:]
            embeddings = f['embeddings'][:]
        # 创建KNN模型
        knn = NearestNeighbors(n_neighbors=self.knn_num, algorithm='ball_tree')
        knn.fit(embeddings)
        return knn, stored_sentences

    def _init_model(self):
        ner2idx_file_path = self.root_dir + os.path.sep + self.params.data_directory_name + os.path.sep + self.params.ner2idx_file_name
        rel2idx_file_path = self.root_dir + os.path.sep + self.params.data_directory_name + os.path.sep + self.params.re2idx_file_name
        with open(ner2idx_file_path, "r") as f:
            ner2idx = json.load(f)  # type: dict
        with open(rel2idx_file_path, "r") as f:
            rel2idx = json.load(f)  # type: dict
        model = DualGate(self.params, ner2idx, rel2idx)
        state_dict = torch.load(self.model_path)
        state_dict.pop('bert.embeddings.position_ids', None)  # 移除不匹配的
        model.load_state_dict(state_dict, strict=False)
        model.cuda()
        return model

    def get_reconstructed_sent_embedding(self, reconstructed_sent_str):
        sent_token = self.tokenizer.tokenize(reconstructed_sent_str)
        x = self.tokenizer(sent_token, return_tensors="pt",
                           padding='longest',
                           max_length=512,
                           truncation=True,
                           is_split_into_words=True)
        x = self.bert(**x)[0]
        reconstructed_sent_embedding = x.transpose(0, 1)
        return reconstructed_sent_embedding

    def get_re_embedding(self, sent_token):
        sent_str = ' '.join(sent_token)
        bert_words = self.tokenizer.tokenize(sent_str)
        token_len = len(bert_words) + 2
        mask = torch.LongTensor(token_len, 1).fill_(0).cuda()
        mask[:token_len, 0] = 1
        _, _, re_embedding = self.model(sent_token, mask)
        return re_embedding

    def _get_final_sent_embedding(self, re_embedding, reconstructed_sent_embedding):
        # 获取平均值
        rel_emb_mean = re_embedding.squeeze(1).mean(dim=0)
        reconstructed_emb_mean = reconstructed_sent_embedding.squeeze(1).mean(dim=0)

        # 获取最大值
        rel_emb_max = re_embedding.squeeze(1).max(dim=0)[0]
        reconstructed_emb_max = reconstructed_sent_embedding.squeeze(1).max(dim=0)[0]

        # 进行拼接
        rel_emb_combined = torch.cat([rel_emb_mean, rel_emb_max]).cpu()
        bert_emb_combined = torch.cat([reconstructed_emb_mean, reconstructed_emb_max]).cpu()
        final_embedding = torch.cat([rel_emb_combined, bert_emb_combined])
        return final_embedding

    def save_sent_embedding_list(self):
        if os.path.exists(self.knn_model_data): return
        logger.info("项目中不存在knn所需数据，现在开始获取knn所需数据")
        for single_task in self.source_data_list:
            sent_token = single_task.fine_tuned_re_model_tokens
            reconstructed_sent = single_task.reconstructed_sent
            re_embedding = self.get_re_embedding(sent_token)
            reconstructed_sent_embedding = self.get_reconstructed_sent_embedding(reconstructed_sent)
            final_embedding = self._get_final_sent_embedding(re_embedding, reconstructed_sent_embedding)
            self.final_embedding_list.append(final_embedding.detach().numpy())
            self.sent_str_list.append(json.dumps(single_task.to_dict()))

        # 存储数据
        with h5py.File(
                self.knn_model_data,
                'w') as f:
            f.create_dataset('sentences', data=np.array(self.sent_str_list, dtype='S'))  # 存储句子
            f.create_dataset('embeddings', data=np.stack(self.final_embedding_list))  # 存储嵌入
        logger.info("knn所需数据获取成功")
        self.knn_model, self.stored_sentences = self._init_knn_model()

    def find_sent(self, sent_data: ReSentBaseData) -> List[IdentifiedReSentBaseData]:
        reconstructed_sent = sent_data.reconstructed_sent
        logger.info("重构句子已获取成功，句子为：{}".format(reconstructed_sent))
        re_sent = sent_data.fine_tuned_re_model_tokens
        logger.info("捕获关系句子已获取成功，句子为：{}".format(re_sent))
        logger.info("正在获取，并拼接对应的嵌入")
        reconstructed_sent_embedding = self.get_reconstructed_sent_embedding(reconstructed_sent)
        re_embedding = self.get_re_embedding(re_sent)
        final_embedding = self._get_final_sent_embedding(re_embedding, reconstructed_sent_embedding)

        distances, indices = self.knn_model.kneighbors(final_embedding.detach().numpy().reshape(1, -1))

        ret_list = []  # type: List[IdentifiedReSentBaseData]
        for indices_index, index in enumerate(indices[0]):
            if distances[0][indices_index] == 0:
                continue
            data = json.loads(self.stored_sentences[index].decode('utf-8'))
            restored_sentence_data = ReSentBaseData.from_dict(data)
            irsb_obj = IdentifiedReSentBaseData(restored_sentence_data.sent,
                                                restored_sentence_data.head_entity,
                                                restored_sentence_data.tail_entity,
                                                restored_sentence_data.relation_type,
                                                restored_sentence_data.sent_token_list,
                                                id=index)
            ret_list.append(irsb_obj)

        return ret_list

    def do_find_similar_sent(self, sent_data: ReSentBaseData) -> List[IdentifiedReSentBaseData]:
        if not os.path.exists(self.knn_model_data):
            self.save_sent_embedding_list()
        return self.find_sent(sent_data)  # type: List[IdentifiedReSentBaseData]
