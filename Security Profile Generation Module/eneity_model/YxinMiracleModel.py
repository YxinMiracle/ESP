import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import AutoModelWithLMHead
import numpy as np

import logging

from eneity_model.entity_config import BERT_MODEL_NAME

logger = logging.getLogger()


class AttAttYxinNerModel_POS(nn.Module):
    def __init__(self):
        super(AttAttYxinNerModel_POS, self).__init__()
        self.num_tag = 79
        self.hidden_dim = 768
        self.target_embedding_dim = 50  # 这个是给label进行编码的
        config = AutoConfig.from_pretrained(BERT_MODEL_NAME)
        config.output_hidden_states = True

        self.model = AutoModelWithLMHead.from_pretrained(BERT_MODEL_NAME, config=config)
        self.target_sequence = True
        self.target_type = "LSTM"
        self.connect_label_background = True
        self.window_size = 7
        self.temp_dim = 384
        self.temp_dim2 = 192
        self.pos_tag_num = 18
        self.pos_embedding_dim = 100

        self.subsqe_linear = nn.Linear(self.window_size * self.hidden_dim, self.hidden_dim)
        # self.NTN = TenorNetworkModule(self.temp_dim2, self.temp_dim2)

        self.output_linear = nn.Linear(self.hidden_dim, self.temp_dim)
        self.output_linear2 = nn.Linear(self.temp_dim, self.temp_dim2)
        self.subseq_output_linear2 = nn.Linear(self.window_size * self.hidden_dim, self.hidden_dim)
        self.hid2wordhiddendim_linear = nn.Linear(self.temp_dim2, self.hidden_dim)
        # self.mutiattention = MultiHeadSelfAttention(self.hidden_dim, 8)

        # 为了控制维度设定一个全链接层
        self.word_dim_linear = nn.Linear(768, self.hidden_dim)

        if self.target_sequence:
            # 这是给label转换为对应的词向量的
            self.target_embedding = nn.Embedding(self.num_tag + 2, self.target_embedding_dim, padding_idx=0)
            self.pos_embedding_layer = nn.Embedding(self.pos_tag_num + 2, self.pos_embedding_dim, padding_idx=0)
            self.POS_LSTM_encoder = nn.LSTM(self.pos_embedding_dim, self.pos_embedding_dim, batch_first=True, bidirectional=True)
            # 将向量维度转为label标签的维度，target_embedding_dim=50
            self.to_target_emb = nn.Linear(self.hidden_dim + 2 * self.target_embedding_dim, self.target_embedding_dim)
            if self.target_type == "LSTM":
                # 用作标签的特征提取
                self.LSTM_encoder = nn.LSTM(int(self.target_embedding_dim), int(self.target_embedding_dim),
                                            batch_first=True, bidirectional=True)
                # 用作获取LSTM最后一个时间步骤的特征，输入的是LSTM_encoder层的输出，输出的维度是hidden_dim，也就是单词的维度，所以这是给单词词向量序列进行注意力权重计算的
                self.LSTM_out_linear = nn.Linear(self.target_embedding_dim * 2, self.hidden_dim)

                self.Bert_to_target2 = nn.Linear(self.hidden_dim * 2, 2 * self.target_embedding_dim)
                self.Bert_to_target = nn.Linear(self.hidden_dim, 2 * self.target_embedding_dim)
            self.se_linear = nn.Linear(self.hidden_dim * 3 + self.target_embedding_dim * 2 + self.pos_embedding_dim * 2,
                                       self.num_tag)
            self.se_linear_first = nn.Linear(
                self.hidden_dim * 3 + self.target_embedding_dim * 2 + self.pos_embedding_dim * 2, self.num_tag)

        else:
            self.linear = nn.Linear(self.hidden_dim, self.num_tag)

    def change_shape(self, word_embedding):
        word_embedding = self.output_linear(word_embedding)
        word_embedding = self.output_linear2(word_embedding)
        return word_embedding

    def dot_attention(self, current_word_embedding, total_y_embedding):
        """
                                               label
        current_word_embedding shape is [bsz, 1, hidden_dim] / [bsz, 1, 2*target_embedding_dim]
                                               word
        total_y_embedding shape is [bs seq_len, hidden_dim] / [bsz, seq_len, 2*target_embedding_dim]
        """
        # 左边的attention计算方式：[bsz, 1, hidden_dim] * [bs, hidden_dim, seq_len] -> [bs, 1, seq_len]
        # 右边的attention计算方式：[bsz, 1, 2*target_embedding_dim] * [bsz, 2*target_embedding_dim, seq_len] = [bsz, 1, seq_len]
        attention_weight = current_word_embedding @ total_y_embedding.permute(0, 2, 1)
        attention_weight = torch.softmax(attention_weight, dim=-1)  # 注意力权重的计算
        # 左边的attention_weight [bs, 1, seq_len]* [bs seq_len, hidden_dim] = [bs,1,hidden_dim]
        # 右边的attention_weight [bs,1,seq_len] * [bsz, seq_len, 2*target_embedding_dim] = [bs, 1, 2*target_embedding_dim]
        relation_information = attention_weight @ total_y_embedding  # 计算对应的结果
        return relation_information

    def get_subseq_idx_list(self, windows, t, T):
        index_list = []
        for u in range(1, windows // 2 + 1):  # 这里是构建子串
            if t - u >= 0:
                index_list.append(t - u)
            if t + u <= T - 1:
                index_list.append(t + u)
        index_list.append(t)
        index_list.sort()
        return index_list

    # 自注意力机制
    def self_attention(self, current_wordseq_feat, total_word_embedding):
        # current_wordseq_feat (bsz ,1, hidden_dim)
        # total_word_embedding (bsz, seq_len, hidden_dim)
        attention_weight = current_wordseq_feat @ total_word_embedding.permute(0, 2, 1)  # -> [bs, 1, seq_len] 是权重
        attention_weight = torch.softmax(attention_weight, dim=-1)  # 注意力权重的计算 # -> [bs, 1, seq_len] 是权重
        relation_information = attention_weight @ total_word_embedding  # 计算对应的结果 [bs, 1, seq_len] * (bsz, seq_len, hidden_dim) = [bsz, 1, hidden_dim]
        return relation_information  # (bsz ,1, hidden_dim)

    def forward(self, X, y, pos_data):
        # 将词，经过bert，转换为对应的词向量
        # torch.set_printoptions(threshold=10000)
        # print(X)
        # print("======================================")
        outputs = self.model(X)  # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        outputs = outputs[1][-1]  # (bsz, seq_len, hidden_dim)
        outputs = self.word_dim_linear(outputs)
        hcl_loss = 0
        pos_modified = torch.where(pos_data < -1, self.pos_tag_num + 1, pos_data).to(
            outputs.device)  # shape is [bsz,seq_len]
        pos_embedding = self.pos_embedding_layer(pos_modified)
        lstm_pos_embedding, (_, _) = self.POS_LSTM_encoder(pos_embedding)
        if self.target_sequence:
            # y_modified shape is [bsz,seq_len]
            # y_modified为 要是y中的值<-1，那么就变成num_tag+1，否则的话还是原来本身的值
            # y中小于0的值都表示的是无意义的标签
            # 之所以这么做是因为要给label标签使用target_embedding进行转为词向量
            y_modified = torch.where(y < -1, self.num_tag + 1, y).to(outputs.device)
            y_embedding = self.target_embedding(y_modified)
            bsz, seq_len, dim = outputs.shape  # 这里面指的是和单词有关的，dim指的是单词词向量的维度，而不是label
            predcits = []
            # init_zero shape is [bsz, 2*self.target_embedding_dim] = [32, 100]
            init_zero = torch.zeros([bsz, 2 * self.target_embedding_dim], dtype=torch.float32, device="cuda")
            for i in range(seq_len):

                index_list = self.get_subseq_idx_list(self.window_size, i, seq_len)
                # 目前这个单词的子序列特征
                subseq_feat = outputs[:, index_list, :]  # [bs,len(index_list),hidden_size]
                size = subseq_feat.size()
                if len(index_list) < self.window_size:
                    # 在第一个维度上进行拼接
                    # 最终subseq_feat的形状都会进行统一，subseq_feat=[bs,windows,hidden_size]
                    subseq_feat = torch.cat(
                        [subseq_feat, torch.zeros((size[0], self.window_size - size[1], size[-1])).cuda()],
                        dim=1)  # [bs,len(index_list),hidden_size]
                # 目前这个单词的子序列特征 (bsz, windows * hidden_dim) 下一步我需要将他改变为 (bsz , hidden_dim) 因为自注意力机制的需要，所以进行形状变化
                subword_feat = torch.reshape(subseq_feat, (size[0], self.window_size * size[2]))
                new_subword_feat = self.subsqe_linear(subword_feat).unsqueeze(dim=1)  # shape (bsz ,1, hidden_dim)
                now_word_feat = self.self_attention(new_subword_feat, outputs).squeeze()
                # now_word_feat = self.mutiattention(new_subword_feat, outputs).squeeze()
                # new_outputs = self.change_shape(outputs[:, i, :])
                # now_word_feat = self.change_shape(now_word_feat)
                # now_subword_feat = self.change_shape(new_subword_feat.squeeze(dim=1))
                # now_word_feat = self.NTN(new_outputs, now_subword_feat, now_word_feat)
                # now_word_feat = self.hid2wordhiddendim_linear(now_word_feat)

                pos_feat = lstm_pos_embedding[:, i, :]  # [bsz, pos_embedding_dim]

                # 循环每一个时间步
                if i == 0:  # 对于第一个时间步骤需要特殊处理
                    # outputs[:, i, :] shape is [bs, hidden_dim] = [32, 768]
                    # init_zero shape is [bsz, 2*self.target_embedding_dim] = [32, 100]
                    # current_word_re shape is [bs,hidden_dim + 2*self.target_embedding_dim] = [32, 768+2*50]
                    current_word_re = torch.cat([now_word_feat, init_zero], dim=1)
                    current_label_embedding = self.to_target_emb(current_word_re)  # 这里的label_embedding还包含了单词的词向量
                    current_word_re = torch.cat([outputs[:, i, :],  # [bsz, hidden_dim] = [32, 768]
                                                 outputs[:, i, :],
                                                 now_word_feat,  # [bsz, hidden_dim] = [32, 768]
                                                 pos_feat,  # [bsz, pos_embedding_dim]
                                                 current_label_embedding.squeeze(),  # [32, 50]
                                                 current_label_embedding.squeeze()],  # [32, 50]
                                                dim=-1)  # [32, 1636]
                    predict = self.se_linear_first(current_word_re)  # 第一次的label识别的概率
                else:
                    # total_y_embedding shape is [bsz, i, target_embedding_dim]
                    total_y_embedding = y_embedding[:, :i, :]  # 获取这个时间步的正确的y(label)对应的embedding,
                    # 把这个时间步骤之前的label embedding数据拿出来
                    # LSTM_encoder input shape is (bsz, seq_len, target_embedding_dim)
                    #             output shape is (bsz, seq_len, 2*target_embedding_dim)
                    output_lstm, (_, _) = self.LSTM_encoder(total_y_embedding)
                    relation_information = output_lstm[:, -1, :]  # 选出了这个句子最后一个单词的特征，论文中将它认定为Q
                    label_memory = self.LSTM_out_linear(
                        relation_information)  # 把label的维度转换为单词的维度，也就是(target_embedding_dim->hidden_size)
                    # 这里是改变了句子中的单词的词向量特征重要程度，也就是使用relation_information，进行注意力机制计算，输出在原句子中有哪些单词对应于relation_information是重要的
                    # 这个输出是h(i,b)
                    label_background = self.dot_attention(label_memory.unsqueeze(dim=1), outputs).squeeze()

                    if self.connect_label_background:
                        # [outputs[:, i, :] shape is [bsz,hidden_dim]
                        # label_background [bsz, hidden_dim]
                        # torch.cat([outputs[:, i, :], label_background] = [bsz, hidden_dim*2]
                        # output_memory shape is [bsz, 2*target_embedding_dim]
                        # 将单词hi 与 经过注意力计算的h(i,b)进行和并，是作为右边BiAttention一部分输入
                        output_memory = self.Bert_to_target2(torch.cat([now_word_feat, label_background], dim=-1))
                    else:
                        # BERTtoTarget = nn.Linear(self.hidden_dim, 2 * self.target_embedding_dim)
                        # [bs,hidden_dim]->[bs,2 * self.target_embedding_dim]
                        output_memory = self.BERTtoTarget(now_word_feat)
                    # 这里计算的就是BiAttention的右边
                    # output_memory.unsqueeze(dim=1) shape is [bsz, 1 , 2*target_embedding_dim]
                    # output_lstm is [bsz, seq_len, 2*target_embedding_dim]
                    # label_context shape is [bs, 2*target_embedding_dim]
                    # 计算目前时间步骤之前的label_embedding那些东西对于目前这个打次来说是重要的
                    label_context = self.dot_attention(output_memory.unsqueeze(dim=1), output_lstm).squeeze()
                    # 论文中提到的将e(i,c) 与 h(i,b) 与 hi 进行合并起来，这两个值都是注意力机制出来的结果
                    total_word_re = torch.cat(
                        [outputs[:, i, :], now_word_feat, pos_feat, label_background, label_context],
                        dim=-1)
                    predict = self.se_linear(total_word_re)
                predcits.append(predict)
            prediction2 = torch.stack(predcits, dim=1)
        else:
            prediction2 = self.linear(outputs)
        return prediction2, hcl_loss

    def test(self, X, pos_data):

        outputs = self.model(X)  # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        outputs = outputs[1][-1]  # (bsz, seq_len, hidden_dim)
        outputs = self.word_dim_linear(outputs)
        pos_modified = torch.where(pos_data < -1, self.pos_tag_num + 1, pos_data).to(
            outputs.device)  # shape is [bsz,seq_len]
        pos_embedding = self.pos_embedding_layer(pos_modified)
        lstm_pos_embedding, (_, _) = self.POS_LSTM_encoder(pos_embedding)
        if self.target_sequence == True:
            bsz, seq_len, dim = outputs.shape
            predcits = []
            init_zero = torch.zeros([bsz, 2 * self.target_embedding_dim], dtype=torch.float32, device='cuda')
            total_predict = None

            for i in range(seq_len):

                index_list = self.get_subseq_idx_list(self.window_size, i, seq_len)
                # 目前这个单词的子序列特征
                subseq_feat = outputs[:, index_list, :]  # [bs,len(index_list),hidden_size]
                size = subseq_feat.size()
                if len(index_list) < self.window_size:
                    # 在第一个维度上进行拼接
                    # 最终subseq_feat的形状都会进行统一，subseq_feat=[bs,windows,hidden_size]
                    subseq_feat = torch.cat(
                        [subseq_feat, torch.zeros((size[0], self.window_size - size[1], size[-1])).cuda()],
                        dim=1)  # [bs,len(index_list),hidden_size]
                # 目前这个单词的子序列特征 (bsz, windows * hidden_dim) 下一步我需要将他改变为 (bsz , hidden_dim) 因为自注意力机制的需要，所以进行形状变化
                subword_feat = torch.reshape(subseq_feat, (size[0], self.window_size * size[2]))
                new_subword_feat = self.subsqe_linear(subword_feat).unsqueeze(dim=1)  # shape (bsz ,1, hidden_dim)
                now_word_feat = self.self_attention(new_subword_feat, outputs).squeeze(1)
                # now_word_feat = self.mutiattention(new_subword_feat, outputs).squeeze()
                # new_outputs = self.change_shape(outputs[:, i, :])
                # now_word_feat = self.change_shape(now_word_feat)
                # now_subword_feat = self.change_shape(new_subword_feat.squeeze(dim=1))
                # now_word_feat = self.NTN(new_outputs, now_subword_feat, now_word_feat)
                # now_word_feat = self.hid2wordhiddendim_linear(now_word_feat)
                pos_feat = lstm_pos_embedding[:, i, :]  # [bsz, pos_embedding_dim]
                if i == 0:
                    current_word_re = torch.cat([now_word_feat, init_zero], dim=1)
                    current_label_embedding = self.to_target_emb(current_word_re)
                    if len(now_word_feat.shape) == len(current_label_embedding.shape):
                        current_word_re = torch.cat(
                            [outputs[:, i, :], outputs[:, i, :], pos_feat, now_word_feat, current_label_embedding,
                             current_label_embedding], dim=-1)
                    else:
                        current_word_re = torch.cat(
                            [outputs[:, i, :], outputs[:, i, :], pos_feat, now_word_feat,
                             current_label_embedding.squeeze(),
                             current_label_embedding.squeeze()], dim=-1)

                    predict = self.se_linear_first(current_word_re)

                else:
                    total_y_embedding = self.target_embedding(total_predict)

                    output_lstm, (hn, cn) = self.LSTM_encoder(total_y_embedding)
                    relation_information = output_lstm[:, -1, :]
                    label_memory = self.LSTM_out_linear(relation_information)
                    label_background = self.dot_attention(label_memory.unsqueeze(dim=1), outputs).squeeze()
                    if len(label_background.shape) == 1:
                        label_background = label_background.unsqueeze(dim=0)

                    if self.connect_label_background:
                        output_memory = self.Bert_to_target2(torch.cat([now_word_feat, label_background], dim=-1))
                    else:
                        output_memory = self.Bert_to_target(now_word_feat)

                    label_context = self.dot_attention(output_memory.unsqueeze(dim=1), output_lstm).squeeze()
                    if len(label_context.shape) == 1:
                        label_context = label_context.unsqueeze(dim=0)

                    if len(now_word_feat.shape) == len(label_background.shape):
                        total_word_re = torch.cat(
                            [outputs[:, i, :], now_word_feat, pos_feat, label_background, label_context],
                            dim=-1)
                    else:
                        total_word_re = torch.cat(
                            [outputs[:, i, :], now_word_feat, pos_feat, label_background.unsqueeze(dim=0),
                             label_context.unsqueeze(dim=0)],
                            dim=-1)

                    predict = self.se_linear(total_word_re)

                current_predict = predict.data.cpu().numpy()
                current_predict = np.argmax(current_predict, axis=1)
                current_predict2 = torch.tensor(current_predict, dtype=torch.long, device='cuda')

                if total_predict == None:
                    total_predict = current_predict2.unsqueeze(dim=1)
                else:
                    total_predict = torch.cat([total_predict, current_predict2.unsqueeze(dim=1)], dim=-1)

                predcits.append(predict)

            prediction2 = torch.stack(predcits, dim=1)
        else:
            prediction2 = self.linear(outputs)
        return prediction2
