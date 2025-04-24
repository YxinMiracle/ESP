import logging

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import AutoModelWithLMHead

logger = logging.getLogger()


class ESP(nn.Module):
    def __init__(self, params):
        super(ESP, self).__init__()
        self.num_tag = params.num_tag
        self.hidden_dim = params.hidden_dim
        self.target_embedding_dim = params.target_embedding_dim
        config = AutoConfig.from_pretrained(params.model_name)
        config.output_hidden_states = True

        self.model = AutoModelWithLMHead.from_pretrained(params.model_name, config=config)
        self.target_sequence = params.target_sequence
        self.target_type = params.target_type  # LSTM
        self.connect_label_background = params.connect_label_background
        self.window_size = params.windows
        self.pos_tag_num = params.pos_tag_num
        self.pos_embedding_dim = params.pos_embedding_dim

        self.subsqe_linear = nn.Linear(self.window_size * self.hidden_dim, self.hidden_dim)
        self.word_dim_linear = nn.Linear(768, self.hidden_dim)

        if self.target_sequence:
            self.target_embedding = nn.Embedding(self.num_tag + 2, self.target_embedding_dim, padding_idx=0)
            self.pos_embedding_layer = nn.Embedding(self.pos_tag_num + 2, self.pos_embedding_dim, padding_idx=0)
            self.POS_LSTM_encoder = nn.LSTM(self.pos_embedding_dim, self.pos_embedding_dim, batch_first=True,
                                            bidirectional=True)
            self.to_target_emb = nn.Linear(self.hidden_dim + 2 * self.target_embedding_dim, self.target_embedding_dim)
            if self.target_type == "LSTM":
                self.LSTM_encoder = nn.LSTM(int(self.target_embedding_dim), int(self.target_embedding_dim),
                                            batch_first=True, bidirectional=True)
                self.LSTM_out_linear = nn.Linear(self.target_embedding_dim * 2, self.hidden_dim)

                self.Bert_to_target2 = nn.Linear(self.hidden_dim * 2, 2 * self.target_embedding_dim)
                self.Bert_to_target = nn.Linear(self.hidden_dim, 2 * self.target_embedding_dim)
            self.se_linear = nn.Linear(self.hidden_dim * 3 + self.target_embedding_dim * 2 + self.pos_embedding_dim * 2,
                                       self.num_tag)
            self.se_linear_first = nn.Linear(
                self.hidden_dim * 3 + self.target_embedding_dim * 2 + self.pos_embedding_dim * 2, self.num_tag)

        else:
            self.linear = nn.Linear(self.hidden_dim, self.num_tag)

    def dot_attention(self, current_word_embedding, total_y_embedding):
        attention_weight = current_word_embedding @ total_y_embedding.permute(0, 2, 1)
        attention_weight = torch.softmax(attention_weight, dim=-1)
        relation_information = attention_weight @ total_y_embedding
        return relation_information

    def get_subseq_idx_list(self, windows, t, T):
        index_list = []
        for u in range(1, windows // 2 + 1):
            if t - u >= 0:
                index_list.append(t - u)
            if t + u <= T - 1:
                index_list.append(t + u)
        index_list.append(t)
        index_list.sort()
        return index_list

    def self_attention(self, current_wordseq_feat, total_word_embedding):
        attention_weight = current_wordseq_feat @ total_word_embedding.permute(0, 2, 1)
        attention_weight = torch.softmax(attention_weight, dim=-1)
        relation_information = attention_weight @ total_word_embedding
        return relation_information

    def forward(self, X, y, pos_data):
        outputs = self.model(X)
        outputs = outputs[1][-1]
        outputs = self.word_dim_linear(outputs)
        hcl_loss = 0
        pos_modified = torch.where(pos_data < -1, self.pos_tag_num + 1, pos_data).to(
            outputs.device)
        pos_embedding = self.pos_embedding_layer(pos_modified)
        lstm_pos_embedding, (_, _) = self.POS_LSTM_encoder(pos_embedding)
        if self.target_sequence:
            y_modified = torch.where(y < -1, self.num_tag + 1, y).to(outputs.device)
            y_embedding = self.target_embedding(y_modified)
            bsz, seq_len, dim = outputs.shape
            predcits = []
            init_zero = torch.zeros([bsz, 2 * self.target_embedding_dim], dtype=torch.float32, device="cuda")
            for i in range(seq_len):

                index_list = self.get_subseq_idx_list(self.window_size, i, seq_len)
                subseq_feat = outputs[:, index_list, :]
                size = subseq_feat.size()
                if len(index_list) < self.window_size:
                    subseq_feat = torch.cat(
                        [subseq_feat, torch.zeros((size[0], self.window_size - size[1], size[-1])).cuda()],
                        dim=1)
                subword_feat = torch.reshape(subseq_feat, (size[0], self.window_size * size[2]))
                new_subword_feat = self.subsqe_linear(subword_feat).unsqueeze(dim=1)  # shape (bsz ,1, hidden_dim)
                now_word_feat = self.self_attention(new_subword_feat, outputs).squeeze()

                pos_feat = lstm_pos_embedding[:, i, :]

                if i == 0:
                    current_word_re = torch.cat([now_word_feat, init_zero], dim=1)
                    current_label_embedding = self.to_target_emb(current_word_re)
                    current_word_re = torch.cat([outputs[:, i, :],
                                                 outputs[:, i, :],
                                                 now_word_feat,
                                                 pos_feat,
                                                 current_label_embedding.squeeze(),
                                                 current_label_embedding.squeeze()],
                                                dim=-1)
                    predict = self.se_linear_first(current_word_re)
                else:
                    total_y_embedding = y_embedding[:, :i, :]
                    output_lstm, (_, _) = self.LSTM_encoder(total_y_embedding)
                    relation_information = output_lstm[:, -1, :]
                    label_memory = self.LSTM_out_linear(
                        relation_information)
                    label_background = self.dot_attention(label_memory.unsqueeze(dim=1), outputs).squeeze()

                    if self.connect_label_background:
                        output_memory = self.Bert_to_target2(torch.cat([now_word_feat, label_background], dim=-1))
                    else:
                        output_memory = self.BERTtoTarget(now_word_feat)
                    label_context = self.dot_attention(output_memory.unsqueeze(dim=1), output_lstm).squeeze()
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
        outputs = self.model(X)
        outputs = outputs[1][-1]
        outputs = self.word_dim_linear(outputs)
        pos_modified = torch.where(pos_data < -1, self.pos_tag_num + 1, pos_data).to(
            outputs.device)
        pos_embedding = self.pos_embedding_layer(pos_modified)
        lstm_pos_embedding, (_, _) = self.POS_LSTM_encoder(pos_embedding)
        if self.target_sequence == True:
            bsz, seq_len, dim = outputs.shape
            predcits = []
            init_zero = torch.zeros([bsz, 2 * self.target_embedding_dim], dtype=torch.float32, device='cuda')
            total_predict = None

            for i in range(seq_len):
                index_list = self.get_subseq_idx_list(self.window_size, i, seq_len)
                subseq_feat = outputs[:, index_list, :]
                size = subseq_feat.size()
                if len(index_list) < self.window_size:
                    subseq_feat = torch.cat(
                        [subseq_feat, torch.zeros((size[0], self.window_size - size[1], size[-1])).cuda()],
                        dim=1)
                subword_feat = torch.reshape(subseq_feat, (size[0], self.window_size * size[2]))
                new_subword_feat = self.subsqe_linear(subword_feat).unsqueeze(dim=1)
                now_word_feat = self.self_attention(new_subword_feat, outputs).squeeze()
                pos_feat = lstm_pos_embedding[:, i, :]
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
