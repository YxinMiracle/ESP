import math
from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class MRU(nn.Module):
    def __init__(self, hidden_dim, window_size, out_dim=0):
        super(MRU, self).__init__()
        self.windows = window_size
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim  # 768

        # 论文中所提到的多层感知机
        self.MLP_linear = nn.Linear((self.windows) * self.hidden_dim, self.hidden_dim)
        self.gate = nn.Linear((self.windows + 1) * hidden_dim, self.hidden_dim)

    def forward(self, outputs):
        windows = self.windows
        batch_size, T, feat_dim = outputs.shape
        prev_state = torch.zeros([batch_size, feat_dim], dtype=torch.float32, device='cuda')  # [batch_size, feat_dim] 针对一个单词的
        final_output = torch.zeros([batch_size, T, feat_dim], dtype=torch.float32, device='cuda')
        for t in range(T):
            index_list = self.get_subseq_idx_list(windows, t, T)
            # 这里的index_list不一定有k个，因为要考虑到开头结尾的情况
            subseq_feat = outputs[:, index_list, :]  # [bs,len(index_list),hidden_size]
            size = subseq_feat.size()
            # 要是没有K个的话，就要进行调整
            if len(index_list) < windows:
                # 在第一个维度上进行拼接
                # 最终subseq_feat的形状都会进行统一，subseq_feat=[bs,windows,hidden_size]
                subseq_feat = torch.cat([subseq_feat, torch.zeros((size[0], windows - size[1], size[-1])).cuda()], dim=1)

            d_k = feat_dim
            prev_state = prev_state.unsqueeze(dim=1)  # [batch_size, feat_dim] -> [batch_size,1,feat_dim]
            # =======================================Attention这是注意力权重的计算=======================================
            # prev_state -> [batch_size,1,feat_dim]
            # subseq_feat.permute(0, 2, 1))=转秩 -> [bs,windows,hidden_size] ->[bs,hidden_size,windows]
            # [bs,1,feat_dim] * [bs,hidden_size,windows] = [bs,1,windows] = [16,1,7]
            probability = torch.softmax(prev_state @ (subseq_feat.permute(0, 2, 1)) / math.sqrt(d_k), dim=-1)

            # probability=[bs,1,windows],temp=[bs,windows,hidden_size] [bs,1,windows]*[bs,windows,hidden_size] = [bs,1,hidden_size]
            prev_state = probability @ subseq_feat  # 更新previous_state-> [bs,1,hidden_size]
            # =======================================Attention这是注意力计算完毕=======================================
            temp1 = torch.reshape(subseq_feat, (size[0], windows * size[2]))  # [bs,window*hidden_size]
            # subseq_feat=[bs, windows, hidden_size]
            temp2 = torch.cat([subseq_feat, prev_state], dim=1)  # [bs,(window+1),hidden_size]
            temp2 = torch.reshape(temp2, (size[0], (windows + 1) * size[2]))  # [bs,(window+1)*hidden_size]
            # 论文中对应的m向量就是这里的temp2
            subsequence_feat = self.MLP_linear(temp1)  # [bs,window*hidden_size] -> [bs,hidden_size] # 自己本身的一个特征提取的行为
            # self.gate = nn.Linear((self.windows + 1) * self.hidden_dim, self.hidden_dim)
            # temp2 = [bs,(window+1)*hidden_size]
            # [bs, hidden_size]
            gate = torch.sigmoid(self.gate(temp2))  # 对应论文中的r变量，一个0-1的值
            prev_state = prev_state.squeeze()  # [bs,1,hidden_size] -> [bs,hidden_size]
            out = subsequence_feat + prev_state * gate  # 类似于lstm遗忘门 [bs,hidden_size] + [bs,hidden_size]*gate
            prev_state = out
            final_output[:, t, :] = out
        return final_output

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


class BiMRU(nn.Module):
    def __init__(self, hidden_dim, window_size, out_dim=0):
        super(BiMRU, self).__init__()
        self.windows = window_size
        self.hidden_dim = hidden_dim
        if out_dim == 0:
            self.out_dim = self.hidden_dim

        # 前向后向模型
        self.forward_RW = MRU(self.hidden_dim, self.windows,
                              out_dim=self.out_dim)  # 前向MRU,传递的参数有hidden_dim windows out_dim(输出的维度)
        self.barward_RW = MRU(self.hidden_dim, self.windows,
                              out_dim=self.out_dim)

    def forward(self, outputs):
        # 这里输入的数据还没有接入子串
        # outputs [bsz, seq_len, hidden_dim]
        forward_out = self.forward_RW(outputs)  # [bs,max_len,hidden_size]
        batch_size, max_len, feat_dim = outputs.shape
        reverse_x = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float32, device='cuda')
        # 转向outputs
        for i in range(max_len):
            reverse_x[:, i, :] = outputs[:, max_len - 1 - i, :]
        backward_out = self.forward_RW(reverse_x)
        return forward_out, backward_out


class MutiLabelModel(nn.Module):
    def __init__(self, n_classes, model_type:str):
        super(MutiLabelModel, self).__init__()
        self.model_type = model_type
        self.num_tag = n_classes
        self.hidden_dim = 768
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.bert = AutoModel.from_pretrained("bert-base-cased")
        # 规定子序列大小
        self.windows = 7

        # 提取细粒度信息
        self.MRU = BiMRU(self.hidden_dim, self.windows)

        # 最后阶段的三个门
        self.gate = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.gate2 = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.gate3 = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

        # 最后一步全连接层 [hidden_dim* num_tag]
        self.linear = nn.Linear(self.hidden_dim, self.num_tag)

    def forward(self, x: List[str]):
        outputs = self.tokenizer(x, return_tensors="pt",
                                 padding='longest',
                                 truncation=True,
                                 max_length=512
                                 ).to(self.device)
        outputs = self.bert(**outputs)[0]
        forward_out, backward_out = self.MRU(outputs)  # [bs,hidden_size]，[bs,hidden_size]
        gate = torch.sigmoid(self.gate(torch.cat([outputs, forward_out, backward_out], dim=-1)))
        gate2 = torch.sigmoid(self.gate2(torch.cat([outputs, forward_out, backward_out], dim=-1)))
        gate3 = torch.sigmoid(self.gate3(torch.cat([outputs, forward_out, backward_out], dim=-1)))
        outputs = gate * outputs + gate2 * forward_out + gate3 * backward_out
        cls_embeddings = outputs[:, 0, :]
        return self.linear(cls_embeddings)
