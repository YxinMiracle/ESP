import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AlbertTokenizer, AlbertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cumsoftmax(x):
    return torch.cumsum(F.softmax(x, -1), dim=-1)


class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.bool
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)


class pfn_unit(nn.Module):
    def __init__(self, args, input_size):
        super(pfn_unit, self).__init__()
        self.args = args

        self.hidden_transform = LinearDropConnect(args.hidden_size, 5 * args.hidden_size, bias=True,
                                                  dropout=args.dropconnect)
        self.input_transform = nn.Linear(input_size, 5 * args.hidden_size, bias=True)  # [768, 5 * hidden_size]

        self.transform = nn.Linear(args.hidden_size * 3, args.hidden_size)
        self.drop_weight_modules = [self.hidden_transform]

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()

    def forward(self, x, hidden):
        h_in, c_in = hidden
        # 这里输入的就是x 和 h(t-1)，也就是，partition filter的输入
        gates = self.input_transform(x) + self.hidden_transform(h_in)  # shape[bs, 5*hidden_dim]
        c, eg_cin, rg_cin, eg_c, rg_c = gates[:, :].chunk(5, 1)  # all shape is [bs, hidden_dim]

        eg_cin = 1 - cumsoftmax(eg_cin)  # e(ct-1)  # 前面会包含更多的信息
        rg_cin = cumsoftmax(rg_cin)  # r(ct-1)  # 后面会包含更多的信息

        eg_c = 1 - cumsoftmax(eg_c)  # e(ct)
        rg_c = cumsoftmax(rg_c)  # r(ct)

        c = torch.tanh(c)  # c

        overlap_c = rg_c * eg_c  # ρs,ct
        upper_c = rg_c - overlap_c  # ρr,ct  # 这就是门控，对于关系的门控，针对上一个时刻的输入
        downer_c = eg_c - overlap_c  # ρe,ct

        overlap_cin = rg_cin * eg_cin  # ρs,ct−1
        upper_cin = rg_cin - overlap_cin  # ρr,ct−1 # 这就是门口，对于关系的门控，针对上一个时刻的输入
        downer_cin = eg_cin - overlap_cin  # ρe,ct−1

        share = overlap_cin * c_in + overlap_c * c  # ps

        # upper_cin * c_in + upper_c * c = pr
        # c_ner = downer_cin * c_in + downer_c * c = pe
        c_re = upper_cin * c_in + upper_c * c + share  # ur = pr + ps
        c_ner = downer_cin * c_in + downer_c * c + share  # ue = pe + ps
        c_share = share  # us = ps

        h_re = torch.tanh(c_re)  # hr = tanh(ur)
        h_ner = torch.tanh(c_ner)  # he = tanh(ue)
        h_share = torch.tanh(c_share)  # hs = tanh(us)

        c_out = torch.cat((c_re, c_ner, c_share), dim=-1)  # [µ(e,t);µ(r,t);µ(s,t)]
        c_out = self.transform(c_out)  # ct = Linear([µ(e,t);µ(r,t);µ(s,t)])
        h_out = torch.tanh(c_out)  # tanh(相当于遗忘门)

        return (h_out, c_out), (h_ner, h_re, h_share)


class encoder(nn.Module):
    def __init__(self, args, input_size):
        super(encoder, self).__init__()
        self.args = args
        self.unit = pfn_unit(args, input_size)

    def hidden_init(self, batch_size):
        h0 = torch.zeros(batch_size, self.args.hidden_size).requires_grad_(False).to(device)
        c0 = torch.zeros(batch_size, self.args.hidden_size).requires_grad_(False).to(device)
        return (h0, c0)

    def forward(self, x):
        seq_len = x.size(0)
        batch_size = x.size(1)
        h_ner, h_re, h_share = [], [], []
        hidden = self.hidden_init(batch_size)

        if self.training:
            self.unit.sample_masks()

        for t in range(seq_len):
            hidden, h_task = self.unit(x[t, :, :], hidden)
            h_ner.append(h_task[0])
            h_re.append(h_task[1])
            h_share.append(h_task[2])

        h_ner = torch.stack(h_ner, dim=0)
        h_re = torch.stack(h_re, dim=0)
        h_share = torch.stack(h_share, dim=0)

        return h_ner, h_re, h_share


class ner_unit(nn.Module):
    def __init__(self, args, ner2idx):
        super(ner_unit, self).__init__()
        self.hidden_size = args.hidden_size
        self.ner2idx = ner2idx

        self.hid2hid = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.hid2tag = nn.Linear(self.hidden_size, len(ner2idx))

        self.elu = nn.ELU()
        self.n = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, h_ner, h_share, mask):
        length, batch_size, _ = h_ner.size()

        h_global = torch.cat((h_share, h_ner), dim=-1)
        h_global = torch.tanh(self.n(h_global))  # 合并之后走入激活函数

        h_global = torch.max(h_global, dim=0)[0] # 每个句子选一个最大值出来
        h_global = h_global.unsqueeze(0).repeat(h_ner.size(0), 1, 1)
        h_global = h_global.unsqueeze(0).repeat(h_ner.size(0), 1, 1, 1)

        st = h_ner.unsqueeze(1).repeat(1, length, 1, 1)
        en = h_ner.unsqueeze(0).repeat(length, 1, 1, 1)

        ner = torch.cat((st, en, h_global), dim=-1)

        ner = self.ln(self.hid2hid(ner))
        ner = self.elu(self.dropout(ner))
        ner = torch.sigmoid(self.hid2tag(ner))

        diagonal_mask = torch.triu(torch.ones(batch_size, length, length)).to(device)
        diagonal_mask = diagonal_mask.permute(1, 2, 0)

        mask_s = mask.unsqueeze(1).repeat(1, length, 1)
        mask_e = mask.unsqueeze(0).repeat(length, 1, 1)

        mask_ner = mask_s * mask_e
        mask = diagonal_mask * mask_ner
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, len(self.ner2idx))

        ner = ner * mask

        return ner


class re_unit(nn.Module):
    def __init__(self, args, re2idx):
        super(re_unit, self).__init__()
        self.hidden_size = args.hidden_size
        self.relation_size = len(re2idx)
        self.re2idx = re2idx

        self.hid2hid = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.hid2rel = nn.Linear(self.hidden_size, self.relation_size)
        self.elu = nn.ELU()

        self.r = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, h_re, h_share, mask):
        length, batch_size, _ = h_re.size()

        h_global = torch.cat((h_share, h_re), dim=-1)
        re_global = torch.tanh(self.r(h_global))

        h_global = torch.max(re_global, dim=0)[0]
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1)
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1, 1)

        r1 = h_re.unsqueeze(1).repeat(1, length, 1, 1)
        r2 = h_re.unsqueeze(0).repeat(length, 1, 1, 1)

        re = torch.cat((r1, r2, h_global), dim=-1)

        re = self.ln(self.hid2hid(re))
        re = self.elu(self.dropout(re))
        re = torch.sigmoid(self.hid2rel(re))

        mask = mask.unsqueeze(-1).repeat(1, 1, self.relation_size)
        mask_e1 = mask.unsqueeze(1).repeat(1, length, 1, 1)
        mask_e2 = mask.unsqueeze(0).repeat(length, 1, 1, 1)
        mask = mask_e1 * mask_e2

        re = re * mask

        return re, re_global


class PFN(nn.Module):
    def __init__(self, args, ner2idx, rel2idx):
        super(PFN, self).__init__()
        self.args = args
        self.feature_extractor = encoder(self.args, self.args.input_size)

        self.ner = ner_unit(self.args, ner2idx)
        self.re = re_unit(self.args, rel2idx)
        self.dropout = nn.Dropout(self.args.dropout)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.bert = AutoModel.from_pretrained("bert-base-cased")

    def forward(self, x, mask):
        x = self.tokenizer(x, return_tensors="pt",
                           padding='longest',
                           is_split_into_words=True).to(device)
        x = self.bert(**x)[0]
        x = x.transpose(0, 1)

        if self.training:
            x = self.dropout(x)

        h_ner, h_re, h_share = self.feature_extractor(x)

        ner_score = self.ner(h_ner, h_share, mask)
        re_core, re_global = self.re(h_re, h_share, mask)
        return ner_score, re_core, re_global
