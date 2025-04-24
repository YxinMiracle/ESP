import logging

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import get_params
from src.pos_index_config import pos_index

nlp = spacy.load("en_core_web_sm")

logger = logging.getLogger()
params = get_params()
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name, local_files_only=True)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

DNRTI = ['O', 'B-Purp', 'I-Purp', 'B-Area', 'I-Area', 'B-SecTeam', 'I-SecTeam', 'B-SamFile', 'I-SamFile',
                 'B-Exp', 'I-Exp', 'B-Time', 'I-Time', 'B-Way', 'I-Way', 'B-OffAct', 'I-OffAct', 'B-Features',
                 'I-Features', 'B-Tool', 'I-Tool', 'B-Idus', 'I-Idus', 'B-HackOrg', 'I-HackOrg', 'B-Org', 'I-Org']

MalwareTextDB = ['O', 'B-Entity', 'I-Entity', 'B-Action', "I-Action", "B-Modifier", "I-Modifier"]  # 7

pos_list = ["[PPAD]", 'PROPN', 'ADJ', 'PUNCT', 'PRON', 'SCONJ', 'VERB', 'NOUN', 'PART', 'AUX', 'ADV', 'INTJ', 'X',
            'CCONJ', 'NUM', 'SYM', 'DET', 'ADP']

domain2labels = {"DNRTI": DNRTI, "MalwareTextDB": MalwareTextDB}


class Dataset(data.Dataset):
    def __init__(self, inputs, labels):
        self.X = inputs
        self.y = labels

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class PosDataset(data.Dataset):
    def __init__(self, inputs, labels, pos_list):
        self.X = inputs
        self.y = labels
        self.pos_list = pos_list

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.pos_list[index]

    def __len__(self):
        return len(self.X)


class GloveFeature:
    def __init__(self, glove_path):
        self.glove_path = glove_path
        self.glove_token2inx, self.glove_dim = self.glove_vocab()

    def glove_vocab(self):
        vocab = set()
        embed_dim = -1
        with open(self.glove_path, 'r', encoding="UTF-8") as file_read:
            for line in file_read:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split(' ')
                if embed_dim < 0:
                    embed_dim = len(tokens) - 1
                word = tokens[0]
                vocab.add(word)
        print('glove vocab done. {} tokens'.format(len(vocab)))
        glove_token2inx = {token: ind for ind, token in enumerate(vocab)}
        return glove_token2inx, embed_dim

    def load_glove_embedding(self):
        file_read = open(self.glove_path, 'r', encoding="UTF-8")
        embedding_dict = dict()
        for line in file_read:
            line = line.strip()
            if len(line) == 0:
                continue
            line = line.split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            embedding_dict[word] = np.array(embedding)

        return embedding_dict

def get_dataloader_for_bilstmtagger(params):
    vocab_tgt = get_vocab("data/%s/vocab.txt" % params.tgt_dm)
    vocab = Vocab()
    vocab.index_words(vocab_tgt)

    logger.info("Load training set data ...")
    inputs_train, labels_train = read_ner_for_bilstm("data/%s/train.txt" % params.tgt_dm, params.tgt_dm, vocab)

    logger.info("Load dev set data ...")
    inputs_dev, labels_dev = read_ner_for_bilstm("data/%s/dev.txt" % params.tgt_dm, params.tgt_dm, vocab)

    logger.info("Load test set data ...")
    inputs_test, labels_test = read_ner_for_bilstm("data/%s/test.txt" % params.tgt_dm, params.tgt_dm, vocab)

    logger.info("train size: %d; dev size %d; test size: %d;" % (len(inputs_train), len(inputs_dev), len(inputs_test)))

    dataset_train = Dataset(inputs_train, labels_train)
    dataset_dev = Dataset(inputs_dev, labels_dev)
    dataset_test = Dataset(inputs_test, labels_test)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=params.batch_size, shuffle=True,
                                  collate_fn=collate_fn_for_bilstm)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=params.batch_size, shuffle=False,
                                collate_fn=collate_fn_for_bilstm)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False,
                                 collate_fn=collate_fn_for_bilstm)

    return dataloader_train, dataloader_dev, dataloader_test, vocab

def read_ner_pos_data(datapath, dataset_name, rate=1):
    given_list = pos_index
    result_word_id_list, result_label_id_list = [], []
    result_pos_list = []
    with open(datapath, "r", encoding="UTF-8") as fp:
        sent_token_list, sent_label_list = [], []
        sent_list = []
        temp_pos_list = []
        sent_index = 0
        for idx, line in enumerate(fp):
            line = line.strip()
            if line == "":
                if len(sent_token_list) > 0:
                    assert len(sent_token_list) == len(sent_label_list)
                    doc = nlp(" ".join(sent_list))
                    for word in doc:
                        temp_pos_list.append(pos_list.index(word.pos_))
                    if given_list == sent_token_list:
                        print(sent_list)
                    result_word_id_list.append(
                        [auto_tokenizer.cls_token_id] + sent_token_list + [auto_tokenizer.sep_token_id])
                    temp_label_list = [pad_token_label_id] + sent_label_list + [pad_token_label_id]
                    result_label_id_list.append(temp_label_list)

                    temp_pos_list_2 = temp_label_list[1:-1]
                    for index, temp_label_id in enumerate(temp_label_list[1:-1]):
                        if temp_label_id != -100:
                            temp_pos_list_2[index] = temp_pos_list[sent_index]
                        else:
                            temp_pos_list_2[index] = -100
                    result_pos_list.append([-100] + temp_pos_list_2 + [-100])

                sent_token_list, sent_label_list, sent_list = [], [], []
                temp_pos_list = []
                continue

            word_and_label_list = line.split(' ')
            if len(word_and_label_list) == 2:
                word = word_and_label_list[0]
                sent_list.append(word)
                label = word_and_label_list[1]
                split_word_2_word_list = auto_tokenizer.tokenize(word)  # type: list
                if len(split_word_2_word_list) > 0:
                    sent_label_list.extend(
                        [domain2labels[dataset_name].index(label)] +
                        [pad_token_label_id] * (len(split_word_2_word_list) - 1)
                    )
                    sent_token_list.extend(auto_tokenizer.convert_tokens_to_ids(split_word_2_word_list))
                else:
                    logger.info("length of subwords for %s is zero; its label is %s" % (word, label))
    return result_word_id_list, result_label_id_list, result_pos_list


def read_ner_data(datapath, dataset_name, rate=1):
    result_word_id_list, result_label_id_list = [], []
    with open(datapath, "r", encoding="UTF-8") as fp:
        sent_token_list, sent_label_list = [], []
        for idx, line in enumerate(fp):
            line = line.strip()
            if line == "":
                if len(sent_token_list) > 0:
                    assert len(sent_token_list) == len(sent_label_list)
                    result_word_id_list.append(
                        [auto_tokenizer.cls_token_id] + sent_token_list + [auto_tokenizer.sep_token_id])
                    result_label_id_list.append([pad_token_label_id] + sent_label_list + [pad_token_label_id])
                sent_token_list, sent_label_list = [], []
                continue

            word_and_label_list = line.split(' ')
            if len(word_and_label_list) == 2:
                word = word_and_label_list[0]
                label = word_and_label_list[1]
                split_word_2_word_list = auto_tokenizer.tokenize(word)  # type: list
                if len(split_word_2_word_list) > 0:
                    sent_label_list.extend(
                        [domain2labels[dataset_name].index(label)] +
                        [pad_token_label_id] * (len(split_word_2_word_list) - 1)
                    )
                    sent_token_list.extend(auto_tokenizer.convert_tokens_to_ids(split_word_2_word_list))
                else:
                    logger.info("length of subwords for %s is zero; its label is %s" % (word, label))
    return result_word_id_list, result_label_id_list


def pos_collate_fn(data):
    X, y, pos_list = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(auto_tokenizer.pad_token_id)  # [bs* max_lengths]
    padded_y = torch.LongTensor(len(X), max_lengths).fill_(pad_token_label_id)
    padded_pos = torch.LongTensor(len(X), max_lengths).fill_(0)
    for i, (seq, y_, pos_data) in enumerate(zip(X, y, pos_list)):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
        padded_y[i, :length] = torch.LongTensor(y_)
        padded_pos[i, :length] = torch.LongTensor(pos_data)
    return padded_seqs, padded_y, padded_pos


def collate_fn(data):
    X, y = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(auto_tokenizer.pad_token_id)  # [bs* max_lengths]
    padded_y = torch.LongTensor(len(X), max_lengths).fill_(
        pad_token_label_id)

    for i, (seq, y_) in enumerate(zip(X, y)):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
        padded_y[i, :length] = torch.LongTensor(y_)

    return padded_seqs, padded_y


def get_dataloader(params):
    logger.info("load training dataloader")
    word_train_data, label_train_data = read_ner_data(params.ner_train_data_path % params.dataset_name, params.dataset_name)
    logger.info("load development dataloader")
    word_dev_data, label_dev_data = read_ner_data(params.ner_dev_data_path % params.dataset_name, params.dataset_name)
    logger.info("load test dataloader")
    word_test_data, label_test_data = read_ner_data(params.ner_test_data_path % params.dataset_name, params.dataset_name)

    logger.info("train size: %d; dev size %d; test size: %d;" % (len(word_train_data),
                                                                 len(word_dev_data),
                                                                 len(word_test_data)))

    train_dataset = Dataset(word_train_data, label_train_data)
    dev_dataset = Dataset(word_dev_data, label_dev_data)
    test_dataset = Dataset(word_test_data, label_test_data)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=params.batch_size,
                                  shuffle=params.shuffle,
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                batch_size=params.batch_size,
                                shuffle=params.shuffle,
                                collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=params.batch_size,
                                 shuffle=params.shuffle,
                                 collate_fn=collate_fn
                                 )
    return train_dataloader, dev_dataloader, test_dataloader


def get_pos_dataloader(params):
    logger.info("load training dataloader")
    word_train_data, label_train_data, pos_train_data = read_ner_pos_data(params.ner_train_data_path % params.dataset_name,
                                                                          params.dataset_name)
    logger.info("load development dataloader")
    word_dev_data, label_dev_data, pos_dev_data = read_ner_pos_data(params.ner_dev_data_path % params.dataset_name,
                                                                    params.dataset_name)
    logger.info("load test dataloader")
    word_test_data, label_test_data, post_test_data = read_ner_pos_data(params.ner_test_data_path % params.dataset_name,
                                                                        params.dataset_name)

    logger.info("train size: %d; dev size %d; test size: %d;" % (len(word_train_data),
                                                                 len(word_dev_data),
                                                                 len(word_test_data)))


    train_dataset = PosDataset(word_train_data, label_train_data, pos_train_data)
    dev_dataset = PosDataset(word_dev_data, label_dev_data, pos_dev_data)
    test_dataset = PosDataset(word_test_data, label_test_data, post_test_data)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=params.batch_size,
                                  shuffle=params.shuffle,
                                  collate_fn=pos_collate_fn)
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                batch_size=params.batch_size,
                                shuffle=params.shuffle,
                                collate_fn=pos_collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=params.batch_size,
                                 shuffle=params.shuffle,
                                 collate_fn=pos_collate_fn
                                 )
    return train_dataloader, dev_dataloader, test_dataloader


def collate_fn_for_bilstm(data):
    X, y = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)

    lengths = torch.LongTensor(lengths)
    return padded_seqs, lengths, y


PAD_INDEX = 0


class Vocab():
    def __init__(self):
        self.word2index = {"PAD": PAD_INDEX}
        self.index2word = {PAD_INDEX: "PAD"}
        self.n_words = 1

    def index_words(self, word_list):
        for word in word_list:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1


def read_ner_for_bilstm(datapath, dataset_name, vocab):
    inputs, labels = [], []
    with open(datapath, "r", encoding="UTF-8") as fr:
        token_list, label_list = [], []
        for i, line in enumerate(fr):
            line = line.strip()
            if line == "":
                if len(token_list) > 0:
                    inputs.append(token_list)
                    labels.append(label_list)

                token_list, label_list = [], []
                continue

            splits = line.split(" ")
            if len(splits) == 2:
                token = splits[0]
                label = splits[1]
                token_list.append(vocab.word2index[token])
                label_list.append(domain2labels[dataset_name].index(label))

    return inputs, labels


def get_vocab(path):
    vocabulary = []
    with open(path, "r", encoding="UTF-8") as f:
        for line in f:
            line = line.strip()
            vocabulary.append(line)
    return vocabulary



if __name__ == '__main__':
    from tqdm import tqdm

    params = get_params()
    word_train_data, label_train_data, train_pos_list = read_ner_pos_data("../../data/MalwareTextDB/train.txt",
                                                                          "MalwareTextDB")
    word_dev_data, label_dev_data, dev_pos_list = read_ner_pos_data("../../data/MalwareTextDB/dev.txt",
                                                                    "MalwareTextDB")
    word_test_data, label_test_data, test_pos_list = read_ner_pos_data("../../data/MalwareTextDB/test.txt",
                                                                       "MalwareTextDB")

    train_pos_dataset = PosDataset(word_train_data, label_train_data, train_pos_list)
    dataloader_train = DataLoader(dataset=train_pos_dataset, batch_size=64, shuffle=True, collate_fn=pos_collate_fn)
    pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
    loss_list = []
