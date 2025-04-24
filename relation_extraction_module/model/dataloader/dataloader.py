import os.path
import random
from argparse import ArgumentParser

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from utils.helper import *


class CTI_RE_DATASET(Dataset):
    def __init__(self, data, max_seq_len):
        self.data = data
        self.len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words = self.data[idx][0]
        ner_labels = self.data[idx][1]
        rc_labels = self.data[idx][2]

        if len(words) > self.len:
            words, ner_labels, rc_labels = self.truncate(self.len, words, ner_labels, rc_labels)

        sent_str = ' '.join(words)
        bert_words = self.tokenizer.tokenize(sent_str)
        bert_len = len(bert_words) + 2
        # bert_len = original sentence + [CLS] and [SEP]

        word_to_bep = self.map_origin_word_to_bert(words)
        ner_labels = self.ner_label_transform(ner_labels, word_to_bep)
        rc_labels = self.rc_label_transform(rc_labels, word_to_bep)

        return (words, ner_labels, rc_labels, bert_len)

    def map_origin_word_to_bert(self, words):
        bep_dict = {}
        current_idx = 0
        for word_idx, word in enumerate(words):
            bert_word = self.tokenizer.tokenize(word)
            word_len = len(bert_word)
            bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
            current_idx = current_idx + word_len
        return bep_dict

    def ner_label_transform(self, ner_label, word_to_bert):
        new_ner_labels = []

        for i in range(0, len(ner_label), 3):
            # +1 for [CLS]
            sta = word_to_bert[ner_label[i]][0] + 1
            end = word_to_bert[ner_label[i + 1]][0] + 1
            new_ner_labels += [sta, end, ner_label[i + 2]]

        return new_ner_labels

    def rc_label_transform(self, rc_label, word_to_bert):
        new_rc_labels = []

        for i in range(0, len(rc_label), 3):
            # +1 for [CLS]
            e1 = word_to_bert[rc_label[i]][0] + 1
            e2 = word_to_bert[rc_label[i + 1]][0] + 1
            new_rc_labels += [e1, e2, rc_label[i + 2]]

        return new_rc_labels

    def truncate(self, max_seq_len, words, ner_labels, rc_labels):
        truncated_words = words[:max_seq_len]
        truncated_ner_labels = []
        truncated_rc_labels = []
        for i in range(0, len(ner_labels), 3):
            if ner_labels[i] < max_seq_len and ner_labels[i + 1] < max_seq_len:
                truncated_ner_labels += [ner_labels[i], ner_labels[i + 1], ner_labels[i + 2]]

        for i in range(0, len(rc_labels), 3):
            if rc_labels[i] < max_seq_len and rc_labels[i + 1] < max_seq_len:
                truncated_rc_labels += [rc_labels[i], rc_labels[i + 1], rc_labels[i + 2]]

        return truncated_words, truncated_ner_labels, truncated_rc_labels


def data_preprocess(data):
    processed = []
    for dic in data:
        text = dic['tokens']
        ner_labels = []  # [2, 4, 'Adverse-Effect', 11, 11, 'Drug', 0, 0, 'Adverse-Effect']
        rc_labels = []  # [2, 11, 'Adverse-Effect', 0, 11, 'Adverse-Effect']
        entity = dic['entities']
        relation = dic['relations']

        for en in entity:
            ner_labels += [en['start'], en['end'] - 1, en['type']]

        for re in relation:
            subj_idx = re['head']
            obj_idx = re['tail']
            subj = entity[subj_idx]
            obj = entity[obj_idx]
            rc_labels += [subj['start'], obj['start'], re['type']]

        processed += [(text, ner_labels, rc_labels)]
    return processed


def get_train_dataloader(project_root_path: str, params: ArgumentParser, ner2idx: dict, rel2idx: dict):
    path = project_root_path + os.path.sep + params.data_directory_name + os.path.sep

    train_raw_data = json_load(path, params.train_file_name)
    test_data = json_load(path, params.test_file_name)
    random.shuffle(train_raw_data)
    split = int(0.15 * len(train_raw_data))
    train_data = train_raw_data[split:]
    dev_data = train_raw_data[:split]

    train_data = data_preprocess(train_data)
    test_data = data_preprocess(test_data)
    dev_data = data_preprocess(dev_data)

    train_dataset = CTI_RE_DATASET(train_data, params.max_seq_len)
    test_dataset = CTI_RE_DATASET(test_data, params.max_seq_len)
    dev_dataset = CTI_RE_DATASET(dev_data, params.max_seq_len)
    collate_fn = collater(ner2idx, rel2idx)

    train_batch = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True, pin_memory=True,
                             collate_fn=collate_fn,drop_last=True)
    test_batch = DataLoader(dataset=test_dataset, batch_size=params.eval_batch_size, shuffle=False, pin_memory=True,
                            collate_fn=collate_fn,drop_last=True)
    dev_batch = DataLoader(dataset=dev_dataset, batch_size=params.eval_batch_size, shuffle=False, pin_memory=True,
                           collate_fn=collate_fn,drop_last=True)

    return train_batch, test_batch, dev_batch
