from src.ALSAPs import ALSAP
from config import get_params
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import torch.utils.data as data
from torch.utils.data import DataLoader

import torch.nn as nn
import spacy

DNRTI = ['O', 'B-Purp', 'I-Purp', 'B-Area', 'I-Area', 'B-SecTeam', 'I-SecTeam', 'B-SamFile', 'I-SamFile',
                 'B-Exp', 'I-Exp', 'B-Time', 'I-Time', 'B-Way', 'I-Way', 'B-OffAct', 'I-OffAct', 'B-Features',
                 'I-Features', 'B-Tool', 'I-Tool', 'B-Idus', 'I-Idus', 'B-HackOrg', 'I-HackOrg', 'B-Org', 'I-Org']

MalwareTextDB = ['O', 'B-Entity', 'I-Entity', 'B-Action', "I-Action", "B-Modifier", "I-Modifier"]  # 7

msb = ['O',
              'B-relevant_term',
              'B-vendor',
              'B-os',
              'I-relevant_term',
              'B-application',
              'B-version',
              'I-version',
              'I-os',
              'I-application',
              'B-update',
              'B-programming_language',
              'B-edition',
              'B-cve_id']

pos_list = ["[PPAD]", 'PROPN', 'ADJ', 'PUNCT', 'PRON', 'SCONJ', 'VERB', 'NOUN', 'PART', 'AUX', 'ADV', 'INTJ', 'X',
            'CCONJ', 'NUM', 'SYM', 'DET', 'ADP']

Enterprise = ['O', 'B-infrastructure_hosting-malware', 'B-authored-by', 'I-email-addr', 'I-hashes-to',
               'B-http-request-ext', 'O', 'B-hashes-to', 'I-malware_virus', 'B-malware_remote-access-trojan',
               'I-domain-name', 'B-process', 'I-windows-registry-key', 'B-attributed-to', 'B-uses', 'B-delivers',
               'B-malware_ransomware', 'I-infrastructure_victim', 'B-location', 'I-file-name', 'B-malware_ddos',
               'B-has', 'I-malware', 'I-alias-of', 'B-malware_webshell', 'B-threat-actor', 'B-malware', 'B-malware_bot',
               'B-file-hash', 'I-malware_ransomware', 'I-malware_bot', 'B-identity', 'I-authored-by',
               'B-communicates-with', 'B-ipv4-addr', 'I-http-request-ext', 'B-consists-of', 'I-malware_worm',
               'B-windows-registry-key', 'B-url', 'B-malware_virus', 'B-malware_keylogger', 'I-intrusion-set',
               'I-downloads', 'I-location', 'B-file-name', 'I-has', 'I-url', 'I-directory', 'B-exploits',
               'B-variant-of', 'I-software', 'B-software', 'I-targets', 'I-communicates-with', 'B-targets', 'I-process',
               'B-domain-name', 'B-alias-of', 'B-malware_worm', 'I-drops', 'I-tool', 'I-infrastructure_attack',
               'I-attributed-to', 'I-resolves-to', 'B-identity_victim', 'I-malware_remote-access-trojan',
               'I-malware_keylogger', 'B-infrastructure_exfiltration', 'B-infrastructure_command-and-control',
               'I-identity_victim', 'I-identity', 'B-malware_screen-capture', 'I-vulnerability', 'B-hosts',
               'I-beacons-to', 'I-infrastructure_hosting-malware', 'I-delivers', 'B-tool', 'I-consists-of',
               'B-vulnerability', 'B-infrastructure_attack', 'I-located-at', 'I-uses', 'I-hosts', 'I-exploits',
               'B-drops', 'I-malware_exploit-kit', 'I-infrastructure', 'B-infrastructure', 'B-intrusion-set',
               'I-campaign', 'B-downloads', 'B-user-account', 'B-mutex', 'B-located-at', 'B-beacons-to', 'B-campaign',
               'I-infrastructure_exfiltration', 'B-owns', 'B-ipv6-addr', 'I-threat-actor', 'B-infrastructure_victim',
               'I-variant-of', 'B-malware_exploit-kit', 'B-resolves-to', 'I-owns', 'B-email-addr', 'I-user-account',
               'B-directory', "B-compromises", "I-compromises", "I-ipv4-addr", "B-attack-pattern", "I-attack-pattern"]

domain2labels = {"DNRTI": DNRTI, "MalwareTextDB": MalwareTextDB, "msb": msb, "Enterprise": Enterprise}

pad_token_label_id = nn.CrossEntropyLoss().ignore_index

params = get_params()

nlp = spacy.load("en_core_web_sm")


class PosDataset(data.Dataset):
    def __init__(self, inputs, labels, pos_list):
        self.X = inputs
        self.y = labels
        self.pos_list = pos_list

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.pos_list[index]

    def __len__(self):
        return len(self.X)


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


auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name, local_files_only=True)


def read_ner_pos_data(datapath, dataset_name, rate=1):
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
    return result_word_id_list, result_label_id_list, result_pos_list


p_at_k_list = []
r_at_k_list = []
for k in range(2, 13):
    model = ALSAP(params)
    model.cuda()
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    model.load_state_dict(torch.load("model.pkl"))
    optimizer.load_state_dict(torch.load("optimizer.pkl"))
    word_test_data, label_test_data, post_test_data = read_ner_pos_data(
        r"./ner_data/DNRTI/test.txt", params.dataset_name)
    test_dataset = PosDataset(word_test_data, label_test_data, post_test_data)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=params.batch_size,
                                 shuffle=params.shuffle,
                                 collate_fn=pos_collate_fn
                                 )
    pred_list = []
    y_list = []
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    all_true_labels = []
    all_top_k_predictions = []

    for i, (X, y, pos_data) in pbar:
        X = X.cuda()
        preds = model.test(X, pos_data=pos_data)
        preds = preds.data.cpu().numpy()
        top_k_indices = np.argsort(preds, axis=-1)[..., -k:]

        true_labels = y.numpy()

        for i in range(X.size(0)):
            sequence_true_labels = true_labels[i]
            sequence_top_k_pred_indices = top_k_indices[i]

            for j in range(X.size(1)):
                true_label_index = sequence_true_labels[j]
                top_k_pred_indices = sequence_top_k_pred_indices[j]

                if true_label_index != -100:
                    true_label = domain2labels[params.dataset_name][true_label_index]
                    all_true_labels.append(true_label)
                    all_top_k_predictions.append(top_k_pred_indices)

    correct_predictions = 0
    predicted_entities = 0
    f1_predicted_entities = 0
    f1_correct_entity = 0

    for true_label, top_k_pred_indices in zip(all_true_labels, all_top_k_predictions):
        f1_predicted_entities += 1
        temp_label_list = []
        for list_index in top_k_pred_indices:
            temp_label_list.append(domain2labels[params.dataset_name][list_index])
        if true_label in temp_label_list:
            f1_correct_entity += 1
        if true_label != 'O':
            predicted_entities += 1
            if true_label in temp_label_list:
                correct_predictions += 1

    total_entities = len([label for label in all_true_labels if label != 'O'])

    P_at_k = f1_correct_entity / f1_predicted_entities if f1_predicted_entities > 0 else 0
    R_at_k = correct_predictions / total_entities if total_entities > 0 else 0
    p_at_k_list.append(P_at_k)
    r_at_k_list.append(R_at_k)
    print(f"P@{k}: {P_at_k:.4f}")
    print(f"R@{k}: {R_at_k:.4f}")
print(p_at_k_list)
print(r_at_k_list)
