import spacy
from config import POS_LIST, ENTITY_LABEL_LIST, BERT_MODEL_NAME, VALID_WORD_POSITION
from config import ENTITY_LABEL_LIST
from transformers import AutoTokenizer
import torch.nn as nn
import torch
from model import AttAttYxinNerModel_POS
import numpy as np
from seqeval.metrics import classification_report

nlp = spacy.load('en_core_web_sm')
# 加载bert中的tokenizer
auto_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, local_files_only=True)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

model = AttAttYxinNerModel_POS()
model.cuda()
model.eval()
model.load_state_dict(
    torch.load("/home/cyx/open_msan_ner/src/model_dir/huawei/yxinmiraclefinalmodel_pos/sub_k_7_2/model.pkl"))
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
optimizer.load_state_dict(
    torch.load("/home/cyx/open_msan_ner/src/model_dir/huawei/yxinmiraclefinalmodel_pos/sub_k_7_2/optimizer.pkl"))




def get_word_label_pos_index_list(sent_list: list):
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
            word_tokenizer_list = auto_tokenizer.tokenize(word_in_sent)
            word_index_list.extend(  # 构建好对应的bert词表中下标
                auto_tokenizer.convert_tokens_to_ids(word_tokenizer_list)
            )

            # 只有第一个位置是真正需要识别的，以后给pred出来的word用的东西
            true_word_index.extend([VALID_WORD_POSITION] + [pad_token_label_id] * (len(word_tokenizer_list) - 1))
            # 只有第一个位置是真正需要识别的，这个是pos的东西
            true_pos_list.extend([pos_list[word_num]] + [pad_token_label_id] * (len(word_tokenizer_list) - 1))

        result_word_index_list.append(
            [auto_tokenizer.cls_token_id] + word_index_list + [auto_tokenizer.sep_token_id]
        )
        result_true_word_index.append(
            [pad_token_label_id] + true_word_index + [pad_token_label_id]
        )
        result_pos_list.append(
            [pad_token_label_id] + true_pos_list + [pad_token_label_id]
        )
    return result_word_index_list, result_true_word_index, result_pos_list


def predict_entity(sent_list: list):
    result_word_index_list, result_true_word_index, result_pos_list = get_word_label_pos_index_list(sent_list)
    result_word_list, result_label_list = [], []
    for sent_index, (word_index_list, true_word_index, pos_list) in enumerate(
            zip(result_word_index_list, result_true_word_index, result_pos_list)):
        pred_list = []
        with torch.no_grad():
            X_torch = torch.tensor([word_index_list], device="cuda").long()
            POS_torch = torch.tensor([pos_list], device="cuda").long()
            preds = model.test(X_torch, pos_data=POS_torch)
            pred_list.extend(preds.data.cpu().numpy())
            pred_list = np.concatenate(pred_list, axis=0)
            pred_list = np.argmax(pred_list, axis=1)
            pred_list = list(pred_list)
            ans_list = []
            for pred_index in pred_list:
                pred_token = ENTITY_LABEL_LIST[pred_index]
                ans_list.append(pred_token)
            res_label_list = [ans_list[idx] for idx, value in enumerate(true_word_index) if value != pad_token_label_id]
            temp_word_list, temp_label_list = [], []
            for word, label in zip([str(word).strip() for word in nlp(sent_list[sent_index])], res_label_list):
                temp_word_list.append(word)
                temp_label_list.append(label)
            result_word_list.append(temp_word_list)
            result_label_list.append(temp_label_list)
    return result_word_list, result_label_list

# 将一段文本进行分为一个个句子
def get_cti_sent_list(cti_content: str):
    doc = nlp(cti_content)
    cti_sent_list = [sent.text for sent in doc.sents]
    return cti_sent_list


def test_ner(test_content: str):
    line_list = test_content.split("\n")
    result_word_id_list, result_label_id_list, result_pos_list = [], [], []
    sent_token_list, sent_label_list = [], []
    result_word_list, temp_word_list = [], []
    sent_list = []
    temp_pos_list = []
    __index = 0
    for sent_index, line in enumerate(line_list):
        line = line.strip()
        if line == "":  # sent_token_list以句子为单位进行单词的存储，len(line)==0表示一个句子单元以及读取完毕
            if len(sent_token_list) > 0 and len(sent_token_list) < 508:
                assert len(sent_token_list) == len(sent_label_list)  # 确保word_list和label_list一直是相同的
                doc = nlp(" ".join(sent_list))
                for word in doc:
                    try:
                        temp_pos_list.append(POS_LIST.index(word.pos_))  # temp_pos_list 存放的是每个单词对应的词性列表中的index
                    except:
                        temp_pos_list.append(0)
                result_word_id_list.append(
                    [auto_tokenizer.cls_token_id] + sent_token_list + [auto_tokenizer.sep_token_id])
                # 对应与[cls]、[sep] 都需添加pad进行对应
                temp_label_list = [pad_token_label_id] + sent_label_list + [pad_token_label_id]
                result_label_id_list.append(temp_label_list)

                temp_pos_list_2 = temp_label_list[1:-1]
                for index, temp_label_id in enumerate(temp_label_list[1:-1]):
                    if temp_label_id != pad_token_label_id:
                        temp_pos_list_2[index] = temp_pos_list[__index]
                        __index += 1
                    else:
                        temp_pos_list_2[index] = pad_token_label_id
                result_pos_list.append([pad_token_label_id] + temp_pos_list_2 + [pad_token_label_id])
            sent_token_list, sent_label_list, sent_list = [], [], []
            result_word_list.append(temp_word_list)
            temp_word_list = []
            temp_pos_list = []
            __index = 0
            continue

        word_and_label_list = line.split(' ')
        if len(word_and_label_list) == 2:
            word = word_and_label_list[0]
            sent_list.append(word)
            temp_word_list.append(word)
            label = word_and_label_list[1]
            split_word_2_word_list = auto_tokenizer.tokenize(word)  # type: list
            if len(split_word_2_word_list) > 0:
                sent_label_list.extend(
                    [ENTITY_LABEL_LIST.index(label)] +
                    [pad_token_label_id] * (len(split_word_2_word_list) - 1)
                )
                sent_token_list.extend(
                    auto_tokenizer.convert_tokens_to_ids(split_word_2_word_list))

    gold_tokens = []
    pred_tokens = []
    word_tokens = []
    total_word_num = 0
    lines = []
    for (word_id_list, label_id_list, pos_id_list, word_list) in zip(result_word_id_list, result_label_id_list,
                                                                     result_pos_list, result_word_list):
        pred_list = []
        with torch.no_grad():
            X_torch = torch.tensor([word_id_list], device="cuda").long()
            POS_torch = torch.tensor([pos_id_list], device="cuda").long()
            preds = model.test(X_torch, pos_data=POS_torch)
            pred_list.extend(preds.data.cpu().numpy())
            pred_list = np.concatenate(pred_list, axis=0)
            pred_list = np.argmax(pred_list, axis=1)
            pred_list = list(pred_list)
            word_tokens.extend(word_list)
            ans_list = []
            for pred_index in pred_list:
                pred_token = ENTITY_LABEL_LIST[pred_index]
                ans_list.append(pred_token)
            res_label_id_list = [value for idx, value in enumerate(label_id_list) if value != -100]
            res_label_list = [ENTITY_LABEL_LIST[value] for value in res_label_id_list]
            res_prd_label_list = [ans_list[idx] for idx, value in enumerate(label_id_list) if
                                  value != -100]  # 这是预测出来的标签

            for pred_token2, gold_token in zip(res_prd_label_list, res_label_list):
                lines.append("w" + " " + pred_token2 + " " + gold_token)
                pred_tokens.append(pred_token2)
                gold_tokens.append(gold_token)
            total_word_num += len(word_list)


    sent_num = len(result_word_id_list)
    report = classification_report([gold_tokens], [pred_tokens], digits=4)
    split_list = report.split("\n")
    result_list = []
    word_label_result_list = []  # 记录的是这个单词有没有被识别错误

    for index, (word, label, pred_label) in enumerate(zip(word_tokens, gold_tokens, pred_tokens)):
        word_label_result_list.append({"nerModelTestResultWord": word,
                                       "nerModelTestResultTrueLabel": label,
                                       "nerModelTestResultPredLabel": pred_label,
                                       "nerModelTestResultIsTrue": 1 if label == pred_label else 0,
                                       "nerModelTestResultWordIndex": index + 1,
                                       })

    for i in split_list:
        if len(i) == 0 or i == "\n" or i == "":
            continue
        if "f1-score" in i:
            continue
        step1_split_list = i.strip().split("    ")
        type_name = step1_split_list[0].strip()
        precision = step1_split_list[1].strip()
        recall = step1_split_list[2].strip()
        f1_score = step1_split_list[3].strip()
        result_list.append({"nerModelTestTypeName": type_name,
                            "nerModelTestPrecision": precision,
                            "nerModelTestRecall": recall,
                            "nerModelTestF1Score": f1_score
                            })
    model_res = {"nerModelTestList": result_list,  # 分数
                 "nerModelTestResults": word_label_result_list,  # 结果
                 "sentNum": sent_num
                 }
    print(model_res)
    return model_res

if __name__ == '__main__':
    word_list = ["cdnver.dll payload installed by the","cdnver.dll payload installed by the"]
    input_str = """
    The O
cdnver.dll O
payload O
installed O
by O
the O
loader O
executable O
is O
a O
variant O
of O
the O
SofacyCarberp B-malware
payload, O
which O
is O
used O
extensively O
by O
the O
Sofacy B-intrusion-set
threat O
group. O

We O
were O
able O
to O
uncover O
some O
other O
techniques O
used O
by O
this O
variant O
of O
ROKRAT B-malware
to O
make O
analysis O
difficult, O
Group B-threat-actor
123 I-threat-actor
used O
an O
anti-debugging O
technique O
related O
to O
NOP O
(No O
Operation). O
    """
    predict_entity(word_list)
    # test_ner(input_str)
