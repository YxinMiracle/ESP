import datetime
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score


def time_format(time_diff):
    seconds = int(round(time_diff))
    return str(datetime.timedelta(seconds=seconds))


def self_f1_score(y_pred, y_true):
    f1 = f1_score(y_true, y_pred, average='weighted')
    return f1


def bert_evaluate(eva_model, eva_dataloader, eva_epoch_th, eva_device, eva_dataset_name):
    eva_model.eval()
    all_pred_labels = []
    all_true_labels = []
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for eva_batch in eva_dataloader:
            eva_batch_data = tuple(item.to(eva_device) for item in eva_batch)
            eva_input_ids, eva_true_mask, eva_seg_ids, eva_pre_mask, eva_true_label_ids, eva_true_label_mask = eva_batch_data
            pred_labels_ids = eva_model(eva_input_ids, eva_true_mask, eva_seg_ids, eva_pre_mask,eva_true_label_mask)

            pre_label_list = [each_pre_label for each_pre_sent in pred_labels_ids for each_pre_label in each_pre_sent]
            print(pre_label_list)
            all_pred_labels.extend(pre_label_list)
            valid_preds = torch.tensor(pre_label_list, dtype=torch.long).to(eva_device)

            valid_true_tensor = torch.masked_select(eva_true_label_ids, eva_true_label_mask.bool())
            valid_true = valid_true_tensor.cpu().detach().tolist()
            print(valid_true)
            all_true_labels.extend(valid_true)

            assert len(all_pred_labels) == len(all_true_labels)

            total = total + len(valid_true)
            assert total == len(all_pred_labels)

            correct = correct + valid_preds.eq(valid_true_tensor).sum().item()
    average_acc = correct / total
    assert len(all_true_labels) == len(all_pred_labels)
    f1 = self_f1_score(np.array(all_pred_labels), np.array(all_true_labels))
    end = time.time()
    print("This is %s:\n Epoch:%d\n Acc:%.2f\n F1: %.2f\n Spending: %s" % \
          (eva_dataset_name,
            eva_epoch_th,
            average_acc * 100.,
            f1 * 100.,
           time_format(end - start)))
    return average_acc, f1


def lc_cal_f1(true_tags, pred_tags):
    return f1_score(true_tags, pred_tags, average='weighted')


def lc_cal_acc(true_tags, pred_tags):
    return accuracy_score(np.array(true_tags), np.array(pred_tags))
