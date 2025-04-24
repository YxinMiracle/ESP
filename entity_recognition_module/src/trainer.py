import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from seqeval.metrics import classification_report
from tqdm import tqdm

from src.coach.dataloader import domain2labels, pad_token_label_id
from src.conll2002_metrics import *

logger = logging.getLogger()


class BaseTrainer(object):
    def __init__(self, params, model):
        self.params = params
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        self.loss_fn = nn.CrossEntropyLoss()
        try:
            self.early_stop = params.early_stop
        except:
            self.early_stop = 30
        self.no_improvement_num = 0
        self.best_acc = 0

    def model_size_in_mb(self):
        total_size = 0
        for param in self.model.parameters():
            total_size += param.nelement() * param.element_size()

        size_in_mb = total_size / (1024 * 1024)
        return size_in_mb

    def train_step(self, X, y, domain=False):
        # y.shape = [32, 50]
        # X.shape = [32, 50]
        self.model.train()  # 训练模式
        preds = self.model(X, domain=domain)  # domain=False
        y = y.view(y.size(0) * y.size(1))
        preds = preds.view(preds.size(0) * preds.size(1), preds.size(2))
        self.optimizer.zero_grad()
        loss = self.loss_fn(preds, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step_for_bilstm(self, X, lengths, y):
        self.model.train()
        preds = self.model(X)
        loss = self.model.crf_loss(preds, lengths, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step_for_bigru(self, X, lengths, y):
        self.model.train()
        preds = self.model(X)
        loss = self.model.crf_loss(preds, lengths, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def esp_train_step(self, X, y, pos_data):
        self.model.train()

        preds, hcl_loss = self.model(X, y=y, pos_data=pos_data)
        y = y.view(y.size(0) * y.size(1))
        preds = preds.view(preds.size(0) * preds.size(1), preds.size(2))

        self.optimizer.zero_grad()
        loss = self.loss_fn(preds, y)
        if self.params.target_sequence:
            loss = loss + hcl_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_score_in_report(self, report):
        split_list = report.split("\n")
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

            if type_name == "micro avg":
                return precision

    def evaluate(self, dataloader, dataset_name, use_bilstm=False, use_ESP=False):
        self.model.eval()

        pred_list = []
        y_list = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        if use_bilstm:
            for i, (X, lengths, y) in pbar:
                y_list.extend(y)
                X, lengths = X.cuda(), lengths.cuda()
                preds = self.model(X)
                preds = self.model.crf_decode(preds, lengths)
                pred_list.extend(preds)
        elif use_ESP:
            for i, (X, y, pos_data) in pbar:
                y_list.extend(y.data.numpy())  # y is a list
                X = X.cuda()
                preds = self.model.test(X, pos_data=pos_data)
                pred_list.extend(preds.data.cpu().numpy())
        else:
            for idx, (X, y) in pbar:
                y_list.extend(y.data.numpy())
                X = X.cuda()
                pred = self.model(X)
                pred_list.extend(pred.data.cpu().numpy())

        # pred shape is [16, 66, 27]
        pred_list = np.concatenate(pred_list, axis=0)  # (length, num_tag)
        if not use_bilstm:
            pred_list = np.argmax(pred_list, axis=1)
        y_list = np.concatenate(y_list, axis=0)

        pred_list = list(pred_list)
        y_list = list(y_list)
        lines = []

        pred_tokens = []
        gold_tokens = []
        for pred_index, gold_index in zip(pred_list, y_list):
            gold_index = int(gold_index)
            if gold_index != pad_token_label_id:
                pred_token = domain2labels[dataset_name][pred_index]
                gold_token = domain2labels[dataset_name][gold_index]
                lines.append("w" + " " + pred_token + " " + gold_token)
                pred_tokens.append(pred_token)
                gold_tokens.append(gold_token)

        report = classification_report([gold_tokens], [pred_tokens], digits=4)
        precision = self.get_score_in_report(report)
        results = conll2002_measure(lines)
        f1 = results["fb1"]
        return f1, report, precision

    def train_ESP_model(self, dataloader_train, dataloader_dev, dataloader_test):
        no_improvement_num = 0
        best_f1 = 0
        dev_target = []
        dev_detail_results = []
        train_ave_loss = []
        valid_acc_score = []
        valid_f1_score = []
        dev_report = []
        best_test_f1 = 0
        epoch_f1_list = []
        epoch_time_list = []
        best_test_epoch = -1
        for e in range(self.params.epoch):
            start_time = time.time()
            logger.info("============== epoch %d ==============" % e)
            pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            loss_list = []
            for i, (X, y, pos_data) in pbar:
                X, y, pos_data = X.cuda(), y.cuda(), pos_data.cuda()
                loss = self.esp_train_step(X, y, pos_data)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))
            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))
            train_ave_loss.append(round(np.mean(loss_list), 2))
            end_time = time.time()
            logger.info("============== Evaluate epoch %d on Train Set ==============" % e)
            f1_train, report_results, precision = self.evaluate(dataloader_train, self.params.dataset_name,
                                                                use_bilstm=self.params.bilstm,
                                                                use_ESP=self.params.ESP)
            logger.info("\n%s", report_results)

            logger.info("Evaluate on Train Set. F1: %.4f." % f1_train)

            logger.info("============== Evaluate epoch %d on Dev Set ==============" % e)
            f1_dev, report_results, precision = self.evaluate(dataloader_dev, self.params.dataset_name,
                                                              use_bilstm=self.params.bilstm,
                                                              use_ESP=self.params.ESP)
            logger.info("\n%s", report_results)

            logger.info("Evaluate on Dev Set. F1: %.4f." % f1_dev)
            dev_target.append(f1_dev)
            dev_detail_results.append(report_results)
            valid_f1_score.append(round(f1_dev, 2))
            valid_acc_score.append(round(float(precision) * 100, 2))
            dev_report.append(report_results)

            if f1_dev > best_f1:
                best_f1 = f1_dev
                no_improvement_num = 0
            else:
                no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (no_improvement_num, self.params.early_stop))
            if no_improvement_num >= self.params.early_stop:
                break

            logger.info("============== Evaluate epoch on Test Set ==============")
            f1_test, report_results, precision = self.evaluate(dataloader_test, self.params.dataset_name,
                                                               use_bilstm=self.params.bilstm,
                                                               use_ESP=self.params.ESP)
            epoch_f1_list.append(f1_test)
            epoch_time_list.append(end_time - start_time)
            if f1_test > best_test_f1:
                best_test_epoch = e
                best_test_f1 = f1_test
            logger.info("\n%s", report_results)
            logger.info("Best dev F1: %.4f." % best_f1)
            logger.info("Best test F1: %.4f." % best_test_f1)
            logger.info("Best epoch F1: %.4f." % best_test_epoch)
            logger.info("Evaluate on Test Set. F1: %.4f." % f1_test)
            logger.info("model parameters size: %.4f." % self.model_size_in_mb())
            logger.info("one epoch time: %.4f." % (end_time - start_time))
            logger.info("avg epoch time: %.4f." % np.mean(epoch_time_list))
            logger.info("epoch f1 list: %s" % epoch_f1_list)

    def train_bilstm_model(self, dataloader_train, dataloader_dev, dataloader_test, dataset_name):
        logger.info("now start train bert+bilstm+crf model")
        no_improvement_num = 0
        best_f1 = 0

        train_ave_loss = []
        valid_acc_score = []
        valid_f1_score = []
        # valid_loss_score = []
        dev_report = []

        f1_score_list = []

        for e in range(self.params.epoch):
            logger.info("============== bert+bilstm+crf model epoch %d ==============" % e)
            pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            loss_list = []
            for i, (X, lengths, y) in pbar:
                X, lengths = X.cuda(), lengths.cuda()
                loss = self.train_step_for_bilstm(X, lengths, y)
                loss_list.append(loss)
                pbar.set_description("(bert+bilstm+crf model Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))
            train_ave_loss.append(round(np.mean(loss_list), 2))
            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))
            logger.info(
                "==================== bert+bilstm+crf model Evaluate epoch %d on Dev Set ===================" % e)
            f1_dev, report_results, precision = self.evaluate(dataloader_dev, dataset_name, use_bilstm=True)
            dev_report.append(report_results)
            logger.info("\n %s", report_results)

            valid_f1_score.append(round(f1_dev, 2))
            valid_acc_score.append(round(float(precision) * 100, 2))
            logger.info("Evaluate on Dev Set. F1: %.4f" % f1_dev)

            if f1_dev > best_f1:
                logger.info("Found better model!")
                best_f1 = f1_dev
                no_improvement_num = 0
            else:
                no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (no_improvement_num, 1))

            logger.info("============== Evaluate on Test Set ==============")
            f1_test, report_results, precision = self.evaluate(dataloader_test, dataset_name, use_bilstm=True)
            logger.info("\n%s", report_results)
            logger.info("Evaluate on Test Set. F1: %.4f." % f1_test)
            f1_score_list.append(f1_test)
            logger.info("epoch f1 list: %s" % f1_score_list)

    def train_bigru_model(self, dataloader_train, dataloader_dev, dataloader_test, dataset_name):
        logger.info("now start train bigru model")
        no_improvement_num = 0
        best_f1 = 0

        train_ave_loss = []
        valid_acc_score = []
        valid_f1_score = []
        # valid_loss_score = []
        dev_report = []
        f1_score_list = []

        for e in range(self.params.epoch):
            logger.info("============== bigru model epoch %d ==============" % e)
            pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            loss_list = []
            for i, (X, lengths, y) in pbar:
                X, lengths = X.cuda(), lengths.cuda()
                loss = self.train_step_for_bigru(X, lengths, y)
                loss_list.append(loss)
                pbar.set_description("(bigru model Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))
            train_ave_loss.append(round(np.mean(loss_list), 2))
            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))
            logger.info(
                "==================== bigru model Evaluate epoch %d on Dev Set ===================" % e)
            f1_dev, report_results, precision = self.evaluate(dataloader_dev, dataset_name, use_bilstm=True)
            dev_report.append(report_results)
            logger.info("\n %s", report_results)
            logger.info("Evaluate on Dev Set. F1: %.4f" % f1_dev)

            valid_f1_score.append(round(f1_dev, 2))
            valid_acc_score.append(round(float(precision) * 100, 2))

            if f1_dev > best_f1:
                logger.info("Found better model!")

                best_f1 = f1_dev
                no_improvement_num = 0
            else:
                no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (no_improvement_num, 1))

            logger.info("============== Evaluate on Test Set ==============")
            f1_test, report_results, precision = self.evaluate(dataloader_test, dataset_name, use_bilstm=True)
            f1_score_list.append(f1_test)
            logger.info("\n%s", report_results)
            logger.info("Evaluate on Test Set. F1: %.4f." % f1_test)
            logger.info("Evaluate on Test Set. F1: %s." % f1_score_list)

    def save_model(self, model_name):
        """
        save the best model
        """
        saved_path = os.path.join(self.params.dump_path, f"{model_name}.pth")
        torch.save({
            "model": self.model,
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)
