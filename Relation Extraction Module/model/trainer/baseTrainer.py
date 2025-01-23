import logging
from typing import List

import torch.nn as nn
from argparse import ArgumentParser
import torch.optim as optim
import os

import torch

from datetime import datetime
from utils.helper import SaveResults
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils.metrics import EvalMicro, EvalMacro, get_f1, LossBase

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, params: ArgumentParser, model: nn.Module, project_root_path: str, ner2idx: dict, rel2idx: dict):
        self.params = params  # type: ArgumentParser
        self.model = model  # type: nn.Module
        self.optimizer = optim.Adam(model.parameters(), lr=params.lr,
                                    weight_decay=params.weight_decay)  # type: optim.Adam
        self.output_dir = project_root_path + os.path.sep + params.save_directory_name  # type: str
        self.loss_fn = LossBase()  # type: LossBase
        self.ner2idx = ner2idx
        self.rel2idx = rel2idx
        self.saved_file_utils = self._init_save_log_file_params()
        self.saved_model_file_name = self.saved_file_utils.filepath + os.path.sep + self.params.saved_mode_name  # type: str
        logger.addHandler(
            logging.FileHandler(self.saved_file_utils.filepath + os.path.sep + self.params.train_log_file_name, 'w'))
        logger.info(params)

    def _init_save_log_file_params(self) -> SaveResults:
        """ 初始化存储训练模型日志的信息 """
        current_datetime = datetime.now()
        output_path_name = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')  # type: str

        file_path = self.output_dir + os.path.sep + output_path_name  # type: str

        return SaveResults(filepath=file_path,
                           filename=output_path_name + ".txt",
                           header="# epoch \t train_loss \t  dev_loss \t test_loss \t dev_ner \t dev_rel \t test_ner \t test_rel")

    def evaluate(self, test_batch: DataLoader, test_or_dev: str):
        """ 评估模型 """
        steps, test_loss = 0, 0
        total_triple_num = [0, 0, 0]
        total_entity_num = [0, 0, 0]
        if self.params.eval_metric == "macro":
            total_triple_num *= len(self.rel2idx)
            total_entity_num *= len(self.ner2idx)

        if self.params.eval_metric == "micro":
            metric = EvalMicro(self.rel2idx, self.ner2idx)
        else:
            metric = EvalMacro(self.rel2idx, self.ner2idx)

        self.model.eval()
        with torch.no_grad():
            for data in test_batch:
                steps += 1
                text = data[0]
                ner_label = data[1].cuda()
                re_label = data[2].cuda()
                mask = data[-1].cuda()

                ner_pred, re_pred, _ = self.model(text, mask)
                loss = self.loss_fn(ner_pred, ner_label, re_pred, re_label)
                test_loss += loss

                entity_num = metric.count_ner_num(ner_pred, ner_label)
                triple_num = metric.count_num(ner_pred, ner_label, re_pred, re_label)

                for i in range(len(entity_num)):
                    total_entity_num[i] += entity_num[i]
                for i in range(len(triple_num)):
                    total_triple_num[i] += triple_num[i]

            triple_result = get_f1(total_triple_num)
            entity_result = get_f1(total_entity_num)

            logger.info("------ {} Results ------".format(test_or_dev))
            logger.info("loss : {:.4f}".format(test_loss / steps))
            logger.info(
                "entity: p={:.4f}, r={:.4f}, f={:.4f}".format(entity_result["p"], entity_result["r"],
                                                              entity_result["f"]))
            logger.info(
                "triple: p={:.4f}, r={:.4f}, f={:.4f}".format(triple_result["p"], triple_result["r"],
                                                              triple_result["f"]))

        return triple_result, entity_result, test_loss / steps

    def train_step(self, data):
        """ 训练模型的每一步 """
        self.model.train()
        self.optimizer.zero_grad()

        text = data[0]  # type: List[List[str]]
        ner_label = data[1].cuda()
        re_label = data[2].cuda()
        mask = data[-1].cuda()
        ner_pred, re_pred, _ = self.model(text, mask)
        loss = self.loss_fn(ner_pred, ner_label, re_pred, re_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.params.clip)
        self.optimizer.step()
        return loss.item()

    def _can_save_model(self, epoch: int, average_f1: float, best_result: float) -> bool:
        """ 看看当前情况下是否可以保存模型，如果可以的话就返回True """
        if epoch == 0 or average_f1 > best_result:
            torch.save(self.model.state_dict(), self.saved_model_file_name)
            logger.info("Best result on dev saved!!!")
            return True
        return False

    def train_model(self, train_batch: DataLoader, test_batch: DataLoader, dev_batch: DataLoader):
        """ 训练模型的总步骤 """
        best_result, triple_best, entity_best = 0, None, None  # 初始化变量，用来判断是否可以用来保存最好的模型
        for epoch in range(self.params.epoch):
            self.model.train()
            pbar = tqdm(train_batch, total=len(train_batch))
            loss_list = []
            for data in pbar:
                loss = self.train_step(data)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(epoch, np.mean(loss_list)))
            logger.info("------ Training Set Results ------")
            logger.info("Finish training epoch %d. loss: %.4f" % (epoch, np.mean(loss_list)))
            logger.info("------ Start Evaluate ------")
            dev_triple, dev_entity, dev_loss = self.evaluate(dev_batch, "dev")
            test_triple, test_entity, test_loss = self.evaluate(test_batch, "test")
            average_f1 = dev_triple["f"] + dev_entity["f"]
            if self._can_save_model(epoch, average_f1, best_result):
                best_result = average_f1
                triple_best = test_triple
                entity_best = test_entity
            self.saved_file_utils.save("\t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}"
                                       .format(epoch, np.mean(loss_list), dev_loss, test_loss,
                                               dev_entity["f"], dev_triple["f"], test_entity["f"], test_triple["f"]))
        self.saved_file_utils.save(
            "best test result ner-p: {:.4f} \t ner-r: {:.4f} \t ner-f: {:.4f} \t re-p: {:.4f} \t re-r: {:.4f} \t re-f: {:.4f} ".format(
                entity_best["p"], entity_best["r"], entity_best["f"], triple_best["p"], triple_best["r"],
                triple_best["f"]))
