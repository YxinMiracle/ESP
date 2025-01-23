import torch
import numpy as np
import random
from src.BILSTM import BiLSTMTagger
from src.trainer import BaseTrainer
from src.utils import init_experiment
from src.coach.dataloader import get_dataloader, get_dataloader_for_bilstmtagger, get_pos_dataloader
from src.config import get_params
from src.BIGRU import BiGruTagger
from src.ALSAPs import ALSAP
from src.CRFmodel import CRFTAGGER

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(params):
    logger = init_experiment(params=params, logger_filename=params.logger_filename)

    if params.bilstm:
        dataloader_train, dataloader_dev, dataloader_test, vocab = get_dataloader_for_bilstmtagger(params)
        BiLSTM_model = BiLSTMTagger(params, vocab)
        BiLSTM_model.cuda()
        trainer = BaseTrainer(params, BiLSTM_model)
        trainer.train_bilstm_model(dataloader_train, dataloader_dev, dataloader_test, params.dataset_name)

    if params.crf:
        dataloader_train, dataloader_dev, dataloader_test, vocab = get_dataloader_for_bilstmtagger(params)
        BiLSTM_model = CRFTAGGER(params, vocab)
        BiLSTM_model.cuda()
        trainer = BaseTrainer(params, BiLSTM_model)
        trainer.train_bilstm_model(dataloader_train, dataloader_dev, dataloader_test, params.dataset_name)

    if params.bigru:
        dataloader_train, dataloader_dev, dataloader_test, vocab = get_dataloader_for_bilstmtagger(params)
        bigruModel = BiGruTagger(params=params, vocab=vocab)
        bigruModel.cuda()
        trainer = BaseTrainer(params, bigruModel)
        trainer.train_bigru_model(dataloader_train, dataloader_dev, dataloader_test, params.dataset_name)

    if params.ALSAP:
        train_datalodar, dev_dataloader, test_dataloader = get_pos_dataloader(params)
        model = ALSAP(params)
        model.cuda()
        trainer = BaseTrainer(params, model)
        trainer.train_ALSAP_model(train_datalodar, dev_dataloader, test_dataloader)


if __name__ == '__main__':
    params = get_params()
    random_seed(params.seed)
    train(params=params)
