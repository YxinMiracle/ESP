import logging
import pathlib
import pickle
import random
from argparse import ArgumentParser
from typing import List

import numpy as np

from model.dataloader.dataloader import get_train_dataloader
from model.dualgate import DualGate
from model.trainer.baseTrainer import BaseTrainer
from model_cofig.config import get_params
from process_model.knn.KnnRetrievalTask import KnnRetrievalTemplate
from process_model.model.PreDataProcess import ProcessTrainDataTemplate
from utils.helper import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_file_full_path_list_in_directory(directory_name: str) -> List[str]:
    files = []
    for entry in os.listdir(directory_name):
        full_path = os.path.join(directory_name, entry)
        if os.path.isfile(full_path):
            files.append(os.path.abspath(full_path))
    return files


def train_re_model(params: ArgumentParser, project_root_path: str):
    with open(params.data_directory_name + os.path.sep + params.ner2idx_file_name, "r") as f:
        ner2idx = json.load(f)  # type: dict
    with open(params.data_directory_name + os.path.sep + params.re2idx_file_name, "r") as f:
        rel2idx = json.load(f)  # type: dict

    train_batch, test_batch, dev_batch = get_train_dataloader(project_root_path, params, ner2idx, rel2idx)
    model = DualGate(params, ner2idx, rel2idx)
    model.cuda()
    trainer = BaseTrainer(params, model, project_root_path, ner2idx, rel2idx)
    trainer.train_model(train_batch, test_batch, dev_batch)


if __name__ == '__main__':
    params = get_params()
    set_seed(params.seed)

    root_dir = str(pathlib.Path(__file__).resolve().parent)
    data_directory_path = root_dir + os.path.sep + "data" + os.path.sep + params.data_directory_name

    with open(f"{data_directory_path}/total_re_task_data_list.pt", "rb") as fp:
        total_re_task_data_list = pickle.load(fp)

    p_obj = ProcessTrainDataTemplate(total_re_task_data_list, params)
    # p_obj.do_process()

    # train_re_model(params, root_dir)

    knn_obj = KnnRetrievalTemplate(total_re_task_data_list, params)
    # search_sent_data = knn_obj.source_data_list[105]
    # similar_sent_data = knn_obj.do_find_similar_sent(search_sent_data)  # type: List[IdentifiedReSentBaseData]

    # llm_re_obj = LLM_RE(params)
    # ans = llm_re_obj.do_llm_search(search_sent_data, similar_sent_data)
    # print(ans)
