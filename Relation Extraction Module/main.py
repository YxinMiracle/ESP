import logging
import random
from argparse import ArgumentParser
from typing import List

import numpy as np
from model.dataloader.dataloader import get_train_dataloader
from model.pfn import PFN
from model.trainer.baseTrainer import BaseTrainer

from huawei_data_process.process_json_data import process_file
from model_cofig.config import get_params
from process_model.knn.KnnResultVo import IdentifiedReSentBaseData
from process_model.model.ReBaseData import ReSentBaseData
from process_model.knn.KnnRetrievalTask import KnnRetrievalTemplate
from process_model.SearchByLLM import LLM_RE
from utils.helper import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    设置随机数种子
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_file_full_path_list_in_directory(directory_name: str) -> List[str]:
    """
    获取某个目录下的所有文件的绝对路径
    :param directory_name:
    :return:
    """
    files = []
    for entry in os.listdir(directory_name):
        full_path = os.path.join(directory_name, entry)
        if os.path.isfile(full_path):
            files.append(os.path.abspath(full_path))
    return files


def get_article_re_task_data(file_path: str, file_id: int) -> List[ReSentBaseData]:
    """
    返回一个文章中所有的能进行关系抽取的句子，会出现有跨句的情况
    :param file_path:
    :param file_id:
    :return:
    """
    with open(file_path, "r", encoding="UTF-8") as fp:
        file_data = json.load(fp)
    article_re_task_res = process_file(file_data, file_id)  # type:List[ReSentBaseData]
    return article_re_task_res


def train_re_model(params: ArgumentParser, project_root_path: str):
    """
    改函数用来训练关系抽取模型
    :param params: 参数
    :param project_root_path:
    :return:
    """
    with open(params.data_directory_name + os.path.sep + params.ner2idx_file_name, "r") as f:
        ner2idx = json.load(f)  # type: dict
    with open(params.data_directory_name + os.path.sep + params.re2idx_file_name, "r") as f:
        rel2idx = json.load(f)  # type: dict

    train_batch, test_batch, dev_batch = get_train_dataloader(project_root_path, params, ner2idx, rel2idx)
    model = PFN(params, ner2idx, rel2idx)
    model.cuda()
    trainer = BaseTrainer(params, model, project_root_path, ner2idx, rel2idx)
    trainer.train_model(train_batch, test_batch, dev_batch)


if __name__ == '__main__':
    input_dir_name = "./huawei_data_process/huawei_json"
    file_list = get_file_full_path_list_in_directory(input_dir_name)
    total_re_task_data_list = []  # type:List[ReSentBaseData]
    for file_id, file_name in enumerate(file_list):
        article_re_task_res = get_article_re_task_data(file_name, file_id)  # type:List[ReSentBaseData]
        if len(article_re_task_res) == 0: continue
        total_re_task_data_list.extend(article_re_task_res)
    # ==== 这里已经获得到了所有的句子以及对应的句子里面存在的实体和关系  ======
    params = get_params()
    set_seed(params.seed)
    """ 处理成模型所需形式 """
    # p_obj = ProcessTrainDataTemplate(total_re_task_data_list, params)
    # p_obj.do_process()
    """ 训练模型 """
    # train_re_model(params, p_obj.root_dir)
    """ KNN模型训练 """
    knn_obj = KnnRetrievalTemplate(total_re_task_data_list, params)
    search_sent_data = knn_obj.source_data_list[105]

    
    similar_sent_data = knn_obj.do_find_similar_sent(search_sent_data)  # type: List[IdentifiedReSentBaseData]
    """ 大模型进行关系抽取 """
    llm_re_obj = LLM_RE(params)
    ans = llm_re_obj.do_llm_search(search_sent_data, similar_sent_data)
