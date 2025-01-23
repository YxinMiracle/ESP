import pathlib
from typing import List
import os
import pickle
import json
import torch

from process_model.knn.KnnResultVo import IdentifiedReSentBaseData
from process_model.model.ReBaseData import ReSentBaseData
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class LLM_RE:
    GET_REASON_TEMPLATE = """
    You are an expert in the field of cybersecurity threat intelligence. Next, I will have some preprocessing tasks related to extracting relationships from threat intelligence texts. The specific task is as follows: I will provide you with a JSON string, formatted as:

    {"sent": "specific sentence", "head_entity_name": "head entity name", "head_entity_type": "head entity type", "tail_entity_name": "tail entity name", "tail_entity_type": "tail entity type", "relation_type": "type of relationship between entities"}

    In this JSON string, sent represents the sentence to be processed, head_entity_name and head_entity_type respectively represent the name and type of the head entity in the sentence. tail_entity_name and tail_entity_type respectively represent the name and type of the tail entity in the sentence, relation_type describes the specific relationship between the head and tail entities in the sentence (such as "uses"). 
    Your task is to analyze and respond why these two entities have this type of relationship in the sentence, and briefly summarize the reasons in one to two sentences, ensuring the explanation is clear and concise.
    """
    GET_RE_ANSWER_TEMPLATE = """
    You are an expert in threat intelligence for cybersecurity. Next, I will have a task related to extracting relationships from threat intelligence texts. Specifically, I will input a JSON string describing the task, and you will need to return a JSON string with the results of the relation extraction task. The JSON string I input will have three main parts.
    The first part is "Task Description". This part describes the task and contains two fields. The first field is "description", which describes this task. Here, I will tell you the predefined relations available for this task, such as beacons-to, targets, NULL, etc., separated by commas. Please note that there might also be no relationship between two entities, in which case you should respond with "NULL". The second field is "predefined_relations", which is a list containing the relations you can choose from.

    The second part is "ICL Demonstrations", a list of five JSON strings, each as an example. Each example is a JSON string formatted as: {"sent": "specific sentence", "head_entity_name": "head entity name", "head_entity_type": "head entity type", "tail_entity_name": "tail entity name", "tail_entity_type": "tail entity type", "relation_type": "type of relationship between entities", "reason": "The reason why these two entities belong to this relation"}.
    In this JSON string, "sent" refers to the sentence you need to process. "head_entity_name" and "head_entity_type" refer to the name and type of the head entity in the sentence, respectively. "tail_entity_name" and "tail_entity_type" refer to the name and type of the tail entity, respectively. "relation_type" describes the relationship between these two entities within the sentence, and "reason" explains why these two entities belong to this relationship. Please carefully study each example to understand how to extract and infer entity relationships from sentences.

    The last part is "Test Input", for which I will input a JSON string, {"sent": "specific sentence", "head_entity_name": "head entity name", "head_entity_type": "head entity type", "tail_entity_name": "tail entity name", "tail_entity_type": "tail entity type"}. This JSON string is similar to the ones above but lacks the reason and relation. You need to determine the relationship and respond with a JSON string, such as {"relation_type":"uses"} or {"relation_type":"NULL"}...etc., to indicate what the relationship between the two entities in the input sentence should be. Note, please do not output any extra content.
    """

    ICL_DEMONSTRATION_TEMPLATE = "Context: <\"{sent}\"> \n Given the context, the relation between <\"{head_entity_name}\"> and <\"{tail_entity_name}\"> is <\"{relation_name}\">."
    QUERY_TEMPLATE = "What are the clues that lead to the relation between <\"{head_entity_name}\"> and <\"{tail_entity_name}\"> to be <\"{relation_name}\"> in the sentence <\"{sent}\">"
    REASON_TEMPLATE = "Reason: {reason}"
    PROMPT_TEMPLATE = "I will predict the relation between two entities given the context. The pre-defined relations are {relation_list_str}. \n I will output NULL if the relation does not belong to them."
    TASK_INPUT_TEMPLATE = "Context: <\"{sent}\">. Given the context, the relation between <\"{head_entity_name}\"> and <\"{tail_entity_name}\"> is < _____ >"

    def __init__(self, params):
        self.params = params
        self.model_name = self.params.llm_model_path
        self.root_dir = str(pathlib.Path(__file__).resolve().parent.parent)
        self.goal_label_cache = self._init_goal_label_cache()
        self.llm_model, self.llm_tokenizer = self._init_llm_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.re2idx_file_path = self.root_dir + os.path.sep + self.params.data_directory_name + os.path.sep + self.params.re2idx_file_name
        self.entity_relation_rule_dict = self._get_entity_relation_rule_dict()

    def _get_entity_relation_rule_dict(self) -> dict:
        with open(
                self.root_dir + os.path.sep + self.params.data_directory_name + os.path.sep + self.params.entity_relation_rule_file_name) as fp:
            entity_relation_rule_dict = json.load(fp)
        return entity_relation_rule_dict

    def _init_llm_model(self):
        """ 初始化本地大模型 """
        llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        llm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return llm_model, llm_tokenizer

    def get_llm_answer(self, prompt: str, question: str) -> str:
        """ 获取大模型答案 """
        messages = [
            {"role": "system",
             "content": prompt},
            {"role": "user", "content": question}
        ]
        text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.llm_model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def _init_goal_label_cache(self):
        """ 初始化金标签缓存 """
        cache_file_name = self.root_dir + os.path.sep + self.params.data_directory_name + os.path.sep + self.params.gold_label_cache_file_name  # type: str
        if os.path.exists(cache_file_name):
            logger.info("有缓存文件，直接获取就ok")
            with open(cache_file_name, "rb") as fp:
                goal_label_cache = pickle.load(fp)  # type: dict
            return goal_label_cache
        logger.info("没有保存缓存文件，需要进行初始化")
        init_cache_dict = {}  # type: dict
        os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
        with open(cache_file_name, "wb") as fp:
            pickle.dump(init_cache_dict, fp)
        return init_cache_dict

    def _update_goal_label_cache(self, new_cache_data: dict):
        """ 更新缓存 """
        cache_file_name = self.root_dir + os.path.sep + self.params.data_directory_name + os.path.sep + self.params.gold_label_cache_file_name  # type: str
        with open(cache_file_name, "wb") as fp:
            logger.info("更新金标签缓存")
            pickle.dump(new_cache_data, fp)
        logger.info("缓存更新成功")

    def get_gold_label_induced(self, similar_sent_list: List[IdentifiedReSentBaseData]) -> List[dict]:
        """ 金标签提示 """
        demonstration_answer_list = []  # type: List[dict]
        for similar_sent in similar_sent_list:
            similar_sent_id = similar_sent.id  # type: int
            if similar_sent_id not in self.goal_label_cache:
                logger.info("该查询结果[不在]缓存中")

                question_dict = {}
                question_dict["sent"] = similar_sent.sent
                question_dict["head_entity_name"] = similar_sent.head_entity.entity_name
                question_dict["head_entity_type"] = similar_sent.head_entity.entity_type
                question_dict["tail_entity_name"] = similar_sent.tail_entity.entity_name
                question_dict["tail_entity_type"] = similar_sent.tail_entity.entity_type
                question_dict["relation_type"] = similar_sent.relation_type

                reason = self.get_llm_answer(prompt=self.GET_REASON_TEMPLATE, question=json.dumps(question_dict))
                question_dict["reason"] = reason

                self.goal_label_cache[similar_sent_id] = json.dumps(question_dict)
                demonstration_answer_list.append(question_dict)
                self._update_goal_label_cache(self.goal_label_cache)
            else:
                logger.info("该查询结果[在]缓存中")
                demonstration_answer_list.append(json.loads(self.goal_label_cache.get(similar_sent_id)))
        return demonstration_answer_list

    def final_ans(self, origin_sent: ReSentBaseData, demonstration_answer_list: List[dict]):
        answer_dict = {}
        answer_dict["Task Description"] = {}

        relation_list = self.entity_relation_rule_dict[origin_sent.head_entity.entity_type][origin_sent.tail_entity.entity_type] + ["NULL"] # type: List[str]
        task_description = self.PROMPT_TEMPLATE.format(relation_list_str=" , ".join(relation_list))
        answer_dict["Task Description"]["description"] = task_description
        answer_dict["Task Description"]["predefined_relations"] = relation_list

        answer_dict["ICL Demonstrations"] = []  # type: List[dict]

        for demonstration_answer in demonstration_answer_list:
            answer_dict["ICL Demonstrations"].append(demonstration_answer)

        input_dict = {}
        input_dict["sent"] = origin_sent.sent
        input_dict["head_entity_name"] = origin_sent.head_entity.entity_name
        input_dict["head_entity_type"] = origin_sent.head_entity.entity_type
        input_dict["tail_entity_name"] = origin_sent.tail_entity.entity_name
        input_dict["tail_entity_type"] = origin_sent.tail_entity.entity_type
        answer_dict["Test Input"] = input_dict

        return self.get_llm_answer(prompt=self.GET_RE_ANSWER_TEMPLATE, question=json.dumps(answer_dict))

    def do_llm_search(self, origin_sent: ReSentBaseData, similar_sent_list: List[IdentifiedReSentBaseData]):
        demonstration_answer_list = self.get_gold_label_induced(similar_sent_list)
        ans = self.final_ans(origin_sent, demonstration_answer_list)
        return ans
