"""
这里主要用来存放entity Java后端相关接口
"""
import json

import requests
from backend_server.backend_cofig import BASE_HEADERS, ADD_CTI_ENTITY_URL, ADD_CTI_CHUNK_URL, UPDATE_CTI_CONTENT_URL, ITEM_SERVER_PATH_PREFIX, GET_ITEM_ID_URL, \
    ADD_CTI_TTP_URL, ADD_UNIQUE_ENTITY, IS_RELATION_TYPE_URL, GET_UNIQUE_ENTITY_ID, ADD_RELATION_URL
import time
import logging


def add_cti_entity(data):
    BASE_HEADERS["inner-req-timestamp"] = str(int(time.time() * 1000))
    response = requests.post(ADD_CTI_ENTITY_URL, headers=BASE_HEADERS, data=json.dumps(data))
    logging.log(logging.INFO, f"add_cti_entity response: {response}")


# 添加cti的chunk
def add_cti_chunk(cti_chunk_data):
    data = {
        "ctiChunkData": cti_chunk_data
    }
    BASE_HEADERS["inner-req-timestamp"] = str(int(time.time() * 1000))
    response = requests.post(ADD_CTI_CHUNK_URL, headers=BASE_HEADERS, data=json.dumps(data))
    logging.log(logging.INFO, f"add_cti_chunk response: {response}")


# 根据ctiId去更新cti的content，目的是为了在前端展示情报的实体图
def update_cti_content_by_cti_id(cti_id, cti_content):
    data = {
        "id": cti_id,
        "content": cti_content
    }
    BASE_HEADERS["inner-req-timestamp"] = str(int(time.time() * 1000))
    response = requests.post(UPDATE_CTI_CONTENT_URL, headers=BASE_HEADERS, data=json.dumps(data))
    logging.log(logging.INFO, f"update cti content response: {response}")


# 根据itemName获取itemId
def get_item_type_id(item_name):
    url = f"{GET_ITEM_ID_URL}/{item_name}"
    BASE_HEADERS["inner-req-timestamp"] = str(int(time.time() * 1000))
    response = requests.post(url=url, headers=BASE_HEADERS)
    res = response.json()
    return res["data"]


# 添加构图后的节点信息
def add_entity_data(entity_name, cti_id, item_id):
    data = {
        "entityName": entity_name,
        "ctiId": cti_id,
        "itemId": item_id
    }
    BASE_HEADERS["inner-req-timestamp"] = str(int(time.time() * 1000))
    response = requests.post(ADD_UNIQUE_ENTITY, headers=BASE_HEADERS, data=json.dumps(data))
    logging.log(logging.INFO, f"add entity data response: {response}")


def get_relation_type_id(start_item_id, end_item_id, relation_name):
    """
    获取对应载数据库中的关系类型，这个关系类型是stix规定好的
    :param startItemId:
    :param endItemId:
    :param relationName:
    :return:
    """
    url = f"{IS_RELATION_TYPE_URL}/{start_item_id}/{end_item_id}/{relation_name}"
    BASE_HEADERS["inner-req-timestamp"] = str(int(time.time() * 1000))
    response = requests.post(url=url, headers=BASE_HEADERS)
    return response.json()["data"]


# 获取父item的id
def get_father_item_id(item_name):
    """
    这个是获取这个类型的父类究竟是什么类型
    :param item_name:
    :return:
    """
    if "_" in item_name:
        item_name = item_name.split("_")[0]
    return get_item_type_id(item_name)


# 获取图上节点的id，对应entity表
def get_detail_entity_id(cti_id, sent_text, item_id):
    data = {
        "ctiId": cti_id,
        "entityName": sent_text,
        "itemId": item_id
    }
    BASE_HEADERS["inner-req-timestamp"] = str(int(time.time() * 1000))
    response = requests.post(url=GET_UNIQUE_ENTITY_ID, headers=BASE_HEADERS, data=json.dumps(data))
    res = response.json()
    return res["data"]


def add_final_relation_data(cti_id, start_detail_cti_chunk_id, end_detail_cti_chunk_id, relation_type_id):
    data = {
        "ctiId": cti_id,
        "startCtiEntityId": start_detail_cti_chunk_id,
        "endCtiEntityId": end_detail_cti_chunk_id,
        "relationTypeId": relation_type_id,
    }
    print(data)
    BASE_HEADERS["inner-req-timestamp"] = str(int(time.time() * 1000))
    response = requests.post(ADD_RELATION_URL, headers=BASE_HEADERS, data=json.dumps(data))
    logging.log(logging.INFO, f"add relation response: {response}")
