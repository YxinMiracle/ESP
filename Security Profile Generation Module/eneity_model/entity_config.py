import requests
import json

from backend_server.backend_cofig import ADD_RELATION_URL
from backend_server.entity_server_utils import add_cti_chunk

POS_LIST = ["[PPAD]", 'PROPN', 'ADJ', 'PUNCT', 'PRON', 'SCONJ', 'VERB', 'NOUN', 'PART', 'AUX', 'ADV', 'INTJ', 'X',
            'CCONJ', 'NUM', 'SYM', 'DET', 'ADP']

ENTITY_LABEL_LIST = ['B-malware_worm', 'B-malware_bot', 'I-infrastructure_hosting-malware', 'B-malware_ddos',
                     'I-vulnerability', 'B-malware_keylogger', 'B-infrastructure_attack', 'I-malware',
                     'I-infrastructure_exfiltration', 'I-windows-registry-key', 'B-infrastructure_reconnaissance',
                     'B-process', 'B-malware_screen-capture', 'B-ipv6-addr', 'B-domain-name', 'I-url', 'I-process',
                     'B-file-hash', 'B-user-account', 'B-email-addr', 'I-domain-name', 'B-windows-registry-key',
                     'B-http-request-ext', 'B-infrastructure_hosting-malware', 'B-infrastructure_botnet',
                     'I-identity_victim',
                     'B-url', 'B-infrastructure', 'I-threat-actor', 'B-malware_virus', 'B-directory', 'I-malware_virus',
                     'I-directory', 'I-http-request-ext', 'I-malware_exploit-kit', 'B-mutex', 'I-file-name',
                     'I-file-hash',
                     'B-malware', 'B-attack-pattern', 'B-file-name', 'B-malware_remote-access-trojan', 'I-tool',
                     'B-infrastructure_command-and-control', 'I-infrastructure_command-and-control', 'B-campaign',
                     'B-ipv4-addr', 'I-campaign', 'I-ipv4-addr', 'O', 'B-intrusion-set', 'I-location', 'B-threat-actor',
                     'I-user-account', 'I-malware_bot', 'B-location', 'I-malware_ransomware',
                     'B-malware_resource-exploitation', 'B-malware_ransomware', 'I-infrastructure', 'I-intrusion-set',
                     'I-software', 'B-malware_exploit-kit', 'I-malware_resource-exploitation', 'B-identity',
                     'B-identity_victim', 'I-malware_keylogger', 'B-malware_webshell', 'I-email-addr',
                     'I-malware_remote-access-trojan', 'I-infrastructure_attack', 'B-infrastructure_exfiltration',
                     'B-infrastructure_victim', 'B-software', 'I-identity', 'I-malware_worm', 'B-vulnerability',
                     'B-tool',
                     'I-infrastructure_victim']

BERT_MODEL_NAME = "bert-base-cased"

VALID_WORD_POSITION = 520

NER_MODEL_PATH = "/home/cyx/open_msan_ner/src/model_dir/huawei/yxinmiraclefinalmodel_pos/sub_k_7_2/model.pkl"
NER_MODEL_OPTIMIZER_PATH = "/home/cyx/open_msan_ner/src/model_dir/huawei/yxinmiraclefinalmodel_pos/sub_k_7_2/optimizer.pkl"

GRAPH_RELATION_CONFIG = {"tool|---|location": ["targets"], "malware|---|domain-name": ["uses", "communicates-with"],
                         "intrusion-set|---|software": ["uses"],
                         "threat-actor|---|ipv4-addr": ["communicates-with", "uses"],
                         "malware|---|infrastructure": ["communicates-with", "compromises", "hosts", "targets", "uses",
                                                        "beacons-to"], "identity|---|identity": ["alias-of"],
                         "infrastructure|---|process": ["consists-of"], "tool|---|tool": ["alias-of", "uses"],
                         "malware|---|vulnerability": ["uses", "exploits"],
                         "file-name|---|malware": ["drops", "alias-of", "hashes-to"],
                         "infrastructure|---|user-account": ["consists-of"], "campaign|---|malware": ["uses"],
                         "intrusion-set|---|infrastructure": ["targets", "uses", "owns", "compromises"],
                         "file-name|---|file-name": ["alias-of"], "identity|---|vulnerability": ["has"],
                         "location|---|campaign": ["originates-from"],
                         "malware|---|tool": ["targets", "uses", "drops", "downloads"],
                         "campaign|---|campaign": ["alias-of"], "ipv4-addr|---|location": ["located-at"],
                         "intrusion-set|---|vulnerability": ["uses", "targets"],
                         "infrastructure|---|url": ["consists-of"],
                         "infrastructure|---|domain-name": ["hosts", "communicates-with", "consists-of"],
                         "identity|---|threat-actor": ["attributed-to"], "ipv4-addr|---|malware": ["uses"],
                         "ipv4-addr|---|domain-name": ["resolves-to", "hosts"],
                         "intrusion-set|---|identity": ["targets"],
                         "campaign|---|location": ["targets", "originates-from"],
                         "intrusion-set|---|threat-actor": ["alias-of", "attributed-to"],
                         "threat-actor|---|software": ["uses", "targets", "exploits", "drops", "downloads"],
                         "malware|---|identity": ["located-at", "targets"],
                         "malware|---|intrusion-set": ["attributed-to", "authored-by"], "tool|---|file-hash": ["drops"],
                         "malware|---|threat-actor": ["uses", "attributed-to", "authored-by"],
                         "malware|---|malware": ["alias-of", "located-at", "delivers", "attributed-to", "exploits",
                                                 "authored-by", "variant-of", "uses", "downloads", "drops"],
                         "threat-actor|---|threat-actor": ["alias-of", "uses"], "ipv4-addr|---|tool": ["uses"],
                         "infrastructure|---|software": ["consists-of", "alias-of"],
                         "file-hash|---|file-name": ["hashes-to"], "malware|---|file-hash": ["drops", "downloads"],
                         "intrusion-set|---|malware": ["authored-by", "owns", "uses"],
                         "infrastructure|---|file-name": ["downloads", "hosts", "drops", "consists-of"],
                         "file-hash|---|malware": ["alias-of"], "location|---|location": ["alias-of", "located-at"],
                         "threat-actor|---|malware": ["authored-by", "targets", "delivers", "uses"],
                         "malware|---|campaign": ["authored-by"], "file-name|---|infrastructure": ["uses"],
                         "infrastructure|---|ipv4-addr": ["hosts", "resolves-to", "consists-of"],
                         "domain-name|---|malware": ["hosts"], "threat-actor|---|file-name": ["uses"],
                         "threat-actor|---|tool": ["authored-by", "uses"],
                         "tool|---|infrastructure": ["uses", "authored-by", "targets"],
                         "intrusion-set|---|tool": ["uses"], "campaign|---|threat-actor": ["attributed-to"],
                         "domain-name|---|file-hash": ["hosts"], "software|---|software": ["alias-of", "targets"],
                         "threat-actor|---|intrusion-set": ["attributed-to"], "campaign|---|vulnerability": ["targets"],
                         "tool|---|vulnerability": ["targets", "has"], "campaign|---|tool": ["uses"],
                         "infrastructure|---|email-addr": ["consists-of"], "software|---|malware": ["has"],
                         "location|---|ipv4-addr": ["owns"],
                         "tool|---|malware": ["targets", "downloads", "uses", "delivers", "drops"],
                         "infrastructure|---|tool": ["uses", "communicates-with", "hosts"],
                         "software|---|vulnerability": ["exploits"], "malware|---|windows-registry-key": ["uses"],
                         "threat-actor|---|location": ["located-at", "targets"],
                         "campaign|---|infrastructure": ["uses", "compromises"], "software|---|identity": ["hosts"],
                         "intrusion-set|---|location": ["located-at", "targets", "originates-from"],
                         "file-name|---|file-hash": ["hashes-to", "drops"],
                         "identity|---|infrastructure": ["alias-of", "owns"],
                         "malware|---|location": ["located-at", "targets", "originates-from"],
                         "campaign|---|identity": ["targets"], "location|---|identity": ["located-at"],
                         "infrastructure|---|identity": ["has"], "infrastructure|---|location": ["located-at"],
                         "threat-actor|---|infrastructure": ["targets", "exploits", "owns", "uses", "compromises"],
                         "infrastructure|---|vulnerability": ["targets", "has"],
                         "campaign|---|intrusion-set": ["attributed-to"], "location|---|threat-actor": ["located-at"],
                         "infrastructure|---|malware": ["uses", "drops", "downloads", "delivers", "hosts"],
                         "malware|---|ipv4-addr": ["uses", "communicates-with"],
                         "ipv4-addr|---|identity": ["attributed-to"], "vulnerability|---|identity": ["has"],
                         "identity|---|malware": ["delivers"], "infrastructure|---|file-hash": ["consists-of"],
                         "intrusion-set|---|intrusion-set": ["alias-of"],
                         "malware|---|file-name": ["drops", "downloads", "uses"],
                         "infrastructure|---|infrastructure": ["alias-of", "targets", "uses", "controls",
                                                               "communicates-with"], "tool|---|file-name": ["drops"],
                         "tool|---|ipv4-addr": ["uses"],
                         "threat-actor|---|identity": ["uses", "originates-from", "alias-of", "attributed-to",
                                                       "targets"], "malware|---|url": ["communicates-with"],
                         "identity|---|location": ["originates-from", "located-at"],
                         "threat-actor|---|vulnerability": ["exploits", "targets"],
                         "software|---|infrastructure": ["alias-of"], "malware|---|software": ["uses", "targets"],
                         "domain-name|---|ipv4-addr": ["resolves-to", "consists-of"],
                         "threat-actor|---|campaign": ["uses"],
                         "threat-actor|---|domain-name": ["communicates-with", "uses"],
                         "campaign|---|domain-name": ["uses"], "software|---|file-name": ["has", "owns"]}

def get_relation_type_id(startItemId, endItemId, relationName):
    """
    获取对应载数据库中的关系类型，这个关系类型是stix规定好的
    :param startItemId:
    :param endItemId:
    :param relationName:
    :return:
    """
    url = f"http://172.29.89.32:8121/api/cti/isRelation/{startItemId}/{endItemId}/{relationName}"
    response = requests.get(url=url)
    return response.json()["data"]


def get_father_item_id(item_name):
    """
    这个是获取这个类型的父类究竟是什么类型
    :param item_name:
    :return:
    """
    if "_" in item_name:
        item_name = item_name.split("_")[0]
    url = f"http://172.29.89.32:8121/api/cti/get/itemId/{item_name}"
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.get(url=url, headers=headers)
    return response.json()["data"]


def add_entity_data(entityName, ctiId, itemId):
    url = f"http://172.29.89.32:8121/api/cti/add/unique/entity"
    data = {
        "entityName": entityName,
        "ctiId": ctiId,
        "itemId": itemId
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(response.json())
    return response.json()["data"]


def get_item_type_id(item_name):
    url = f"http://172.29.89.32:8121/api/cti/get/itemId/{item_name}"
    response = requests.get(url=url)
    return response.json()["data"]


def get_detail_entity_id(ctiId, sentText, itemId):
    url = f"http://172.29.89.32:8121/api/cti/unique/entityId/{ctiId}/{sentText}/{itemId}"
    response = requests.get(url=url)
    return response.json()["data"]


def add_final_relation_data(ctiId, startDetailCtiChunkId, endDetailCtiChunkId, relationTypeId):
    data = {
        "ctiId": ctiId,
        "startCtiEntityId": startDetailCtiChunkId,
        "endCtiEntityId": endDetailCtiChunkId,
        "relationTypeId": relationTypeId,
    }

    response = requests.post(url=ADD_RELATION_URL, headers=headers, data=json.dumps(data))
    return response.json()["data"]




