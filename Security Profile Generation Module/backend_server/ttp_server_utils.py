"""
这里主要用来存放ttp Java后端相关接口
"""
import json

import requests
from backend_server.backend_cofig import ADD_CTI_TTP_URL, BASE_HEADERS
import time
import logging


def add_cti_ttp(data):
    BASE_HEADERS["inner-req-timestamp"] = str(int(time.time() * 1000))
    response = requests.post(ADD_CTI_TTP_URL, headers=BASE_HEADERS, data=json.dumps(data))
    logging.log(logging.INFO, f"add_cti_ttp response: {response}")


if __name__ == '__main__':
    add_cti_ttp({
        "ctiId": 123123123123123,
        "sentLevelTtp": '{zxcsadfasdasdasfsdgsdfhdhdfhdfh}',
        "articleLevelTtp": '{zxcsadfasdasdasfsdgsdfhdhdfhdfh}'
    })
