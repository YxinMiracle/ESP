from flask import Blueprint, jsonify, request
from app.response_utils.BaseResponse import make_response
from app.response_utils.StatusCode import StatusCode
from backend_server.backend_cofig import AUTH_SECRET_KEY, AUTH_SECRET
from ttp_model.CtiTtp import CtiTTpModel
import json
import pandas as pd
from backend_server.ttp_server_utils import add_cti_ttp

ttp_model = CtiTTpModel()

ttp_blueprint = Blueprint('ttp', __name__, url_prefix='/ttp')

@ttp_blueprint.route('/extract', methods=['POST'])
def handle_other_route():
    # 对data进行数据检查
    data = request.json
    if not isinstance(data, dict):
        return make_response(StatusCode.PARAMETER_ERROR)

    # 对cti_id进行数据检查
    cti_id = data.get('ctiId', -1)
    if cti_id == -1:
        return make_response(StatusCode.PARAMETER_ERROR)

    # 对header信息进行校验
    auth_secret = request.headers.get(AUTH_SECRET_KEY)
    if auth_secret != AUTH_SECRET:
        return make_response(StatusCode.NO_AUTH)

    # 开始进行抽取
    content = data.get("content", "")
    if content == '':
        return make_response(StatusCode.PARAMETER_ERROR)

    cti_te_res, sent_te_list = ttp_model.get_cti_ttp_result(content)

    ret = {
        "ctiId": cti_id,
        "sentLevelTtp": json.dumps(sent_te_list),
        "articleLevelTtp": json.dumps(cti_te_res)
    }

    # 将抽取出来的数据返回给后端进行处理
    add_cti_ttp(ret)
    return make_response(StatusCode.SUCCESS)
