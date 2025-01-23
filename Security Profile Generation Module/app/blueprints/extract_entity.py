from flask import Blueprint, jsonify, request
from app.response_utils.BaseResponse import make_response
from app.response_utils.StatusCode import StatusCode
from backend_server.backend_cofig import AUTH_SECRET_KEY, AUTH_SECRET
from backend_server.entity_server_utils import add_cti_entity
from eneity_model.CtiEntity import CtiEntityModel
import json

entity_blueprint = Blueprint('entity', __name__, url_prefix='/entity')

entity_model = CtiEntityModel()


@entity_blueprint.route('/annotation', methods=['POST'])
def annotation_cti():
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

    # cti_id、content
    cti_sent_list = entity_model.get_cti_sent_list(content)
    result_word_list, result_label_list = entity_model.predict_entity(cti_sent_list)

    ret = {
        "ctiId": cti_id,
        "wordList": json.dumps(result_word_list),
        "labelList": json.dumps(result_label_list)
    }

    # 添加cti实体信息
    add_cti_entity(ret)
    # 复制chunk
    entity_model.process_text(result_word_list, result_label_list, cti_id)
    # 构图
    entity_model.create_graph(sent_list=result_word_list, label_list=result_label_list, cti_report_id=cti_id)

    return make_response(StatusCode.SUCCESS)
