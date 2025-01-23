from flask import Blueprint, jsonify, request, Response
from app.response_utils.BaseResponse import make_response
from app.response_utils.StatusCode import StatusCode
from backend_server.backend_cofig import AUTH_SECRET_KEY, AUTH_SECRET
from rule_model.LlmRuleModel import LLM_RULE_MODEL
from uuid import uuid4
import json

llm_rule_blueprint = Blueprint('rule', __name__, url_prefix='/rule')

rule_llm = LLM_RULE_MODEL()


@llm_rule_blueprint.route('/create/yara', methods=['POST'])
def create_cti_yara_rule():
    # 对data进行数据检查
    data = request.json
    if not isinstance(data, dict):
        return make_response(StatusCode.PARAMETER_ERROR)

    # 对header信息进行校验
    auth_secret = request.headers.get(AUTH_SECRET_KEY)
    if auth_secret != AUTH_SECRET:
        return make_response(StatusCode.NO_AUTH)

    relation_graph = data.get("relationGraph", "")
    ioc_data = data.get("iocData", "")

    question = rule_llm.YARA_QUESTION_TEMPLATE.format(relation_graph=relation_graph, ioc_data=ioc_data)

    ans = rule_llm.get_llm_answer(rule_llm.SYSTEM_YARA_PROMPT, question)

    return make_response(StatusCode.SUCCESS, ans)


@llm_rule_blueprint.route('/create/snort', methods=['POST'])
def create_cti_snort_rule():
    # 对data进行数据检查
    data = request.json
    if not isinstance(data, dict):
        return make_response(StatusCode.PARAMETER_ERROR)

    # 对header信息进行校验
    auth_secret = request.headers.get(AUTH_SECRET_KEY)
    if auth_secret != AUTH_SECRET:
        return make_response(StatusCode.NO_AUTH)

    ioc_data = data.get("iocData", "")
    if ioc_data == "":
        return make_response(StatusCode.PARAMETER_ERROR)

    ans = rule_llm.get_llm_answer(rule_llm.SYSTEM_SNORT_PROMPT, ioc_data)
    return make_response(StatusCode.SUCCESS, ans)


@llm_rule_blueprint.route('/abstract', methods=['POST'])
def get_cti_abstract():
    # 对data进行数据检查
    data = request.json
    if not isinstance(data, dict):
        return make_response(StatusCode.PARAMETER_ERROR)

    # 对header信息进行校验
    auth_secret = request.headers.get(AUTH_SECRET_KEY)
    if auth_secret != AUTH_SECRET:
        return make_response(StatusCode.NO_AUTH)

    cti_content = data.get("content", "")
    if cti_content == "":
        return make_response(StatusCode.PARAMETER_ERROR)

    ans = rule_llm.get_llm_answer(rule_llm.ABSTRACT_SYSTEM_PROMPT, cti_content)
    return make_response(StatusCode.SUCCESS, ans)


@llm_rule_blueprint.route('/chat', methods=['GET'])
def llm_chat():
    # 对header信息进行校验
    auth_secret = request.headers.get(AUTH_SECRET_KEY)
    if auth_secret != AUTH_SECRET:
        return make_response(StatusCode.NO_AUTH)

    system_prompts = request.args.get('systemPrompts', default='', type=str)
    user_question = request.args.get('userQuestion', default='', type=str)
    if system_prompts == "" and user_question == "":
        return make_response(StatusCode.PARAMETER_ERROR)

    def generate():
        try:
            streamer = rule_llm.get_ans_streamer(system_prompts,
                                                 user_question)
            uuid_count = uuid4().hex
            completed_text = ""
            for generate_text in streamer:
                completed_text += generate_text
                ret_data = {"msg": "process_generating", "event_id": uuid_count, "output": generate_text}
                yield f'data: {json.dumps(ret_data)}\n\n'
            yield f'data: {json.dumps({"msg": "process_completed", "event_id": uuid_count, "data": completed_text})}\n\n'
        except Exception as e:
            yield f'data: {json.dumps({"msg": "error", "details": str(e)})}\n\n'

    return Response(generate(), mimetype='text/event-stream')
