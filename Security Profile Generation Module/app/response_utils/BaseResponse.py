from flask import jsonify


def make_response(status_code, data: str = None):
    response = {
        "code": status_code.code,
        "msg": status_code.message,
        "data": data
    }
    return jsonify(response)


def make_response_have_msg(status_code, msg, data=None):
    response = {
        "code": status_code.code,
        "msg": msg,
        "data": data
    }
    return jsonify(response)
