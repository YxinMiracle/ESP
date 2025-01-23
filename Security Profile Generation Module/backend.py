# from flask import Flask, request, jsonify
# from config import RET, create_graph, process_text
# import spacy
# from model_utils import get_cti_sent_list, predict_entity, test_ner
#
# app = Flask(__name__)
#
#
# @app.route('/annotation/cti', methods=['POST'])
# def receive_json():
#     cti_data = request.get_json()
#
#     cti_title = cti_data.get('title', '')
#     cti_content = cti_data.get('content', '')
#     cti_id = cti_data.get('id', '')
#     cti_sent_list = get_cti_sent_list(cti_content)
#     result_word_list, result_label_list = predict_entity(cti_sent_list)
#     return jsonify({"wordList": result_word_list, "labelList": result_label_list, "id": cti_id})
#
#
# @app.route('/graph/cti', methods=['POST'])
# def ret_graph():
#     cti_data = request.get_json()
#     sent_list = cti_data.get('wordList', '')
#     label_list = cti_data.get('labelList', '')
#     cti_id = cti_data.get('id', '')
#     create_graph(sent_list, label_list, cti_id)
#     return "添加成功"
#
#
# @app.route('/copy', methods=['POST'])
# def copy_model_data():
#     cti_data = request.get_json()
#     sent_list = cti_data.get('wordList', '')
#     label_list = cti_data.get('labelList', '')
#     cti_id = cti_data.get('id', '')
#     process_text(sent_list, label_list, cti_id)
#     return "添加成功"
#
#
# @app.route('/test/content', methods=['POST'])
# def test_ner_model():
#     cti_data = request.get_json()
#     test_content = cti_data.get('nerTestData', '')
#     print(test_content)
#     model_res = test_ner(test_content)
#     return jsonify(model_res)
#
#
# @app.route('/retrain', methods=['POST'])
# def test_ner_modessl():
#     cti_data = request.get_json()
#     test_content = cti_data.get('trainData', '')
#     print(test_content)
#     # todo retrain
#
#
# if __name__ == '__main__':
#     app.run("0.0.0.0", port=4499, debug=True)
