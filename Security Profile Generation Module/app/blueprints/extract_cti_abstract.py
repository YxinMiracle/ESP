from flask import Blueprint, jsonify

entity_blueprint = Blueprint('abstract', __name__, url_prefix='/abstract')

@entity_blueprint.route('/some_route', methods=['GET'])
def handle_some_route():
    return jsonify({'message': 'Response from entity route'})
