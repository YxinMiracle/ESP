from flask import Flask
from  app.blueprints.extract_entity import entity_blueprint
from app.blueprints.extract_ttp import ttp_blueprint
from app.blueprints.construct_cti_rule import llm_rule_blueprint

def create_app():
    app = Flask(__name__)
    app.register_blueprint(entity_blueprint)
    app.register_blueprint(ttp_blueprint)
    app.register_blueprint(llm_rule_blueprint)
    return app
