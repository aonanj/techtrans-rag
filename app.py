from flask import Flask
from flask_cors import CORS
from infrastructure.logger import setup_logger
from routes.web import web_bp
from routes.api import api_bp
import infrastructure.database as db


setup_logger()

def create_app():
    app = Flask(__name__, template_folder='static')
    CORS(app)
    app.config.from_object('config.Config')

    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)

    with app.app_context():
        db.init_db()

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8080)