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

    @app.teardown_appcontext
    def _close_db(exception):
        db.close_db()

    with app.app_context():
        db.init_db()

    @app.cli.command("list-docs")
    def list_docs():
        db.print_all_documents()
    
    @app.cli.command("clear-db")
    def clear_db():
        db.clear_database()

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8080)