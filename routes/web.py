from flask import Blueprint, render_template

web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    return render_template('index.html')

@web_bp.route('/upload')
def upload():
    return render_template('upload.html')

@web_bp.route('/manifest-status')
def manifest_status():
    return render_template('manifest.html')

@web_bp.route('/chunk-status')
def chunk_status():
    return render_template('chunks.html')

@web_bp.route('/db-status')
def db_status():
    return render_template('dbtables.html')
