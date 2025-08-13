from flask import jsonify
from werkzeug.exceptions import HTTPException

def register_error_handlers(app):
    @app.errorhandler(HTTPException)
    def handle_http(e: HTTPException):
        return jsonify({"status": "error", "code": e.code, "message": e.description}), e.code

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"status": "error", "code": 413, "message": "File too large"}), 413

    @app.errorhandler(Exception)
    def handle_generic(e):
        app.logger.exception("Unhandled exception")
        return jsonify({"status": "error", "code": 500, "message": "Internal server error"}), 500
