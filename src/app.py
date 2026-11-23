"""Local Flask frontend and upload inference API."""
import argparse
import io

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from src.data import DATA, ROOT
from src.model import load_model
from src.predict import predict_image

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 12_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def create_app():
    torch.set_num_threads(1)
    model, checkpoint = load_model()
    app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES + 65536

    @app.get("/")
    def homepage():
        return app.send_static_file("index.html")

    @app.get("/api/model")
    def model_info():
        return jsonify(
            name=checkpoint["name"], experiment=checkpoint["experiment"],
            epoch=checkpoint["epoch"], status="ready", max_bytes=MAX_BYTES,
        )

    @app.get("/demo/<path:filename>")
    def demo_image(filename):
        return send_from_directory(DATA / "demo", filename)

    @app.errorhandler(413)
    def oversized(_error):
        return jsonify(error="图片超过 5 MB，请缩小后重试。"), 413

    @app.post("/api/predict")
    def predict():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="请选择要识别的图片。"), 400
        payload = upload.stream.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            return oversized(None)
        try:
            with Image.open(io.BytesIO(payload)) as image:
                if image.format not in ALLOWED_FORMATS:
                    return jsonify(error="支持 JPG、PNG 和 WebP 图片。"), 415
                if image.width * image.height > MAX_PIXELS:
                    return jsonify(error="图片像素数超过 1200 万，请缩小后重试。"), 413
                image.load()
                result = predict_image(model, image)
        except Image.DecompressionBombError:
            return jsonify(error="图片像素数超过 1200 万，请缩小后重试。"), 413
        except (UnidentifiedImageError, OSError, ValueError):
            return jsonify(error="无法读取图片，请确认文件格式和内容。"), 400
        return jsonify(result)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    create_app().run(host="127.0.0.1", port=args.port)
