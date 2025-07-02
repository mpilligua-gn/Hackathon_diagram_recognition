from flask import Flask, request, jsonify
from utils import infer_from_handwriting_points
import os

app = Flask(__name__)


@app.route("/process_diagram", methods=["POST"])
def process_diagram():
    data = request.get_json()
    data = request.get_json()
    handwriting = data.get("handwriting", [])
    # ensure the image exists
    if len(handwriting) == 0:
        return jsonify({"error": "empty handwriting"}), 401

    shapes, edges, texts = infer_from_handwriting_points(handwriting)
    if handwriting is None:
        return jsonify({"error": "No image provided"}), 400

    return jsonify({
        "shapes": shapes,
        "edges": edges,
        "texts": texts
    })


if __name__ == "__main__":
    app.run(port=5000)
