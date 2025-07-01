from flask import Flask, request, jsonify
from utils import infer_flowmind2digital
import os

app = Flask(__name__)

@app.route("/process_diagram", methods=["POST"])
def process_diagram():
    data = request.get_json()
    image = data.get("image", None)

    # ensure the image exists
    if image is None or not os.path.exists(image):
        return jsonify({"error": "Image file does not exist"}), 401

    shapes, edges = infer_flowmind2digital(image)

    if image is None:
        return jsonify({"error": "No image provided"}), 400
    
    return jsonify({
        "shapes": shapes,
        "edges": edges
    })

if __name__ == "__main__":
    app.run(port=5000)