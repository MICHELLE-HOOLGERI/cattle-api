from flask import Flask, request, jsonify
from cattle_model import CattleModel


app = Flask(__name__)
model = CattleModel()

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = "temp.jpg"
    file.save(file_path)

    pred, conf = model.predict(file_path)

    return jsonify({
        "model": "cattle",
        "prediction": pred,
        "confidence": conf
    })

@app.route("/")
def home():
    return jsonify({"message": "Cattle Model API is running"})

if __name__ == "__main__":
    app.run(debug=True)

