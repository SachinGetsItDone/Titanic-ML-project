from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "API is running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]

        # frontend order: [sex, pclass, age, fare]
        sex, pclass, age, fare = features

        # model training order: [pclass, sex, age, fare]
        X = np.array([[pclass, sex, age, fare]], dtype=float)

        prediction = int(model.predict(X)[0])
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()

