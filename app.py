from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("logistic_model.pkl", "rb"))

@app.route("/")
def home():
    return "ML API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    X = np.array(data).reshape(1, -1)

    prediction = int(model.predict(X)[0])
    probability = model.predict_proba(X).tolist()

    return jsonify({
        "prediction": prediction,
        "probability": probability
    })

if __name__ == "__main__":
    app.run()
