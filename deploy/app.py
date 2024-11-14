from flask import Flask, request, jsonify
import numpy as np
from predict import Predictor

app = Flask(__name__)

# Load model weights
weights = np.load('scripts/model_weights.npy')
bias = np.load('scripts/model_bias.npy')
predictor = Predictor(weights, bias)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X = np.array(data['features']).reshape(1, -1)
    prediction = predictor.predict(X)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

