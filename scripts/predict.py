import numpy as np
import pandas as pd

class Predictor:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return [1 if i > 0.5 else 0 for i in self._sigmoid(linear_model)]

if __name__ == "__main__":
    data = pd.read_csv('data/processed_data.csv').drop('is_churn', axis=1)
    X = data.values

    weights = np.load('scripts/model_weights.npy')
    bias = np.load('scripts/model_bias.npy')
    predictor = Predictor(weights, bias)
    predictions = predictor.predict(X)
    print("Predictions:", predictions)

