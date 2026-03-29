from src.model import build_model
import numpy as np

def train_model(Sxx, target):
    X = Sxx.reshape(1, Sxx.shape[0], Sxx.shape[1], 1)
    y = np.array([target])

    model = build_model((X.shape[1], X.shape[2], 1))
    model.fit(X, y, epochs=10)

    pred = model.predict(X)
    return pred