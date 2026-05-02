import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_catboost_pipeline.pkl")

_model_obj = None


def load_model():
    global _model_obj
    if _model_obj is None:
        _model_obj = joblib.load(MODEL_PATH)
    return _model_obj


def predict(feature_vector):
    model_obj = load_model()

    pipeline = model_obj["pipeline"]
    feature_columns = model_obj["feature_columns"]

    x = np.asarray(feature_vector, dtype=float)

    if x.ndim == 1:
        if x.shape[0] != len(feature_columns):
            raise ValueError(
                f"Expected feature vector of length {len(feature_columns)}, got {x.shape[0]}."
            )
        x = x.reshape(1, -1)
    elif x.ndim == 2 and x.shape[0] == 1:
        if x.shape[1] != len(feature_columns):
            raise ValueError(
                f"Expected feature vector of length {len(feature_columns)}, got {x.shape[1]}."
            )
    else:
        raise ValueError("feature_vector must be a 1D vector or shape (1, n_features).")

    X_df = pd.DataFrame(x, columns=feature_columns)

    pred = pipeline.predict(X_df)
    pred = int(np.ravel(pred)[0])

    if pred == 1:
        return "fruit"
    elif pred == 0:
        return "vegetable"
    else:
        raise ValueError(f"Unexpected predicted class: {pred}")