from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

pipeline = joblib.load(BASE_DIR / "best_catboost_pipeline.pkl")

with open(BASE_DIR / "feature_columns.json", "r") as f:
    feature_columns = json.load(f)


def predict(feature_vector):
    x = np.asarray(feature_vector, dtype=float).reshape(1, -1)

    if x.shape[1] != len(feature_columns):
        raise ValueError(
            f"Expected {len(feature_columns)} features, got {x.shape[1]}."
        )

    X = pd.DataFrame(x, columns=feature_columns)

    pred = pipeline.predict(X)
    pred = np.ravel(pred)[0]
    print(pipeline.classes_)
    print(pipeline.predict_proba(X))

    return pred