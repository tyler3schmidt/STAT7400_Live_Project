import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,  "super_learner_nnls_model.pkl")

_model_obj = None


def load_model():
    global _model_obj
    if _model_obj is None:
        _model_obj = joblib.load(MODEL_PATH)
    return _model_obj


def aligned_predict_proba(model, X, class_order):
    proba = model.predict_proba(X)
    model_classes = list(model.classes_)

    aligned = np.zeros((len(X), len(class_order)), dtype=float)
    for j, cls in enumerate(model_classes):
        idx = class_order.index(cls)
        aligned[:, idx] = proba[:, j]

    return aligned


def build_meta_features(base_models, X, class_order, base_model_names):
    blocks = []
    for name in base_model_names:
        model = base_models[name]
        proba = aligned_predict_proba(model, X, class_order)
        blocks.append(proba)
    return np.hstack(blocks)


def predict_labels_nnls(meta_X, weights_raw, class_order):
    scores = meta_X @ weights_raw.T
    pred_idx = np.argmax(scores, axis=1)
    return np.array([class_order[i] for i in pred_idx])


def predict(feature_vector):
    model_obj = load_model()

    feature_names = model_obj["feature_names"]
    class_order = model_obj["class_order"]
    base_models = model_obj["base_models"]
    base_model_names = model_obj["base_model_names"]
    nnls_weights_raw = model_obj["nnls_weights_raw"]

    x = np.asarray(feature_vector, dtype=float)

    if x.ndim == 1:
        if x.shape[0] != len(feature_names):
            raise ValueError(
                f"Expected feature vector of length {len(feature_names)}, got {x.shape[0]}."
            )
        x = x.reshape(1, -1)
    elif x.ndim == 2 and x.shape[0] == 1:
        if x.shape[1] != len(feature_names):
            raise ValueError(
                f"Expected feature vector of length {len(feature_names)}, got {x.shape[1]}."
            )
    else:
        raise ValueError("feature_vector must be a 1D vector or shape (1, n_features).")

    X_df = pd.DataFrame(x, columns=feature_names)
    meta_X = build_meta_features(base_models, X_df, class_order, base_model_names)
    pred = predict_labels_nnls(meta_X, nnls_weights_raw, class_order)[0]

    return str(pred)