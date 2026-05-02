import numpy as np
import pandas as pd
import joblib
import os

current_dir = os.path.dirname(__file__)

with open(os.path.join(current_dir, "./Models/cat.pkl"), "rb") as f:
    cat = joblib.load(f)

with open(os.path.join(current_dir, "./Models/tabpfn.pkl"), "rb") as f:
    tabpfn = joblib.load(f)

with open(os.path.join(current_dir, "./Models/scaler.pkl"), "rb") as f:
    scaler = joblib.load(f)

with open(os.path.join(current_dir, "./Models/weights.pkl"), "rb") as f:
    weights = joblib.load(f)

def predict(feature_vector):

    if isinstance(feature_vector, np.ndarray) and feature_vector.dtype == object:
        try:
            feature_vector = feature_vector.item()
        except:
            pass

    # single sample
    if isinstance(feature_vector, dict):
        X = pd.DataFrame([feature_vector])

    # numpy array input
    else:
        arr = np.asarray(feature_vector)

        # single sample
        if arr.ndim == 1:
            X = pd.DataFrame([arr], columns=[f"feat_{i}" for i in range(arr.shape[0])])

        # batch samples
        elif arr.ndim == 2:
            X = pd.DataFrame(arr, columns=[f"feat_{i}" for i in range(arr.shape[1])])

        else:
            raise ValueError(f"Unsupported input shape: {arr.shape}")

    # Keep feat columns
    X = X[[col for col in X.columns if "feat" in col]]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = X.reindex(
        sorted(X.columns, key=lambda x: int(x.split("_")[1])),
        axis=1
    )

    # Check feature number
    print("X shape before scaler:", X.shape)

    X_scaled = scaler.transform(X.values)

    probs = np.column_stack([
        cat.predict_proba(X)[:, 1],
        tabpfn.predict_proba(X_scaled)[:, 1]
    ])

    final = (probs @ weights >= 0.5).astype(int)

    labels = np.where(final == 1, "fruit", "vegetable")

    return labels

# Example
# data = np.load(r"./Data/my_apple.npy", allow_pickle=True)
data = np.load(r"./Data/New_test.npy", allow_pickle=True)

apple_pred = predict(data)
print(apple_pred)
