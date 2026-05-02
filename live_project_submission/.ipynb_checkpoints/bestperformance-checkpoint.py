import numpy as np
import pandas as pd
import joblib
import os

current_dir = os.path.dirname(__file__)

with open(os.path.join(current_dir, "./Models/best_model.pkl"), "rb") as f:
    model = joblib.load(f)


def bestmodel_predict(feature_vector):

    if isinstance(feature_vector, np.ndarray) and feature_vector.dtype == object:
        try:
            feature_vector = feature_vector.item()
        except:
            pass

    if isinstance(feature_vector, dict):
        X = pd.DataFrame([feature_vector])

    else:
        arr = np.asarray(feature_vector)

        # single sample
        if arr.ndim == 1:
            X = pd.DataFrame([arr], columns=[f"feat_{i}" for i in range(arr.shape[0])])

        # batch input
        elif arr.ndim == 2:
            X = pd.DataFrame(arr, columns=[f"feat_{i}" for i in range(arr.shape[1])])

        else:
            raise ValueError(f"Unsupported input shape: {arr.shape}")

    # Manage variables
    X = X[[col for col in X.columns if "feat" in col]]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = X.reindex(
        sorted(X.columns, key=lambda x: int(x.split("_")[1])),
        axis=1
    )

    print("X shape before prediction:", X.shape)

    # Predict
    pred = model.predict(X)

    labels = np.where(pred == 1, "fruit", "vegetable")

    return labels

# Example
# data = np.load(r"./Data/my_apple.npy", allow_pickle=True)
# data = np.load(r"./Data/X_test.npy", allow_pickle=True)
data = np.load(r"./Data/New_test.npy", allow_pickle=True)
apple_pred = bestmodel_predict(data)
print(apple_pred)