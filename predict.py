from pathlib import Path
import joblib
import numpy as np

# -----------------------
# Load model and scaler once
# -----------------------
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "svm_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")


# -----------------------
# Required prediction API
# -----------------------
def predict(x):
    x = np.asarray(x).reshape(1, -1)
    x_scaled = scaler.transform(x)

    pred = model.predict(x_scaled)[0]

    return "fruit" if pred else "vegetable"