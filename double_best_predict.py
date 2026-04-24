from pathlib import Path
import joblib
import numpy as np

# -----------------------
# Load model and scaler once
# -----------------------
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "catboost_combined_final_model.pkl")



# -----------------------
# Required prediction API
# -----------------------
def predict(x):
    x = np.asarray(x).reshape(1, -1)

    pred = model.predict(x)[0]

    return "fruit" if pred else "vegetable"