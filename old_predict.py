from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

# Ensure Python can see preprocess_utils.py
sys.path.append(str(Path(__file__).resolve().parent))
from preprocess_utils import extract_features_from_image

# -----------------------
# Load model and scaler
# -----------------------
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------
# Predict function
# -----------------------
def predict_image(image_path):
    features, _, _, _ = extract_features_from_image(image_path)
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    return "fruit" if pred else "vegetable"

# -----------------------
# Main execution
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict fruit vs vegetable for a folder of images.")
    parser.add_argument("input_dir", type=str, help="Path to folder containing images")
    parser.add_argument("output_csv", type=str, help="Path to output CSV file")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    # Collect all images recursively
    image_files = list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.jpeg"))
    print(f"Found {len(image_files)} images.")

    predictions = []

    for img_path in image_files:
        try:
            pred_label = predict_image(img_path)
            predictions.append({"image_id": img_path.name, "prediction": pred_label})
        except Exception as e:
            print(f"Skipping {img_path.name} due to error: {e}")

    # Write CSV
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")