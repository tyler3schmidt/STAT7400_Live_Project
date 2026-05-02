from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Relative paths
PROJECT_ROOT = Path(".")
NEW_IMG_DIR = PROJECT_ROOT / "Test_Images"
SCRIPTS_DIR = PROJECT_ROOT / "Utility"
OUTPUT_PATH = PROJECT_ROOT / "Data/New_test.npy"

# Import feature extraction function
sys.path.append(str(SCRIPTS_DIR))
from preprocess_utils import extract_features_from_image

# Image extensions
image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

records = []

image_files = sorted([f for f in NEW_IMG_DIR.iterdir() if f.suffix in image_exts])

for img_path in image_files:
    try:
        feature_vector, best_mask, best_score, best_params = extract_features_from_image(img_path)

        feature_vector = np.asarray(feature_vector).ravel()
        records.append(feature_vector)

        print(f"Done: {img_path.name}")

    except Exception as e:
        print(f"Failed: {img_path.name} --> {e}")

# Save as numpy array
X_new = np.vstack(records)

np.save(OUTPUT_PATH, X_new)

print("\nFinished.")
print("Saved to:", OUTPUT_PATH)
print("Shape:", X_new.shape)