from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Ensure Python can see preprocess_utils.py
sys.path.append(str(Path(__file__).resolve().parent))
from preprocess_utils import extract_features_from_image

# -----------------------
# Paths
# -----------------------
dataset_path = Path(__file__).parent / "STAT_7400_Image_Submission"
labels_csv = Path(__file__).parent / "fruit_labels_metadata.csv"
features_cache = Path("features.npy")
labels_cache = Path("labels.npy")

mapping = pd.read_csv(labels_csv)

# -----------------------
# Load images
# -----------------------
image_files = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.jpeg"))
print(f"Found {len(image_files)} images")

# -----------------------
# Extract features
# -----------------------
X, y = [], []
skipped_images = []
expected_length = None

for i, img_path in enumerate(image_files, 1):
    image_id = img_path.stem
    print(f"[{i}/{len(image_files)}] Processing {img_path.name}")

    # Lookup by image_id
    row = mapping.loc[mapping["image_id"] == image_id]
    if row.empty:
        skipped_images.append(img_path)
        print(f"Skipping {img_path} — no label in CSV")
        continue

    label = row["is_fruit"].iloc[0]  # Fruit vs vegetable

    try:
        features, _, _, _ = extract_features_from_image(img_path)

        # Ensure consistent feature vector length
        if expected_length is None:
            expected_length = len(features)
        elif len(features) != expected_length:
            skipped_images.append(img_path)
            print(f"Skipping {img_path} — inconsistent feature length")
            continue

        X.append(features)
        y.append(label)

    except Exception as e:
        skipped_images.append(img_path)
        print(f"Error processing {img_path}: {e}")
        continue

# Convert to numpy arrays and save
X = np.array(X)
y = np.array(y)
np.save(features_cache, X)
np.save(labels_cache, y)

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Skipped {len(skipped_images)} images")