import time
from pathlib import Path
import sys
import random

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from preprocess_utils import extract_features_from_image
from best_predict import predict


image_dir = BASE_DIR / "New_pictures"

image_files = (
    list(image_dir.glob("*.jpg")) +
    list(image_dir.glob("*.jpeg")) +
    list(image_dir.glob("*.JPG")) +
    list(image_dir.glob("*.JPEG"))
)


start = time.time()

for img_path in image_files:
    features, _, _, _ = extract_features_from_image(img_path)
    label = predict(features)
    print(img_path.name, "->", label)

end = time.time()



print("\nTotal runtime:", end - start, "seconds")