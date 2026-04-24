import time
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from preprocess_utils import extract_features_from_image
from master_predict import predict


image_dir = BASE_DIR / "calibration_images"
image_files = list(image_dir.glob("*.jpg"))

start = time.time()

for img_path in image_files:
    features, _, _, _ = extract_features_from_image(img_path)
    label = predict(features)
    print(img_path.name, "->", label)

end = time.time()



print("\nTotal runtime:", end - start, "seconds")