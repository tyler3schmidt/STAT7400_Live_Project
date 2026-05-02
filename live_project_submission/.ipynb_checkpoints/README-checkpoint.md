Here’s a **short, clean README.md**:

````markdown
# Fruit vs Vegetable Classification

This project classifies images as **fruit** or **vegetable** using pretrained models.

---

## Installation

```bash
pip install -r requirements.txt
pip install torch
````

---

## Optional Preprocessing

If you have new images in `./New_pictures`, you can generate features by:

```bash
python make_new_test.py
```

This step relies on `extract_features_from_image` from `preprocess_utils.py` (from previous live project homework) and will generate:

```
New_test.npy
```

---

## Prediction

After generating `New_test.npy`, run:

```bash
python bestperformance.py
```

or

```bash
python meta_model.py
```

to obtain predictions.
