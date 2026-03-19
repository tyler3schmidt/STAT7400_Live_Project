import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# -----------------------
# Load cached features
# -----------------------
X = np.load("features.npy")
y = np.load("labels.npy")

print(f"Loaded features: {X.shape}, labels: {y.shape}")

# -----------------------
# Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------
# Feature Scaling
# -----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Train SVM
# -----------------------
print("Training SVM...")
model = SVC(kernel="rbf", C=10, gamma="scale")
model.fit(X_train_scaled, y_train)

# -----------------------
# Evaluate
# -----------------------
pred = model.predict(X_test_scaled)
print("\nModel Performance")
print("------------------")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# -----------------------
# Save model
# -----------------------
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nSVM training complete. Saved: svm_model.pkl and scaler.pkl")