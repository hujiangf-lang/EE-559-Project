import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from imblearn.over_sampling import SMOTE

# =========================
# 1. Load the dataset
# =========================
df = pd.read_csv("data/creditcard.csv")

print("Data shape:", df.shape)
print("Class distribution:\n", df["Class"].value_counts())

# =========================
# 2. Basic preprocessing
# =========================
# Remove the Time feature since it is not very useful here
df = df.drop("Time", axis=1)

X = df.drop("Class", axis=1)
y = df["Class"]

# =========================
# 3. Train / Validation / Test split
# =========================
# First split off test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Then split temp into train and validation
# Since temp is 80% of the data, validation is 0.25 of temp = 20% of total
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

print("\nSplit sizes:")
print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)

# =========================
# 4. Helper functions
# =========================
def find_best_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 on validation set."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def evaluate_model(name, model, X_val, y_val, X_test, y_test):
    """Tune threshold on validation set, then evaluate on test set."""
    # Validation probabilities
    y_val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, best_val_f1 = find_best_threshold(y_val, y_val_prob)

    # Test probabilities
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_test, y_test_prob)
    pr_auc = average_precision_score(y_test, y_test_prob)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print(f"\n=== {name} ===")
    print(f"Best threshold on validation set: {best_threshold:.2f}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print("\nTest classification report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test PR-AUC:  {pr_auc:.4f}")

    cm = confusion_matrix(y_test, y_test_pred)

    return {
        "name": name,
        "model": model,
        "threshold": best_threshold,
        "y_test_prob": y_test_prob,
        "y_test_pred": y_test_pred,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": cm,
    }

# =========================
# 5. Train three variants
# =========================
results = []

# -------- 1. Baseline --------
model_base = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model_base.fit(X_train, y_train)
results.append(evaluate_model("Baseline RF", model_base, X_val, y_val, X_test, y_test))

# -------- 2. Class Weight --------
model_weight = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model_weight.fit(X_train, y_train)
results.append(evaluate_model("Class Weight RF", model_weight, X_val, y_val, X_test, y_test))

# -------- 3. SMOTE --------
# Apply SMOTE only on the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model_smote = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model_smote.fit(X_train_smote, y_train_smote)
results.append(evaluate_model("SMOTE RF", model_smote, X_val, y_val, X_test, y_test))

# =========================
# 6. Summary table
# =========================
summary = pd.DataFrame([
    {
        "Model": r["name"],
        "Threshold": r["threshold"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "ROC-AUC": r["roc_auc"],
        "PR-AUC": r["pr_auc"],
    }
    for r in results
])

print("\n=== Summary ===")
print(summary.to_string(index=False))

# =========================
# 7. Plot Precision-Recall curves
# =========================
plt.figure(figsize=(8, 6))

for r in results:
    precision, recall, _ = precision_recall_curve(y_test, r["y_test_prob"])
    plt.plot(recall, precision, label=f'{r["name"]} (AP={r["pr_auc"]:.4f})')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Random Forest)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =========================
# 8. Plot ROC curves
# =========================
plt.figure(figsize=(8, 6))

for r in results:
    fpr, tpr, _ = roc_curve(y_test, r["y_test_prob"])
    plt.plot(fpr, tpr, label=f'{r["name"]} (AUC={r["roc_auc"]:.4f})')

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =========================
# 9. Plot confusion matrices
# =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, r in zip(axes, results):
    disp = ConfusionMatrixDisplay(confusion_matrix=r["cm"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(r["name"])

plt.tight_layout()
plt.show()

# =========================
# 10. Feature importance (best for RF report)
# =========================
feature_names = X.columns
importances = model_base.feature_importances_

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n=== Top 15 Feature Importances (Baseline RF) ===")
print(fi_df.head(15).to_string(index=False))

plt.figure(figsize=(10, 6))
top_k = fi_df.head(15).iloc[::-1]
plt.barh(top_k["Feature"], top_k["Importance"])
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances (Baseline RF)")
plt.grid(True, axis="x", alpha=0.3)
plt.show()