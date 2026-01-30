# ────────────────────────────────────────────────
# Typical modern workflow (2025–2026 style)
# ────────────────────────────────────────────────

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# ─── Example: Breast cancer dataset ───────────────────────────────
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data["data"]
Y = data["target"]

print(X, end="\n\n")
print(Y, end="\n\n")

#######################################################
print("Classes:", np.bincount(Y))   # ← check balance


# Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size= 0.25,
    random_state= 42,
    stratify= Y,                            # important for imbalanced data
)

# Scale(almost always needed for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model ─ most important parameters in 2025+
model = LogisticRegression(
    penalty= 'l2',              # 'l1', 'l2', 'elasticnet', None
    C= 1.0,                     # Inverse of regularization strength (smaller = stronger reg)
    solver= 'lbfgs',            # 'lbfgs' (default), 'liblinear', 'saga', 'newton-cg'...
    max_iter= 1000,             
    random_state= 42, 
    class_weight= None          # 'balanced' if classes are imbalanced

)



