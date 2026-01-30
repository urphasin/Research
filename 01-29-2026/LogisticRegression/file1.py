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
print(Y)

#######################################################
print("Classes:", np.bincount(Y))   # ← check balance


# Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size= 0.25,
    random_state= 42,
    stratify= Y,
)
