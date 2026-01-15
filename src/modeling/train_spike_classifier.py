import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.modeling.features import FEATURES
from src.modeling.split import temporal_split


def recall_at_top_k(y_true: pd.Series, scores: np.ndarray, frac: float = 0.05) -> float:
    """
    Business metric: among the top X% highest-risk predictions, what fraction are actually spikes?
    This is equivalent to 'precision in the top-risk segment' (often used operationally).
    Note: We'll also compute 'spike capture rate' below (true positive capture).
    """
    n = len(y_true)
    k = max(1, int(frac * n))
    top_idx = np.argsort(scores)[-k:]
    return float(y_true.iloc[top_idx].mean())


def capture_rate_at_top_k(y_true: pd.Series, scores: np.ndarray, frac: float = 0.05) -> float:
    """
    Of all actual spikes, what percent are contained in the top X% risk bucket?
    This is the most operationally meaningful KPI for targeted outreach.
    """
    n = len(y_true)
    k = max(1, int(frac * n))
    top_idx = np.argsort(scores)[-k:]
    spikes_total = y_true.sum()
    if spikes_total == 0:
        return 0.0
    spikes_in_top = y_true.iloc[top_idx].sum()
    return float(spikes_in_top / spikes_total)


# Load data
df = pd.read_csv("data/synthetic/member_month_with_volatility.csv")

# Ensure label exists
df = df.dropna(subset=["spike_flag_member"])

# Split by time (out-of-time validation)
train, test, cutoff_ym = temporal_split(df)
print("Temporal split cutoff ym:", cutoff_ym)
print("Train rows:", len(train), "Test rows:", len(test))

# Use leakage-safe features only (same FEATURES file you edited)
X_train = train[FEATURES]
y_train = train["spike_flag_member"].astype(int)

X_test = test[FEATURES]
y_test = test["spike_flag_member"].astype(int)

# Pipeline = preprocessing + model (production pattern)
clf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LogisticRegression(max_iter=2000)),
])

clf.fit(X_train, y_train)

# Probabilities for ranking members by risk
probs = clf.predict_proba(X_test)[:, 1]

# Metrics
auc = roc_auc_score(y_test, probs)

base_rate = y_test.mean()
precision_top5 = recall_at_top_k(y_test, probs, frac=0.05)
capture_top5 = capture_rate_at_top_k(y_test, probs, frac=0.05)

print("\nSpike Classification Results")
print("----------------------------")
print(f"Base spike rate (test): {base_rate:.4f}")
print(f"ROC-AUC: {auc:.3f}")
print(f"Top-5% bucket spike rate (precision@5%): {precision_top5:.4f}")
print(f"Spike capture rate in top-5% (recall@5%): {capture_top5:.4f}")
