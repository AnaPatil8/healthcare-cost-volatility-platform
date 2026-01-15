import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

from src.modeling.features import FEATURES
from src.modeling.split import temporal_split


def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred)) ** 0.5


def make_tiers(series: pd.Series, q=(0.8, 0.95)):
    """
    Create tiers based on quantiles:
    - low: bottom 80%
    - med: 80–95%
    - high: top 5%
    """
    s = series.copy()
    p80 = s.quantile(q[0])
    p95 = s.quantile(q[1])

    tier = pd.Series(index=s.index, dtype="object")
    tier[s < p80] = "low"
    tier[(s >= p80) & (s < p95)] = "med"
    tier[s >= p95] = "high"
    return tier, p80, p95


# Load
df = pd.read_csv("data/synthetic/member_month_with_volatility.csv")

# Out-of-time split
train, test, cutoff_ym = temporal_split(df)
print("Temporal split cutoff ym:", cutoff_ym)
print("Train rows:", len(train), "Test rows:", len(test))

# -----------------------------
# 1) Spike classifier
# -----------------------------
train_c = train.dropna(subset=["spike_flag_member"])
test_c = test.dropna(subset=["spike_flag_member"])

X_train_c = train_c[FEATURES]
y_train_c = train_c["spike_flag_member"].astype(int)

X_test_c = test_c[FEATURES]
y_test_c = test_c["spike_flag_member"].astype(int)

spike_clf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LogisticRegression(max_iter=2000)),
])

spike_clf.fit(X_train_c, y_train_c)
p_spike = spike_clf.predict_proba(X_test_c)[:, 1]

auc = roc_auc_score(y_test_c, p_spike)
base_rate = y_test_c.mean()

print("\nSpike model (test)")
print("------------------")
print(f"Base rate: {base_rate:.4f}")
print(f"ROC-AUC:   {auc:.3f}")

# -----------------------------
# 2) Volatility regression (leakage-safe)
# -----------------------------
train_r = train.dropna(subset=["volatility_score"])
test_r = test.dropna(subset=["volatility_score"])

X_train_r = train_r[FEATURES]
y_train_r = train_r["volatility_score"]

X_test_r = test_r[FEATURES]
y_test_r = test_r["volatility_score"]

vol_reg = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LinearRegression()),
])

vol_reg.fit(X_train_r, y_train_r)
vol_pred = vol_reg.predict(X_test_r)

mae = mean_absolute_error(y_test_r, vol_pred)
r = rmse(y_test_r, vol_pred)

print("\nVolatility model (test)")
print("-----------------------")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {r:.2f}")

# -----------------------------
# 3) Build FVI on the TEST period
# Align by row index: we want a common frame to score
# We'll score on rows where both are available (features exist; labels optional)
# -----------------------------
score_df = test.copy()
score_df = score_df.reset_index(drop=True)

X_score = score_df[FEATURES]

p_spike_all = spike_clf.predict_proba(X_score)[:, 1]
vol_pred_all = vol_reg.predict(X_score)

score_df["p_spike"] = p_spike_all
score_df["vol_pred"] = vol_pred_all

# FVI = risk × impact
score_df["fvi"] = score_df["p_spike"] * score_df["vol_pred"]

# Tiers based on TEST distribution (what ops would see)
score_df["fvi_tier"], p80, p95 = make_tiers(score_df["fvi"], q=(0.80, 0.95))

print("\nFVI distribution (test scoring period)")
print("--------------------------------------")
print(score_df["fvi"].describe())
print("\nFVI tier counts:")
print(score_df["fvi_tier"].value_counts())

print(f"\nTier thresholds: p80={p80:.2f}, p95={p95:.2f}")

# -----------------------------
# 4) Validate that high FVI actually corresponds to spikes (sanity)
# Only where spike label exists
# -----------------------------
eval_df = score_df.dropna(subset=["spike_flag_member"]).copy()
eval_df["spike_flag_member"] = eval_df["spike_flag_member"].astype(int)

high_bucket = eval_df[eval_df["fvi_tier"] == "high"]
overall_spike = eval_df["spike_flag_member"].mean()
high_spike = high_bucket["spike_flag_member"].mean()

print("\nFVI sanity check (spike concentration)")
print("-------------------------------------")
print(f"Overall spike rate (test): {overall_spike:.4f}")
print(f"Spike rate in HIGH FVI tier: {high_spike:.4f}")

# -----------------------------
# 5) Member-level rollup (how you'd operationalize outreach lists)
# -----------------------------
member_rollup = (
    score_df.groupby("member_id", as_index=False)
    .agg(
        fvi_mean=("fvi", "mean"),
        fvi_max=("fvi", "max"),
        p_spike_max=("p_spike", "max"),
        vol_pred_mean=("vol_pred", "mean"),
        months_scored=("fvi", "size"),
    )
    .sort_values("fvi_max", ascending=False)
)

out_path = "data/synthetic/fvi_scored_test_period.csv"
score_df.to_csv(out_path, index=False)
out_path2 = "data/synthetic/fvi_member_rollup.csv"
member_rollup.to_csv(out_path2, index=False)

print(f"\nWrote scoring output: {out_path}")
print(f"Wrote member rollup:  {out_path2}")

print("\nTop 10 members by FVI max:")
print(member_rollup.head(10))
