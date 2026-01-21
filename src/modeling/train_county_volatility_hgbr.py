from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

DATA_PATH = Path("data/public/processed/model_table.csv")

FEATURES = [
    "cms_benes_total_cnt",
    "cms_bene_avg_risk_score",
    "cms_tot_mdcr_pymt_pc",
    "cms_er_visits_per_1000_benes",
    "cms_ip_cvrd_stays_per_1000_benes",
    "acs_population",
    "acs_median_household_income",
    "acs_poverty_rate",
]

TARGET = "target_volatility_next_year"


def eval_split(name: str, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def top_decile_capture(df: pd.DataFrame, pred_col="pred", target_col=TARGET):
    threshold = df[pred_col].quantile(0.90)
    top = df[df[pred_col] >= threshold]
    return {
        "threshold_pred_p90": float(threshold),
        "mean_actual_top10pct": float(top[target_col].mean()),
        "mean_actual_overall": float(df[target_col].mean()),
        "rows_top10pct": int(len(top)),
        "rows_total": int(len(df)),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    # Recommended split:
    # train: <=2019, valid: 2020, test: >=2021
    train_df = df[df["year"] <= 2019].copy()
    valid_df = df[df["year"] == 2020].copy()
    test_df = df[df["year"] >= 2021].copy()

    print(f"[info] Train rows: {len(train_df):,} | years {train_df.year.min()}..{train_df.year.max()}")
    print(f"[info] Valid rows: {len(valid_df):,} | years {valid_df.year.min()}..{valid_df.year.max()}")
    print(f"[info] Test  rows: {len(test_df):,} | years {test_df.year.min()}..{test_df.year.max()}")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_valid, y_valid = valid_df[FEATURES], valid_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # Tree model that handles non-linearities (stronger than Ridge on tabular)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("hgbr", HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_depth=6,
                max_iter=600,
                min_samples_leaf=30,
                l2_regularization=0.0,
                random_state=42,
            )),
        ]
    )

    model.fit(X_train, y_train)

    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    eval_split("Validation (year=2020)", y_valid, pred_valid)
    eval_split("Test (year>=2021)", y_test, pred_test)

    # Business-style metric on test
    test_out = test_df.copy()
    test_out["pred"] = pred_test
    capture = top_decile_capture(test_out)

    print("\nTop 10% Counties (TEST)")
    print("-----------------------")
    print(f"Pred threshold (p90):   {capture['threshold_pred_p90']:.3f}")
    print(f"Mean actual (top 10%):  {capture['mean_actual_top10pct']:.2f}")
    print(f"Mean actual (overall):  {capture['mean_actual_overall']:.2f}")
    print(f"Rows: {capture['rows_top10pct']}/{capture['rows_total']}")

    # Save model + test predictions (ignored by git if you use my .gitignore)
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(model, "artifacts/hgbr_volatility_model.joblib")
    test_out[["year", "county_fips5", "county_name", TARGET, "pred"]].to_csv(
        "artifacts/test_predictions.csv", index=False
    )
    print("\n[info] saved: artifacts/hgbr_volatility_model.joblib")
    print("[info] saved: artifacts/test_predictions.csv")


if __name__ == "__main__":
    main()
