from pathlib import Path
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

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


def main():
    df = pd.read_csv(DATA_PATH)

    # Time-based split
    train_df = df[df["year"] <= 2019].copy()
    test_df = df[df["year"] >= 2020].copy()

    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows:  {len(test_df):,}")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print("\nBaseline Volatility Regression")
    print("------------------------------")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    # Business-style metric: top decile capture
    test_df = test_df.copy()
    test_df["pred"] = preds

    threshold = test_df["pred"].quantile(0.90)
    top_decile = test_df[test_df["pred"] >= threshold]

    print("\nTop 10% Counties")
    print("----------------")
    print(f"Mean actual volatility: {top_decile[TARGET].mean():.2f}")
    print(f"Overall mean volatility: {test_df[TARGET].mean():.2f}")


if __name__ == "__main__":
    main()
