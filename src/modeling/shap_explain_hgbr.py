from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import shap
import matplotlib.pyplot as plt

DATA_PATH = Path("data/public/processed/model_table.csv")
MODEL_PATH = Path("artifacts/hgbr_volatility_model.joblib")

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
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run train_county_volatility_hgbr.py first.")

    df = pd.read_csv(DATA_PATH)

    # Explain on TEST window (year>=2021)
    test_df = df[df["year"] >= 2021].copy()
    X = test_df[FEATURES]

    model = joblib.load(MODEL_PATH)

    # Pipeline: imputer -> hgbr
    imputer = model.named_steps["imputer"]
    hgbr = model.named_steps["hgbr"]

    X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURES)

    # SHAP explain
    # For some sklearn models, TreeExplainer may or may not work depending on shap/sklearn versions.
    # We'll try TreeExplainer first; fallback to generic Explainer.
    try:
        explainer = shap.TreeExplainer(hgbr)
        shap_values = explainer.shap_values(X_imp)
    except Exception as e:
        print("[warn] TreeExplainer failed, falling back to shap.Explainer:", repr(e))
        background = X_imp.sample(min(500, len(X_imp)), random_state=42)
        explainer = shap.Explainer(hgbr.predict, background)
        shap_values = explainer(X_imp).values

    Path("reports").mkdir(exist_ok=True)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_imp, show=False)
    out_png = Path("reports/shap_summary_hgbr.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[info] wrote: {out_png.as_posix()}")

    # Global importance table: mean absolute SHAP
    imp = np.mean(np.abs(shap_values), axis=0)
    imp_df = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": imp}).sort_values(
        "mean_abs_shap", ascending=False
    )
    out_csv = Path("reports/shap_importance_hgbr.csv")
    imp_df.to_csv(out_csv, index=False)
    print(f"[info] wrote: {out_csv.as_posix()}")

    print("\nTop features:")
    print(imp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
