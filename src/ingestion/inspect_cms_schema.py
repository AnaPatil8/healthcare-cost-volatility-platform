import requests
import pandas as pd

DATASET_ID = "6219697b-8f6c-4164-bed4-cd9317c58ebc"
URL = f"https://data.cms.gov/data-api/v1/dataset/{DATASET_ID}/data"

MISSING_TOKENS = {"NA", "*", ""}

def to_numeric_safe(s: pd.Series) -> pd.Series:
    # Replace common missing tokens with NA, then coerce
    s = s.replace(list(MISSING_TOKENS), pd.NA)
    return pd.to_numeric(s, errors="coerce")

def main():
    print("starting cms schema inspect + clean...")

    r = requests.get(URL, params={"page[size]": 1000})
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)

    print("Fetched rows:", len(df))
    print("Columns:", len(df.columns))

    # Keep a small set of “core” columns we care about first
    core_cols = [
        "YEAR", "BENE_GEO_LVL", "BENE_GEO_DESC", "BENE_GEO_CD", "BENE_AGE_LVL",
        "BENES_TOTAL_CNT", "BENE_AVG_RISK_SCRE",
        "TOT_MDCR_PYMT_AMT", "TOT_MDCR_PYMT_PC",
        "BENES_ER_VISITS_CNT", "ER_VISITS_PER_1000_BENES",
        "BENES_IP_CVRD_STAY_CNT", "IP_CVRD_STAYS_PER_1000_BENES",
    ]
    core_cols = [c for c in core_cols if c in df.columns]
    core = df[core_cols].copy()

    # Convert numeric-like columns
    numeric_candidates = [c for c in core.columns if c not in ["YEAR","BENE_GEO_LVL","BENE_GEO_DESC","BENE_GEO_CD","BENE_AGE_LVL"]]
    for c in numeric_candidates:
        core[c] = to_numeric_safe(core[c])

    print("\nDtypes (core):")
    print(core.dtypes)

    print("\nSample (core):")
    print(core.head(10).to_string(index=False))

    out_path = "data/public/cms_geo_profile_sample.csv"
    core.to_csv(out_path, index=False)
    print("\nWrote:", out_path)

if __name__ == "__main__":
    main()
