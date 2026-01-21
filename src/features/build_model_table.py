# src/features/build_model_table.py
"""
Build a unified modeling table:
CMS (county-year outcomes/features) + ACS (county-year sociodemographics) + BLS CPI (year-level macro)

Inputs:
  data/public/cms_geo_profile_full.csv
  data/public/processed/acs5_county_features_YYYY_YYYY.csv
  data/public/processed/bls_cpi_annual.csv

Outputs:
  data/public/processed/model_table.csv

Target:
  target_volatility_next_year = abs(YoY % change in CMS TOT_MDCR_PYMT_PC) shifted -1 within county
  (i.e., features at year t predict volatility at year t+1)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def read_cms(cms_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cms_path, low_memory=False)

    # Normalize expected columns
    needed = [
        "YEAR",
        "BENE_GEO_LVL",
        "BENE_GEO_DESC",
        "BENE_GEO_CD",
        "BENE_AGE_LVL",
        "BENES_TOTAL_CNT",
        "BENE_AVG_RISK_SCRE",
        "TOT_MDCR_PYMT_AMT",
        "TOT_MDCR_PYMT_PC",
        "BENES_ER_VISITS_CNT",
        "ER_VISITS_PER_1000_BENES",
        "BENES_IP_CVRD_STAY_CNT",
        "IP_CVRD_STAYS_PER_1000_BENES",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"CMS file missing columns: {missing}")

    # Filter to county level & all ages
    df = df[(df["BENE_GEO_LVL"] == "County") & (df["BENE_AGE_LVL"] == "All")].copy()

    # YEAR -> int
    df["year"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")

    # County FIPS key: BENE_GEO_CD is already 5-digit for county rows, but pad just in case
    df["county_fips5"] = df["BENE_GEO_CD"].astype(str).str.strip()
    df["county_fips5"] = df["county_fips5"].str.replace(r"\.0$", "", regex=True)  # safety if read as float
    df["county_fips5"] = df["county_fips5"].str.zfill(5)

    # Keep a clean county name (description)
    df["county_name"] = df["BENE_GEO_DESC"].astype(str).str.strip()

    # Convert numeric core columns
    num_cols = [
        "BENES_TOTAL_CNT",
        "BENE_AVG_RISK_SCRE",
        "TOT_MDCR_PYMT_AMT",
        "TOT_MDCR_PYMT_PC",
        "BENES_ER_VISITS_CNT",
        "ER_VISITS_PER_1000_BENES",
        "BENES_IP_CVRD_STAY_CNT",
        "IP_CVRD_STAYS_PER_1000_BENES",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Select modeling columns
    out = df[
        [
            "year",
            "county_fips5",
            "county_name",
            "BENES_TOTAL_CNT",
            "BENE_AVG_RISK_SCRE",
            "TOT_MDCR_PYMT_AMT",
            "TOT_MDCR_PYMT_PC",
            "BENES_ER_VISITS_CNT",
            "ER_VISITS_PER_1000_BENES",
            "BENES_IP_CVRD_STAY_CNT",
            "IP_CVRD_STAYS_PER_1000_BENES",
        ]
    ].copy()

    # Rename to consistent snake_case
    out = out.rename(
        columns={
            "BENES_TOTAL_CNT": "cms_benes_total_cnt",
            "BENE_AVG_RISK_SCRE": "cms_bene_avg_risk_score",
            "TOT_MDCR_PYMT_AMT": "cms_tot_mdcr_pymt_amt",
            "TOT_MDCR_PYMT_PC": "cms_tot_mdcr_pymt_pc",
            "BENES_ER_VISITS_CNT": "cms_benes_er_visits_cnt",
            "ER_VISITS_PER_1000_BENES": "cms_er_visits_per_1000_benes",
            "BENES_IP_CVRD_STAY_CNT": "cms_benes_ip_cvrd_stay_cnt",
            "IP_CVRD_STAYS_PER_1000_BENES": "cms_ip_cvrd_stays_per_1000_benes",
        }
    )

    # Drop rows with missing keys
    out = out.dropna(subset=["year", "county_fips5"]).copy()
    out["year"] = out["year"].astype(int)

    return out.sort_values(["county_fips5", "year"]).reset_index(drop=True)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build volatility target from cms_tot_mdcr_pymt_pc:
      yoy_pct = pct_change * 100 per county
      volatility = abs(yoy_pct)
      target_volatility_next_year = volatility shifted -1 (i.e. year t -> year t+1)
    """
    df = df.sort_values(["county_fips5", "year"]).copy()

    df["cms_tot_pymt_pc_yoy_pct"] = (
        df.groupby("county_fips5")["cms_tot_mdcr_pymt_pc"]
        .pct_change(fill_method=None)
        * 100.0
    )

    df["cms_tot_pymt_pc_volatility"] = df["cms_tot_pymt_pc_yoy_pct"].abs()

    # This is the label we train on: next year's volatility
    df["target_volatility_next_year"] = (
        df.groupby("county_fips5")["cms_tot_pymt_pc_volatility"].shift(-1)
    )

    return df


def read_acs(acs_path: Path) -> pd.DataFrame:
    df = pd.read_csv(acs_path, low_memory=False)

    needed = ["year", "county_fips5", "population", "median_household_income", "poverty_rate"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"ACS file missing columns: {missing}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["county_fips5"] = df["county_fips5"].astype(str).str.zfill(5)

    # Rename with prefix
    df = df.rename(
        columns={
            "population": "acs_population",
            "median_household_income": "acs_median_household_income",
            "poverty_rate": "acs_poverty_rate",
        }
    )

    return df.dropna(subset=["year", "county_fips5"]).assign(year=lambda d: d["year"].astype(int))


def read_cpi(cpi_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cpi_path, low_memory=False)
    if "year" not in df.columns:
        raise RuntimeError("BLS CPI file missing 'year' column")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # Keep only CPI columns (everything except year)
    keep_cols = ["year"] + [c for c in df.columns if c != "year"]
    df = df[keep_cols].copy()

    # Prefix CPI columns so they’re obvious
    rename = {c: f"cpi_{c}" for c in df.columns if c != "year"}
    df = df.rename(columns=rename)

    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cms", type=str, default="data/public/cms_geo_profile_full.csv")
    ap.add_argument("--acs", type=str, default="data/public/processed/acs5_county_features_2014_2023.csv")
    ap.add_argument("--cpi", type=str, default="data/public/processed/bls_cpi_annual.csv")
    ap.add_argument("--out", type=str, default="data/public/processed/model_table.csv")
    args = ap.parse_args()

    cms_path = Path(args.cms)
    print(f"[info] CMS path: {cms_path.as_posix()}")

    acs_path = Path(args.acs)
    cpi_path = Path(args.cpi)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[info] reading CMS...")
    df_cms = read_cms(cms_path)
    print(f"[info] CMS county+All rows: {len(df_cms):,} | years {df_cms['year'].min()}..{df_cms['year'].max()}")

    print("[info] adding targets...")
    df_cms = add_targets(df_cms)

    print("[info] reading ACS...")
    df_acs = read_acs(acs_path)
    print(f"[info] ACS rows: {len(df_acs):,}")

    print("[info] reading CPI...")
    df_cpi = read_cpi(cpi_path)
    print(f"[info] CPI rows: {len(df_cpi):,}")

    print("[info] merging CMS + ACS (county-year)...")
    df = df_cms.merge(df_acs, on=["year", "county_fips5"], how="left")

    print("[info] merging + CPI (year-level)...")
    df = df.merge(df_cpi, on=["year"], how="left")

    # Drop rows where target is missing (last year per county)
    before = len(df)
    df = df.dropna(subset=["target_volatility_next_year"]).copy()
    after = len(df)
    print(f"[info] dropped {before - after:,} rows without target (end-of-series years)")

    # Optional: drop rows with missing core feature values
    # Keep this light; let modeling pipeline decide imputation if needed.
    df = df.sort_values(["county_fips5", "year"]).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"[info] wrote model table: {out_path.as_posix()} | rows={len(df):,} cols={df.shape[1]}")

    print("[info] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
