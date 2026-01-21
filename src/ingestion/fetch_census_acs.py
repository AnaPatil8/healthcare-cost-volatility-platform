# src/ingestion/fetch_census_acs.py
"""
Keyless Census ACS ingestion (county-level).

We use ACS 5-year endpoints by year:
  https://api.census.gov/data/{YEAR}/acs/acs5

Outputs:
  data/public/raw/acs5_county_{start}_{end}.csv
  data/public/processed/acs5_county_features_{start}_{end}.csv

Features included (county-level):
- median household income (B19013_001E)
- poverty count + poverty universe (B17001_002E, B17001_001E) -> poverty_rate
- total population (B01003_001E)

All joined by:
  year, state_fips, county_fips, county_fips5 (2+3), name
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests


@dataclass
class FetchConfig:
    start_year: int
    end_year: int
    timeout_s: int = 30
    max_retries: int = 5
    sleep_s_between_calls: float = 0.6


ACS_VARS = {
    "NAME": "NAME",
    "population": "B01003_001E",
    "median_household_income": "B19013_001E",
    "poverty_count": "B17001_002E",
    "poverty_universe": "B17001_001E",
}


def _request_with_retries(url: str, params: dict, timeout_s: int, max_retries: int):
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout_s)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            sleep_s = min(8.0, 0.8 * (2 ** (attempt - 1)))
            print(f"[warn] Census request failed (attempt {attempt}/{max_retries}): {e}")
            print(f"[warn] sleeping {sleep_s:.1f}s then retrying...")
            time.sleep(sleep_s)
    raise RuntimeError(f"Census request failed after {max_retries} retries: {last_err}")  # type: ignore


def fetch_acs5_county_year(cfg: FetchConfig, year: int) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    url = f"{base}"

    # Build GET list: NAME + variables + state + county
    get_vars = [
        ACS_VARS["NAME"],
        ACS_VARS["population"],
        ACS_VARS["median_household_income"],
        ACS_VARS["poverty_count"],
        ACS_VARS["poverty_universe"],
    ]
    params = {
        "get": ",".join(get_vars),
        "for": "county:*",
        "in": "state:*",
        # keyless: no API key
    }

    print(f"[info] Fetching ACS5 county for year={year} ...")
    data = _request_with_retries(url, params=params, timeout_s=cfg.timeout_s, max_retries=cfg.max_retries)

    if not data or len(data) < 2:
        raise RuntimeError(f"No ACS data returned for year {year}")

    header = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=header)

    # Standardize
    df["year"] = year
    df = df.rename(columns={
        "state": "state_fips",
        "county": "county_fips",
        "NAME": "name",
        ACS_VARS["population"]: "population",
        ACS_VARS["median_household_income"]: "median_household_income",
        ACS_VARS["poverty_count"]: "poverty_count",
        ACS_VARS["poverty_universe"]: "poverty_universe",
    })

    # FIPS padding + combined county fips5
    df["state_fips"] = df["state_fips"].astype(str).str.zfill(2)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(3)
    df["county_fips5"] = df["state_fips"] + df["county_fips"]

    # Numeric conversion (Census sometimes returns nulls as None or strings)
    for c in ["population", "median_household_income", "poverty_count", "poverty_universe"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[[
        "year", "county_fips5", "state_fips", "county_fips", "name",
        "population", "median_household_income", "poverty_count", "poverty_universe"
    ]]


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Poverty rate
    df["poverty_rate"] = (df["poverty_count"] / df["poverty_universe"]) * 100.0

    # Basic sanity flags
    df["poverty_rate"] = df["poverty_rate"].where(df["poverty_universe"] > 0)

    # Optional: cap impossible values
    df["poverty_rate"] = df["poverty_rate"].clip(lower=0, upper=100)

    # Keep only what we want for modeling
    return df[[
        "year", "county_fips5", "name",
        "population", "median_household_income", "poverty_rate"
    ]].sort_values(["year", "county_fips5"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--end-year", type=int, default=datetime.now().year - 1)
    ap.add_argument("--out-raw", type=str, default="")
    ap.add_argument("--out-features", type=str, default="")
    args = ap.parse_args()

    start_year = args.start_year
    end_year = args.end_year

    out_raw = Path(args.out_raw) if args.out_raw else Path(f"data/public/raw/acs5_county_{start_year}_{end_year}.csv")
    out_features = Path(args.out_features) if args.out_features else Path(f"data/public/processed/acs5_county_features_{start_year}_{end_year}.csv")

    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_features.parent.mkdir(parents=True, exist_ok=True)

    cfg = FetchConfig(start_year=start_year, end_year=end_year)

    print("[info] starting Census ACS5 fetch (keyless)...")
    all_years: List[pd.DataFrame] = []
    for y in range(start_year, end_year + 1):
        df_y = fetch_acs5_county_year(cfg, y)
        print(f"[info] year={y} rows: {len(df_y):,}")
        all_years.append(df_y)
        time.sleep(cfg.sleep_s_between_calls)

    df_raw = pd.concat(all_years, ignore_index=True)
    print(f"[info] total raw rows: {len(df_raw):,}")

    df_raw.to_csv(out_raw, index=False)
    print(f"[info] wrote raw: {out_raw.as_posix()}")

    df_feat = build_features(df_raw)
    print(f"[info] total feature rows: {len(df_feat):,}")

    df_feat.to_csv(out_features, index=False)
    print(f"[info] wrote features: {out_features.as_posix()}")

    print("[info] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
