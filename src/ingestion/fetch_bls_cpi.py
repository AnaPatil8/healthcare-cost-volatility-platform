# src/ingestion/fetch_bls_cpi.py
"""
Keyless BLS CPI ingestion.

Outputs:
  data/public/raw/bls_cpi_monthly.csv
  data/public/processed/bls_cpi_annual.csv

Notes:
- Keyless BLS v2 is rate-limited and may restrict years per request.
- We fetch monthly CPI series and compute annual averages + YoY.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


DEFAULT_SERIES = {
    # CPI-U, U.S. city average, All items (SA)
    "cpi_all_items": "CUUR0000SA0",
    # CPI-U, U.S. city average, All items less food and energy (core) (SA)
    "cpi_core": "CUUR0000SA0L1E",
    # CPI-U, U.S. city average, Medical care (SA)
    "cpi_medical": "CUUR0000SAM",
}


@dataclass
class FetchConfig:
    start_year: int
    end_year: int
    series_map: Dict[str, str]
    timeout_s: int = 30
    max_retries: int = 5
    sleep_s_between_calls: float = 0.6  # be polite to keyless API


def _safe_int(x: str, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return default


def _request_with_retries(
    url: str,
    payload: dict,
    timeout_s: int,
    max_retries: int,
) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            # exponential backoff
            sleep_s = min(8.0, 0.8 * (2 ** (attempt - 1)))
            print(f"[warn] BLS request failed (attempt {attempt}/{max_retries}): {e}")
            print(f"[warn] sleeping {sleep_s:.1f}s then retrying...")
            time.sleep(sleep_s)
    raise RuntimeError(f"BLS request failed after {max_retries} retries: {last_err}")  # type: ignore


def fetch_bls_monthly(cfg: FetchConfig) -> pd.DataFrame:
    """
    Fetch monthly series data from BLS for given years (inclusive).
    Returns tidy dataframe:
      series_name, series_id, year, period, period_name, value
    """
    # BLS keyless sometimes behaves better with smaller year windows.
    # We'll chunk into <=10-year blocks.
    CHUNK_YEARS = 10
    series_ids = list(cfg.series_map.values())
    inverse_map = {v: k for k, v in cfg.series_map.items()}

    all_rows: List[dict] = []

    y0 = cfg.start_year
    while y0 <= cfg.end_year:
        y1 = min(cfg.end_year, y0 + CHUNK_YEARS - 1)

        payload = {
            "seriesid": series_ids,
            "startyear": str(y0),
            "endyear": str(y1),
            # no registrationkey (keyless)
        }

        print(f"[info] Fetching BLS series for years {y0}..{y1} (series={len(series_ids)})")
        data = _request_with_retries(BLS_URL, payload, cfg.timeout_s, cfg.max_retries)

        # BLS uses status and message fields
        status = data.get("status")
        if status != "REQUEST_SUCCEEDED":
            msg = data.get("message")
            raise RuntimeError(f"BLS API returned status={status}, message={msg}")

        results = data.get("Results", {})
        series_list = results.get("series", [])

        for s in series_list:
            sid = s.get("seriesID")
            sname = inverse_map.get(sid, sid)
            for pt in s.get("data", []):
                year = _safe_int(pt.get("year", ""), None)
                period = pt.get("period")  # e.g., "M01".."M12" or "M13" annual avg
                period_name = pt.get("periodName")
                value = pt.get("value")

                # Keep only months M01..M12 for clean time series
                if not isinstance(period, str) or not period.startswith("M"):
                    continue
                m = _safe_int(period.replace("M", ""), None)
                if m is None or m < 1 or m > 12:
                    continue
                if year is None:
                    continue

                all_rows.append(
                    {
                        "series_name": sname,
                        "series_id": sid,
                        "year": year,
                        "month": m,
                        "period": period,
                        "period_name": period_name,
                        "value": float(value) if value not in (None, "", "NA") else None,
                    }
                )

        # polite sleep between calls
        time.sleep(cfg.sleep_s_between_calls)
        y0 = y1 + 1

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No BLS rows returned. Try a smaller year range or check series IDs.")
    return df.sort_values(["series_name", "year", "month"]).reset_index(drop=True)


def compute_annual_and_yoy(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monthly CPI levels into annual average and YoY change.
    Output columns:
      year, cpi_all_items, cpi_core, cpi_medical, cpi_*_yoy
    """
    # Annual average CPI level
    annual = (
        df_monthly.groupby(["series_name", "year"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "annual_avg"})
    )

    wide = annual.pivot(index="year", columns="series_name", values="annual_avg").reset_index()

    # YoY % change (annual_avg[t] / annual_avg[t-1] - 1)
    for col in [c for c in wide.columns if c != "year"]:
        wide[f"{col}_yoy"] = wide[col].pct_change() * 100.0

    return wide.sort_values("year").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--end-year", type=int, default=datetime.now().year - 1)
    ap.add_argument("--out-raw", type=str, default="data/public/raw/bls_cpi_monthly.csv")
    ap.add_argument("--out-annual", type=str, default="data/public/processed/bls_cpi_annual.csv")
    args = ap.parse_args()

    cfg = FetchConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        series_map=DEFAULT_SERIES,
    )

    out_raw = Path(args.out_raw)
    out_annual = Path(args.out_annual)
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_annual.parent.mkdir(parents=True, exist_ok=True)

    print("[info] starting BLS CPI fetch (keyless)...")
    df_monthly = fetch_bls_monthly(cfg)
    print(f"[info] fetched monthly rows: {len(df_monthly):,}")

    df_monthly.to_csv(out_raw, index=False)
    print(f"[info] wrote raw monthly: {out_raw.as_posix()}")

    df_annual = compute_annual_and_yoy(df_monthly)
    print(f"[info] annual rows: {len(df_annual):,} (years={df_annual['year'].min()}..{df_annual['year'].max()})")

    df_annual.to_csv(out_annual, index=False)
    print(f"[info] wrote processed annual: {out_annual.as_posix()}")

    print("[info] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
