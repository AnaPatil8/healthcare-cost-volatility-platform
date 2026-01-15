"""
Synthetic Healthcare Member-Month Cost Generator
------------------------------------------------
Generates a realistic(ish) member-month dataset with:
- skewed base costs (log-normal)
- utilization-driven costs
- rare shock events (heavy tail)
- seasonality (winter higher)
- correlations (chronic_count -> higher shock probability)

Output:
  data/synthetic/member_month_costs.csv
"""

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Config
# ----------------------------
@dataclass
class GenConfig:
    n_members: int = 20000
    n_months: int = 24
    start_year: int = 2023
    start_month: int = 1
    seed: int = 42

    # cost distribution controls
    base_log_mu: float = 6.2      # controls typical base cost scale
    base_log_sigma: float = 0.55  # controls skew/heavy tail

    # utilization
    util_lambda_base: float = 1.8
    er_base_rate: float = 0.08
    ip_base_rate: float = 0.03

    # shocks
    shock_base_prob: float = 0.015
    shock_cost_mu: float = 9.5
    shock_cost_sigma: float = 0.9


def month_index_to_ym(start_year: int, start_month: int, idx: int) -> Tuple[int, int]:
    """0-based month index -> (year, month)"""
    m = start_month - 1 + idx
    year = start_year + (m // 12)
    month = (m % 12) + 1
    return year, month


def seasonality_multiplier(month: int) -> float:
    """
    Simple seasonality:
    - Nov–Mar slightly higher (flu season)
    - Summer slightly lower
    """
    if month in (11, 12, 1, 2, 3):
        return 1.12
    if month in (6, 7, 8):
        return 0.95
    return 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_members", type=int, default=20000)
    parser.add_argument("--n_months", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = GenConfig(n_members=args.n_members, n_months=args.n_months, seed=args.seed)

    rng = np.random.default_rng(cfg.seed)

    # ----------------------------
    # Member-level features
    # ----------------------------
    member_id = np.arange(1, cfg.n_members + 1)

    # age: biased adult population
    age = rng.integers(18, 85, size=cfg.n_members)

    # chronic_count: correlated with age (older -> more chronic)
    # Use a soft mapping: expected chronic rises with age
    chronic_mean = np.clip((age - 30) / 20, 0, 3)  # ~0 to 3
    chronic_count = rng.poisson(lam=0.6 + chronic_mean, size=cfg.n_members)
    chronic_count = np.clip(chronic_count, 0, 8)

    # risk_score: latent factor combining age + chronic burden
    risk_score = (
        0.015 * (age - 40)
        + 0.35 * chronic_count
        + rng.normal(0, 0.6, size=cfg.n_members)
    )

    # Expand to member-month
    n_rows = cfg.n_members * cfg.n_months
    member_id_rep = np.repeat(member_id, cfg.n_months)
    age_rep = np.repeat(age, cfg.n_months)
    chronic_rep = np.repeat(chronic_count, cfg.n_months)
    risk_rep = np.repeat(risk_score, cfg.n_months)

    # month index per row
    m_idx = np.tile(np.arange(cfg.n_months), cfg.n_members)

    # year/month columns
    years = np.empty(n_rows, dtype=int)
    months = np.empty(n_rows, dtype=int)
    for i in range(cfg.n_months):
        y, m = month_index_to_ym(cfg.start_year, cfg.start_month, i)
        years[m_idx == i] = y
        months[m_idx == i] = m

    # ----------------------------
    # Utilization generation
    # ----------------------------
    # utilization_events ~ Poisson; higher with risk
    util_lambda = cfg.util_lambda_base * np.exp(0.18 * risk_rep)
    utilization_events = rng.poisson(lam=np.clip(util_lambda, 0.1, 20))

    # ER / inpatient flags: Bernoulli with risk dependence
    er_prob = np.clip(cfg.er_base_rate * np.exp(0.22 * risk_rep), 0, 0.75)
    ip_prob = np.clip(cfg.ip_base_rate * np.exp(0.30 * risk_rep), 0, 0.40)
    er_visit = rng.binomial(1, er_prob)
    inpatient = rng.binomial(1, ip_prob)

    # ----------------------------
    # Base cost generation (skewed)
    # ----------------------------
    # Base cost is lognormal, scaled by utilization and seasonality
    base_cost = rng.lognormal(mean=cfg.base_log_mu, sigma=cfg.base_log_sigma, size=n_rows)

    # Utilization adds cost (nonlinear-ish)
    util_cost = utilization_events * rng.lognormal(mean=5.0, sigma=0.35, size=n_rows)

    # ER and inpatient add bigger chunks
    er_cost = er_visit * rng.lognormal(mean=7.2, sigma=0.45, size=n_rows)
    ip_cost = inpatient * rng.lognormal(mean=8.0, sigma=0.55, size=n_rows)

    # Seasonality multiplier
    season_mult = np.vectorize(seasonality_multiplier)(months).astype(float)

    # ----------------------------
    # Shock events (rare, high tail)
    # ----------------------------
    # Shock probability increases with chronic burden + risk
    shock_prob = np.clip(cfg.shock_base_prob * np.exp(0.25 * risk_rep) * (1 + 0.12 * chronic_rep), 0, 0.35)
    shock_event = rng.binomial(1, shock_prob)

    # Shock costs: heavy tail via lognormal with higher mean/sigma
    shock_cost = shock_event * rng.lognormal(mean=cfg.shock_cost_mu, sigma=cfg.shock_cost_sigma, size=n_rows)

    total_cost = (base_cost + util_cost + er_cost + ip_cost) * season_mult + shock_cost

    # guardrail: avoid negative/NaN (shouldn't happen)
    total_cost = np.nan_to_num(total_cost, nan=0.0, posinf=np.nanmax(total_cost[np.isfinite(total_cost)]))

    df = pd.DataFrame({
        "member_id": member_id_rep,
        "year": years,
        "month": months,
        "age": age_rep,
        "chronic_count": chronic_rep,
        "risk_score": np.round(risk_rep, 3),
        "utilization_events": utilization_events,
        "er_visit": er_visit,
        "inpatient": inpatient,
        "shock_event": shock_event,
        "total_cost": np.round(total_cost, 2),
    })

    out_path = os.path.join("data", "synthetic", "member_month_costs.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    # Print quick summary for sanity
    print(f"Wrote: {out_path}")
    print(df[["total_cost"]].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())
    print("\nShock rate:", df["shock_event"].mean().round(4))
    print("Rows:", len(df))


if __name__ == "__main__":
    main()
