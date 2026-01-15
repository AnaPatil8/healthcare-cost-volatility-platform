from __future__ import annotations

import os
import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path",
        type=str,
        default=os.path.join("data", "synthetic", "member_month_costs.csv"),
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=os.path.join("data", "synthetic", "member_month_with_volatility.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)

    # sort for rolling windows
    df["ym"] = df["year"] * 100 + df["month"]
    df = df.sort_values(["member_id", "ym"]).reset_index(drop=True)

    # Global threshold for "absolute spike"
    global_p99 = df["total_cost"].quantile(0.99)

    g = df.groupby("member_id", group_keys=False)

    # Rolling mean/std over 6 months
    df["rolling_mean_6"] = g["total_cost"].apply(lambda s: s.rolling(6, min_periods=6).mean())
    df["rolling_std_6"] = g["total_cost"].apply(lambda s: s.rolling(6, min_periods=6).std())

    # Rolling quantiles (p90 over 6, p95 over 12)
    df["rolling_p90_6"] = g["total_cost"].apply(lambda s: s.rolling(6, min_periods=6).quantile(0.90))
    df["rolling_p95_12"] = g["total_cost"].apply(lambda s: s.rolling(12, min_periods=12).quantile(0.95))

    # Spike flags
    df["spike_flag_member"] = (df["total_cost"] > df["rolling_p95_12"]).astype("Int64")
    df["spike_flag_global"] = (df["total_cost"] > global_p99).astype(int)

    # Spike rate over last 6 months
    df["spike_rate_6"] = g["spike_flag_member"].apply(lambda s: s.rolling(6, min_periods=6).mean())

    # Volatility tiers from rolling_std_6 quantiles
    std_vals = df["rolling_std_6"].dropna()
    q1 = std_vals.quantile(0.33)
    q2 = std_vals.quantile(0.66)

    def tier(x):
        if pd.isna(x):
            return "unknown"
        if x <= q1:
            return "low"
        if x <= q2:
            return "med"
        return "high"

    df["volatility_tier"] = df["rolling_std_6"].apply(tier)
    df["volatility_score"] = df["rolling_std_6"]

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    df.to_csv(args.out_path, index=False)

    print(f"Wrote: {args.out_path}")
    print("Global p99 threshold:", float(global_p99))
    print("Rows:", len(df))
    print("Non-null volatility_score rows:", int(df["volatility_score"].notna().sum()))
    print("Member spike rate mean:", float(df["spike_flag_member"].mean(skipna=True)))
    print("Tier counts:\n", df["volatility_tier"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
