import pandas as pd

def temporal_split(df, train_frac=0.75):
    """
    Industry concept: Out-of-time validation.
    Splits by time using ym (YYYY-MM) so test is always later months.
    """
    d = df.copy()
    d["ym_dt"] = pd.to_datetime(d["ym"].astype(str), format="%Y%m")

    # Sort by time, then choose a cutoff at train_frac
    unique_ym = sorted(d["ym"].unique())
    cutoff_idx = int(len(unique_ym) * train_frac) - 1
    cutoff_ym = unique_ym[cutoff_idx]

    train = d[d["ym"] <= cutoff_ym]
    test  = d[d["ym"] > cutoff_ym]

    return train, test, cutoff_ym

