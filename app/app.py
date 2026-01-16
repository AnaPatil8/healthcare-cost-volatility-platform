import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Financial Volatility Index (FVI)", layout="wide")

st.title("Financial Volatility Index (FVI)")
st.caption("Decision support for identifying members at risk of near-term financial unpredictability.")

uploaded = st.file_uploader("Upload FVI scored CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    required = {"member_id", "fvi", "fvi_tier", "p_spike", "vol_pred"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {sorted(missing)}")
        st.stop()

    # ---- KPIs ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows scored", f"{len(df):,}")
    c2.metric("Unique members", f"{df['member_id'].nunique():,}")
    c3.metric("High-tier rows", f"{int((df['fvi_tier']=='high').sum()):,}")
    c4.metric("Max FVI", f"{df['fvi'].max():.2f}")

    st.divider()

    # ---- Filters ----
    st.subheader("Risk filtering")
    left, mid, right = st.columns(3)

    tier = left.multiselect("Tier", ["high", "med", "low"], default=["high"])
    min_p = mid.slider("Minimum spike probability (p_spike)", 0.0, 1.0, 0.50, 0.01)
    sort_by = right.selectbox("Sort by", ["fvi", "p_spike", "vol_pred"], index=0)

    top_n = st.slider("Rows to show", 10, 500, 50, 10)

    view = df[df["fvi_tier"].isin(tier)].copy()
    view = view[view["p_spike"] >= min_p].copy()
    view = view.sort_values(sort_by, ascending=False).head(top_n)

    cols = ["member_id", "fvi", "fvi_tier", "p_spike", "vol_pred"]
    if "year" in df.columns and "month" in df.columns:
        cols += ["year", "month"]

    st.dataframe(view[cols], use_container_width=True)

    st.download_button(
        "Download current view (CSV)",
        data=view[cols].to_csv(index=False),
        file_name="fvi_watchlist.csv",
        mime="text/csv",
    )

    st.divider()

    # ---- Member rollup ----
    st.subheader("Member-level watchlist (rollup)")
    st.caption("Roll up months into a member-level list (max risk is often used for outreach queues).")

    roll = (
        df.groupby("member_id", as_index=False)
          .agg(
              fvi_max=("fvi", "max"),
              fvi_mean=("fvi", "mean"),
              p_spike_max=("p_spike", "max"),
              vol_pred_mean=("vol_pred", "mean"),
              months_scored=("fvi", "size"),
          )
          .sort_values("fvi_max", ascending=False)
    )

    member_top_n = st.slider("Members to show", 10, 500, 50, 10, key="members_n")
    st.dataframe(roll.head(member_top_n), use_container_width=True)

    st.download_button(
        "Download member rollup (CSV)",
        data=roll.to_csv(index=False),
        file_name="fvi_member_rollup_from_app.csv",
        mime="text/csv",
    )

    st.divider()

    # ---- Optional: sanity check if spike label exists ----
    if "spike_flag_member" in df.columns:
        st.subheader("Sanity check (if labels present)")
        labeled = df.dropna(subset=["spike_flag_member"]).copy()
        labeled["spike_flag_member"] = labeled["spike_flag_member"].astype(int)

        overall = labeled["spike_flag_member"].mean()
        high_rate = labeled.loc[labeled["fvi_tier"] == "high", "spike_flag_member"].mean()

        s1, s2 = st.columns(2)
        s1.metric("Overall spike rate", f"{overall:.3f}")
        s2.metric("Spike rate in HIGH tier", f"{high_rate:.3f}")

        st.caption("This is a quick concentration check: HIGH tier should have much higher spike rate than overall.")
else:
    st.info("Upload `fvi_scored_test_period.csv` to begin.")
