# src/ingestion/pull_cms_geo_profile.py
import time
import pandas as pd
import requests

DATASET_ID = "6219697b-8f6c-4164-bed4-cd9317c58ebc"
BASE_URL = f"https://data.cms.gov/data-api/v1/dataset/{DATASET_ID}/data"

OUT_CSV = "data/public/cms_geo_profile_full.csv"

# Keep this smaller at first while testing; can increase later.
PAGE_SIZE = 5000  # max supported is 5000 :contentReference[oaicite:2]{index=2}
SLEEP_SECONDS = 0.2

# Optional: limit columns to reduce download size (faster + smaller CSV)
COLUMNS = [
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

def fetch_page(offset: int) -> list[dict]:
    params = {
        "size": PAGE_SIZE,
        "offset": offset,
        "sort": "YEAR",  # helps ensure you don’t only see early-year pages
        "columns": ",".join(COLUMNS),
        # Optional filters (uncomment if you want):
        # "filter[BENE_GEO_LVL]": "County",
        # "filter[BENE_AGE_LVL]": "All",
    }
    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected response type: {type(data)}")
    return data

def main():
    print("[info] downloading CMS dataset with pagination...")
    all_rows = []
    offset = 0
    page_num = 1

    while True:
        rows = fetch_page(offset)
        if not rows:
            break

        all_rows.extend(rows)
        print(f"[info] fetched page {page_num} | rows this page: {len(rows)} | total: {len(all_rows)}")

        # If we got fewer than PAGE_SIZE, we’re at the end.
        if len(rows) < PAGE_SIZE:
            break

        offset += PAGE_SIZE
        page_num += 1
        time.sleep(SLEEP_SECONDS)

    df = pd.DataFrame(all_rows)

    # quick sanity checks
    if "YEAR" in df.columns:
        years = pd.to_numeric(df["YEAR"], errors="coerce")
        print(f"[info] YEAR min..max: {years.min()}..{years.max()}")

    df.to_csv(OUT_CSV, index=False)
    print(f"[info] wrote: {OUT_CSV} | rows={len(df)} cols={len(df.columns)}")

if __name__ == "__main__":
    main()
