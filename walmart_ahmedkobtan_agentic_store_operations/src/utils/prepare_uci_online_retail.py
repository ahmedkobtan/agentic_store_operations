# scripts/prepare_uci_online_retail.py
# Purpose: Convert UCI Online Retail II transactions into a single-store hourly demand series.
# Input:  data/raw/online_retail_II.xlsx  (downloaded from UCI)
# Output: data/processed/hourly_demand.parquet

import pandas as pd
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import RAW_PATH_1, RAW_PATH_2, PROCESSED_OUT_DIR, PROCESSED_OUT_PATH

def main():
    if not RAW_PATH_1.exists() and not RAW_PATH_2.exists():
        raise FileNotFoundError(
            f"Missing {RAW_PATH_1} and {RAW_PATH_2}. Download 'online_retail_II_2009.xlsx' and 'online_retail_II_2010.xlsx' from the UCI dataset page "
            f"and place them under data/raw/. See README for link."
        )

    # Read Excel (single sheet; some variants have two year-splits—this file includes 2009–2011 in one)
    # If you happen to have two files (2009–2010 and 2010–2011), just concatenate them before grouping.
    df1 = pd.read_excel(RAW_PATH_1, engine="openpyxl")
    df2 = pd.read_excel(RAW_PATH_2, engine="openpyxl")
    df = pd.concat([df1, df2], ignore_index=True)

    # Keep only the columns we need for hourly counts and consistent types
    needed_cols = ["Invoice", "InvoiceDate"]
    df = df.dropna(subset=['Invoice', 'Quantity', 'InvoiceDate'])
    # remove cancellations
    df = df[~df["Invoice"].astype(str).str.startswith("C")].copy()
    # remove transactions with negative or zero quantity
    df = df[df['Quantity'] > 0]
    df = df[needed_cols].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Floor to the hour and count number of unique invoices per hour
    df["timestamp_local"] = df["InvoiceDate"].dt.floor("h")
    hourly = (
        df.groupby(["timestamp_local"])["Invoice"]
        .nunique()
        .reset_index(name="demand_count")
    )

    # Regularize index to hourly grid
    full_idx = pd.date_range(hourly["timestamp_local"].min(),
                             hourly["timestamp_local"].max(),
                             freq="h")
    hourly = hourly.set_index("timestamp_local").reindex(full_idx)
    hourly.index.name = "timestamp_local"
    hourly["demand_count"] = hourly["demand_count"].fillna(0).astype(int)

    # Attach single store_id (Week 1 MVP)
    hourly["store_id"] = "S001"
    hourly["dow"] = hourly.index.dayofweek
    hourly["hour"] = hourly.index.hour
    hourly["doy"] = hourly.index.dayofyear
    # Simple holiday proxy: mark Nov–Dec as 1 (holiday season)
    hourly["is_holiday"] = hourly.index.month.isin([11, 12]).astype(int)

    hourly = hourly.reset_index().rename(columns={"index":"timestamp_local"})

    PROCESSED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(PROCESSED_OUT_PATH, index=False)

    print(f"Wrote {PROCESSED_OUT_PATH} with {len(hourly):,} hourly rows "
          f"from {hourly['timestamp_local'].min()} to {hourly['timestamp_local'].max()}.")

if __name__ == "__main__":
    main()
