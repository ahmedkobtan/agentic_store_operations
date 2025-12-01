import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import PROPHET_MODEL_PATH, OPEN_HOUR, CLOSE_HOUR


def _filter_open_hours(df):
    df = df.copy()
    h = df["timestamp_local"].dt.hour
    return df[(h >= OPEN_HOUR) & (h < CLOSE_HOUR)].copy()

def fit_baseline_prophet(df_hourly):
    df_hourly = _filter_open_hours(df_hourly)
    # Prophet expects columns: ds (datetime), y (target)
    df = df_hourly.rename(columns={"timestamp_local": "ds", "demand_count": "y"})
    # Stabilize: log1p transform helps multiplicative patterns
    df["y"] = np.log1p(df["y"])

    # Prophet can handle hourly if freq='H'
    m = Prophet(
        seasonality_mode="multiplicative", # instead of additive
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.5,   # a bit more flexibility
        interval_width=0.8,
    )

    # Slightly richer seasonality for hourly data
    m.add_seasonality(name="daily_extra", period=1, fourier_order=10)
    m.add_seasonality(name="weekly_extra", period=7, fourier_order=6)

    m.fit(df)

    PROPHET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, PROPHET_MODEL_PATH)
    return m

def forecast(df_hourly, horizon_hours=168):
    m = joblib.load(PROPHET_MODEL_PATH)
    # Create future only for open hours
    last_ts = df_hourly["timestamp_local"].max()
    future = pd.date_range(last_ts + pd.Timedelta(hours=1),
                           last_ts + pd.Timedelta(hours=horizon_hours),
                           freq="H")
    future = future[(future.hour >= OPEN_HOUR) & (future.hour < CLOSE_HOUR)]
    fut = pd.DataFrame({"ds": future})
    fcst = m.predict(fut)
    # invert log1p
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        fcst[c] = np.expm1(fcst[c])
    # Keep only needed columns
    return fcst.rename(columns={
        "ds": "timestamp_local",
        "yhat_lower": "yhat_p10",
        "yhat": "yhat_p50",
        "yhat_upper": "yhat_p90"
    })[["timestamp_local","yhat_p10","yhat_p50","yhat_p90"]]
