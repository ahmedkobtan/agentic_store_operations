import numpy as np
import pandas as pd
from prophet import Prophet
from walmart_ahmedkobtan_agentic_store_operations.src.utils.constants import OPEN_HOUR, CLOSE_HOUR

def _smape(y_true, y_pred, eps=1e-6):
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return 100 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def _mape_safe(y_true, y_pred, min_actual=5.0):
    y = np.maximum(y_true, min_actual)
    return 100 * np.mean(np.abs((y_pred - y) / y))

def rolling_backtest(df_hourly, train_weeks=8, test_weeks=1,  open_hour=OPEN_HOUR, close_hour=CLOSE_HOUR):
    df = df_hourly.copy()
    h = df["timestamp_local"].dt.hour
    df = df[(h >= open_hour) & (h < close_hour)]

    df_p = df.rename(columns={"timestamp_local": "ds", "demand_count": "y"})

    df_p = df_p.sort_values("ds")
    H = test_weeks * 7 * (close_hour - open_hour)  # number of open hours in test week
    step = H

    metrics = []
    for start in range(0, len(df_p) - (train_weeks + test_weeks)*7*(close_hour - open_hour), step):
        train = df_p.iloc[start:start + train_weeks*7*(close_hour - open_hour)].copy()
        test = df_p.iloc[start + train_weeks*7*(close_hour - open_hour):start + (train_weeks + test_weeks)*7*(close_hour - open_hour)].copy()
        m = Prophet(
                    seasonality_mode="multiplicative",
                    weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=True,
                    changepoint_prior_scale=0.5, interval_width=0.8
        )
        # log1p like in fit
        train["y"] = np.log1p(train["y"])
        m.fit(train)

        # future = m.make_future_dataframe(periods=H, freq="H")
        fut = pd.DataFrame({"ds": test["ds"].values})
        fcst = m.predict(fut)

        # invert log1p
        pred = np.expm1(fcst["yhat"].values)
        actual = test["y"].values

        # Calculate MAPE and sMAPE
        mape = _mape_safe(y_true=actual, y_pred=pred, min_actual=5.0)
        smape = _smape(y_true=actual, y_pred=pred)

        metrics.append({"window_start": test["ds"].min(), "MAPE": mape, "sMAPE": smape})
    return metrics
