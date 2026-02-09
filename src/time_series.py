import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def train_test_split_time(df: pd.DataFrame, year_col="Ano", target="despesa_total", test_years: int = 3):
    d = df.sort_values(year_col).copy()
    years = sorted(d[year_col].dropna().unique())
    if len(years) <= test_years + 2:
        raise ValueError("Série muito curta para split temporal com test_years=%s" % test_years)
    cut = years[-test_years]
    train = d[d[year_col] < cut].copy()
    test = d[d[year_col] >= cut].copy()
    y_train = train[target].astype(float).values
    y_test = test[target].astype(float).values
    return train, test, y_train, y_test

def forecast_arima(y_train, steps: int, order=(1,1,1)):
    model = ARIMA(y_train, order=order)
    fit = model.fit()
    fc = fit.forecast(steps=steps)
    return np.asarray(fc), fit

def forecast_ets(y_train, steps: int, trend="add", seasonal=None, seasonal_periods=None):
    model = ExponentialSmoothing(
        y_train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )
    fit = model.fit(optimized=True)
    fc = fit.forecast(steps)
    return np.asarray(fc), fit
