import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }

def train_test_split_last_years(df: pd.DataFrame, year_col: str, test_years: int = 3):
    years = sorted(df[year_col].dropna().astype(int).unique())
    test_set = years[-test_years:]
    test_mask = df[year_col].astype(int).isin(test_set)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy(), test_set

def build_models(random_state: int = 42):
    return {
        "LinearRegression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.05, max_iter=10000))]),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=random_state),
    }

def compare_models(df: pd.DataFrame, feature_cols: list, target_col: str, year_col: str = "Ano", test_years: int = 3):
    data = df[[year_col, target_col] + feature_cols].dropna().sort_values(year_col)
    train_df, test_df, test_set = train_test_split_last_years(data, year_col, test_years=test_years)

    X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
    X_test, y_test = test_df[feature_cols].values, test_df[target_col].values

    models = build_models()
    rows = []
    preds = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        preds[name] = y_pred
        m = evaluate(y_test, y_pred)
        rows.append({"Modelo": name, **m})
    results = pd.DataFrame(rows).sort_values("RMSE")
    return results, test_df[[year_col, target_col]].reset_index(drop=True), preds, test_set

def time_series_cv_rmse(df: pd.DataFrame, feature_cols: list, target_col: str, year_col: str = "Ano", n_splits: int = 4):
    data = df[[year_col, target_col] + feature_cols].dropna().sort_values(year_col)
    X, y = data[feature_cols].values, data[target_col].values
    tscv = TimeSeriesSplit(n_splits=n_splits)

    models = build_models()
    rows = []
    for name, mdl in models.items():
        rmses = []
        for tr, te in tscv.split(X):
            mdl.fit(X[tr], y[tr])
            pred = mdl.predict(X[te])
            rmses.append(rmse(y[te], pred))
        rows.append({"Modelo": name, "RMSE_medio_CV": float(np.mean(rmses)), "RMSE_std_CV": float(np.std(rmses))})
    return pd.DataFrame(rows).sort_values("RMSE_medio_CV")
