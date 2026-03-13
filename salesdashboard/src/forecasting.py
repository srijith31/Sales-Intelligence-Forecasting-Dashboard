from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class ForecastResult:
    history: pd.DataFrame
    forecast: pd.DataFrame
    metrics: dict[str, float]
    error: str | None = None


@dataclass(slots=True)
class ModelComparisonResult:
    comparison_df: pd.DataFrame
    best_model: str
    best_accuracy: float
    future_frame: pd.DataFrame
    history: pd.DataFrame
    error: str | None = None


def _build_time_features(dates: pd.Series, start_index: int = 0) -> pd.DataFrame:
    frame = pd.DataFrame({"order_date": pd.to_datetime(dates)})
    time_index = np.arange(start_index, start_index + len(frame))
    month = frame["order_date"].dt.month

    features = pd.DataFrame(
        {
            "time_index": time_index,
            "year": frame["order_date"].dt.year,
            "quarter": frame["order_date"].dt.quarter,
            "month": month,
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        }
    )
    return features


def _add_lag_features(df: pd.DataFrame, sales_col: str = "sales") -> pd.DataFrame:
    out = df.copy()
    out["lag_1"] = out[sales_col].shift(1)
    out["lag_2"] = out[sales_col].shift(2)
    out["rolling_3"] = out[sales_col].shift(1).rolling(3, min_periods=1).mean()
    out = out.dropna(subset=["lag_1"])
    out["lag_2"] = out["lag_2"].fillna(out["lag_1"])
    out["rolling_3"] = out["rolling_3"].fillna(out["lag_1"])
    return out


def train_sales_forecast_model(df: pd.DataFrame, forecast_periods: int = 6) -> ForecastResult:
    monthly = (
        df.groupby(pd.Grouper(key="order_date", freq="MS"))
        .agg(sales=("sales", "sum"), quantity=("quantity", "sum"))
        .reset_index()
        .dropna(subset=["order_date"])
    )

    if len(monthly) < 12:
        return ForecastResult(
            history=monthly,
            forecast=pd.DataFrame(),
            metrics={},
            error="At least 12 months of history are recommended for forecasting.",
        )

    time_features = _build_time_features(monthly["order_date"])
    quantity_model = LinearRegression()
    quantity_model.fit(time_features[["time_index"]], monthly["quantity"])

    feature_frame = time_features.copy()
    feature_frame["quantity"] = monthly["quantity"].values
    target = monthly["sales"].values

    split_index = max(8, int(len(monthly) * 0.8))
    split_index = min(split_index, len(monthly) - 2)

    X_train = feature_frame.iloc[:split_index]
    X_test = feature_frame.iloc[split_index:]
    y_train = target[:split_index]
    y_test = target[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    full_predictions = model.predict(feature_frame)

    future_dates = pd.date_range(
        monthly["order_date"].max() + pd.offsets.MonthBegin(1),
        periods=forecast_periods,
        freq="MS",
    )
    future_features = _build_time_features(future_dates.to_series(index=None), start_index=len(monthly))
    future_quantities = quantity_model.predict(future_features[["time_index"]])
    future_features["quantity"] = np.clip(future_quantities, a_min=1, a_max=None)
    future_sales = model.predict(future_features)

    history = monthly.copy()
    history["predicted_sales"] = full_predictions
    history["series"] = "Historical"

    forecast = pd.DataFrame(
        {
            "order_date": future_dates,
            "predicted_sales": np.clip(future_sales, a_min=0, a_max=None),
            "predicted_quantity": np.round(np.clip(future_quantities, a_min=1, a_max=None)),
            "series": "Forecast",
        }
    )

    metrics = {
        "mae": float(mean_absolute_error(y_test, test_predictions)),
        "r2": float(r2_score(y_test, test_predictions)),
        "average_forecast_sales": float(forecast["predicted_sales"].mean()),
    }

    return ForecastResult(history=history, forecast=forecast, metrics=metrics)


def compare_forecast_models(df: pd.DataFrame, forecast_periods: int = 6) -> ModelComparisonResult:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return ModelComparisonResult(
            comparison_df=pd.DataFrame(),
            best_model="",
            best_accuracy=0.0,
            future_frame=pd.DataFrame(),
            history=pd.DataFrame(),
            error="XGBoost is not installed. Run: pip install xgboost",
        )

    monthly = (
        df.groupby(pd.Grouper(key="order_date", freq="MS"))
        .agg(sales=("sales", "sum"), quantity=("quantity", "sum"))
        .reset_index()
        .dropna(subset=["order_date"])
    )

    if len(monthly) < 12:
        return ModelComparisonResult(
            comparison_df=pd.DataFrame(),
            best_model="",
            best_accuracy=0.0,
            future_frame=pd.DataFrame(),
            history=monthly,
            error="At least 12 months of history are required for model comparison.",
        )

    time_features = _build_time_features(monthly["order_date"])
    monthly_feat = pd.concat([monthly.reset_index(drop=True), time_features.reset_index(drop=True)], axis=1)
    monthly_feat = _add_lag_features(monthly_feat, sales_col="sales")

    feature_cols = ["time_index", "year", "quarter", "month", "month_sin", "month_cos", "quantity", "lag_1", "lag_2", "rolling_3"]
    X_all = monthly_feat[feature_cols].values
    y_all = monthly_feat["sales"].values

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    models: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=0),
    }

    records = []
    fitted_models: dict[str, object] = {}

    mean_actual = float(np.mean(y_all)) if len(y_all) > 0 else 1.0

    for model_name, model in models.items():
        model.fit(X_all_scaled, y_all)
        in_sample_preds = model.predict(X_all_scaled)
        mae = float(mean_absolute_error(y_all, in_sample_preds))
        accuracy = max(0.0, min(100.0, (1.0 - mae / mean_actual) * 100)) if mean_actual > 0 else 0.0
        records.append({
            "Model": model_name,
            "Accuracy (%)": round(accuracy, 1),
            "Avg. Prediction Error ($)": round(mae, 2),
        })
        fitted_models[model_name] = model

    comparison_df = pd.DataFrame(records).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    best_model_name = str(comparison_df.iloc[0]["Model"])
    best_accuracy = float(comparison_df.iloc[0]["Accuracy (%)"])
    best_model = fitted_models[best_model_name]

    history = monthly_feat[["order_date", "sales", "quantity"]].copy()
    history["predicted_sales"] = best_model.predict(X_all_scaled)

    # autoregressive future forecast
    future_dates = pd.date_range(
        monthly["order_date"].max() + pd.offsets.MonthBegin(1),
        periods=forecast_periods,
        freq="MS",
    )

    quantity_model = LinearRegression()
    time_features_full = _build_time_features(monthly["order_date"])
    quantity_model.fit(time_features_full[["time_index"]], monthly["quantity"])

    known_sales = list(monthly["sales"].values)
    future_preds = []

    for i, fdate in enumerate(future_dates):
        fi = len(monthly) + i
        ft = _build_time_features(pd.Series([fdate]), start_index=fi)
        fq = float(quantity_model.predict(ft[["time_index"]])[0])
        lag1 = known_sales[-1]
        lag2 = known_sales[-2] if len(known_sales) >= 2 else lag1
        roll3 = float(np.mean(known_sales[-3:])) if len(known_sales) >= 3 else float(np.mean(known_sales))
        row = np.array([[fi, fdate.year, (fdate.month - 1) // 3 + 1, fdate.month,
                         np.sin(2 * np.pi * fdate.month / 12),
                         np.cos(2 * np.pi * fdate.month / 12),
                         max(1, fq), lag1, lag2, roll3]])
        row_scaled = scaler.transform(row)
        pred = float(best_model.predict(row_scaled)[0])
        pred = max(0.0, pred)
        future_preds.append(pred)
        known_sales.append(pred)

    future_frame = pd.DataFrame(
        {
            "order_date": future_dates,
            "predicted_sales": future_preds,
            "series": "Forecast",
        }
    )

    return ModelComparisonResult(
        comparison_df=comparison_df,
        best_model=best_model_name,
        best_accuracy=best_accuracy,
        future_frame=future_frame,
        history=history,
    )
