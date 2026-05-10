import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import holidays
import joblib

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

df = pd.read_excel(
    r"D:\Time Series Forecasting System with API\forecasting.xlsx"
)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df[["Date", "Total"]]

df["lag_1"] = df["Total"].shift(1)
df["lag_7"] = df["Total"].shift(7)
df["lag_30"] = df["Total"].shift(30)

df["rolling_mean_7"] = (
    df["Total"].rolling(window=7).mean()
)

df["rolling_mean_30"] = (
    df["Total"].rolling(window=30).mean()
)

df["rolling_std_7"] = (
    df["Total"].rolling(window=7).std()
)

df["rolling_std_30"] = (
    df["Total"].rolling(window=30).std()
)

df["day_of_week"] = df["Date"].dt.dayofweek

df["month"] = df["Date"].dt.month

df["week_of_year"] = (
    df["Date"].dt.isocalendar().week.astype(int)
)

df["quarter"] = df["Date"].dt.quarter

df["is_weekend"] = df["day_of_week"].apply(
    lambda x: 1 if x >= 5 else 0
)

us_holidays = holidays.US()

df["is_holiday"] = df["Date"].apply(
    lambda x: 1 if x in us_holidays else 0
)

df = df.dropna()

split = int(len(df) * 0.8)

train_df = df[:split]
test_df = df[split:]

def evaluate_model(y_true, y_pred):

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return rmse, mae, r2

print("\nTRAINING ARIMA MODEL...")

arima_model = ARIMA(
    train_df["Total"],
    order=(5,1,2)
)

arima_model_fit = arima_model.fit()

arima_predictions = arima_model_fit.forecast(
    steps=len(test_df)
)

arima_rmse, arima_mae, arima_r2 = evaluate_model(
    test_df["Total"],
    arima_predictions
)

print("\nTRAINING PROPHET MODEL...")

prophet_df = df[["Date", "Total"]].rename(
    columns={
        "Date": "ds",
        "Total": "y"
    }
)

train_prophet = prophet_df[:split]
test_prophet = prophet_df[split:]

prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

prophet_model.add_country_holidays(
    country_name="US"
)

prophet_model.fit(train_prophet)

future = prophet_model.make_future_dataframe(
    periods=len(test_prophet),
    freq="D"
)

forecast = prophet_model.predict(future)

prophet_predictions = (
    forecast["yhat"]
    .tail(len(test_prophet))
    .values
)

prophet_rmse, prophet_mae, prophet_r2 = evaluate_model(
    test_prophet["y"],
    prophet_predictions
)

print("\nTRAINING XGBOOST MODEL...")

X = df.drop(columns=["Date", "Total"])

y = df["Total"]

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

xgb_rmse, xgb_mae, xgb_r2 = evaluate_model(
    y_test,
    xgb_predictions
)

print("\nTRAINING LSTM MODEL...")

scaler = MinMaxScaler()

scaled_total = scaler.fit_transform(
    df["Total"].values.reshape(-1,1)
)

X_lstm = []
y_lstm = []

sequence_length = 30

for i in range(sequence_length, len(scaled_total)):

    X_lstm.append(
        scaled_total[i-sequence_length:i]
    )

    y_lstm.append(
        scaled_total[i]
    )

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

split_lstm = int(len(X_lstm) * 0.8)

X_train_lstm = X_lstm[:split_lstm]
X_test_lstm = X_lstm[split_lstm:]

y_train_lstm = y_lstm[:split_lstm]
y_test_lstm = y_lstm[split_lstm:]

lstm_model = Sequential()

lstm_model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            X_train_lstm.shape[1],
            X_train_lstm.shape[2]
        )
    )
)

lstm_model.add(Dropout(0.2))

lstm_model.add(LSTM(32))

lstm_model.add(Dense(1))

lstm_model.compile(
    optimizer="adam",
    loss="mse"
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

lstm_model.fit(
    X_train_lstm,
    y_train_lstm,
    validation_data=(
        X_test_lstm,
        y_test_lstm
    ),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

lstm_predictions = lstm_model.predict(
    X_test_lstm
)

lstm_predictions = scaler.inverse_transform(
    lstm_predictions
)

y_test_actual = scaler.inverse_transform(
    y_test_lstm.reshape(-1,1)
)

lstm_rmse, lstm_mae, lstm_r2 = evaluate_model(
    y_test_actual,
    lstm_predictions
)

results = pd.DataFrame({

    "Model": [
        "ARIMA",
        "PROPHET",
        "XGBOOST",
        "LSTM"
    ],

    "RMSE": [
        arima_rmse,
        prophet_rmse,
        xgb_rmse,
        lstm_rmse
    ],

    "MAE": [
        arima_mae,
        prophet_mae,
        xgb_mae,
        lstm_mae
    ],

    "R2 Score": [
        arima_r2,
        prophet_r2,
        xgb_r2,
        lstm_r2
    ]
})

results["RMSE"] = results["RMSE"].round(2)
results["MAE"] = results["MAE"].round(2)
results["R2 Score"] = results["R2 Score"].round(2)

results = results.sort_values(
    by="RMSE"
)

print("\nFINAL MODEL COMPARISON")
print("-" * 50)

print(results)

joblib.dump(
    xgb_model,
    "best_model.pkl"
)

print("\nMODEL SAVED SUCCESSFULLY")