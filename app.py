import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import holidays
import joblib
import uvicorn

app = FastAPI(
    title="Sales Forecasting API"
)

model = joblib.load("best_model.pkl")

df = pd.read_excel(
    r"D:\Time Series Forecasting System with API\forecasting.xlsx"
)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

class ForecastRequest(BaseModel):
    days: int = 7

@app.get("/")
def home():

    return {
        "message": "Sales Forecasting API Running"
    }

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

@app.post("/forecast")
def forecast(request: ForecastRequest):

    temp_df = df.copy()

    results = []

    us_holidays = holidays.US()

    for i in range(request.days):

        latest_total = temp_df["Total"].iloc[-1]

        lag_1 = latest_total

        lag_7 = (
            temp_df["Total"].iloc[-7]
            if len(temp_df) >= 7
            else latest_total
        )

        lag_30 = (
            temp_df["Total"].iloc[-30]
            if len(temp_df) >= 30
            else latest_total
        )

        rolling_mean_7 = (
            temp_df["Total"]
            .tail(7)
            .mean()
        )

        rolling_mean_30 = (
            temp_df["Total"]
            .tail(30)
            .mean()
        )

        rolling_std_7 = (
            temp_df["Total"]
            .tail(7)
            .std()
        )

        rolling_std_30 = (
            temp_df["Total"]
            .tail(30)
            .std()
        )

        future_date = (
            temp_df["Date"].max()
            + pd.Timedelta(days=1)
        )

        day_of_week = future_date.dayofweek

        month = future_date.month

        week_of_year = (
            future_date.isocalendar().week
        )

        quarter = future_date.quarter

        is_weekend = (
            1 if day_of_week >= 5 else 0
        )

        is_holiday = (
            1 if future_date in us_holidays else 0
        )

        features = pd.DataFrame({

            "lag_1": [lag_1],
            "lag_7": [lag_7],
            "lag_30": [lag_30],

            "rolling_mean_7": [rolling_mean_7],
            "rolling_mean_30": [rolling_mean_30],

            "rolling_std_7": [rolling_std_7],
            "rolling_std_30": [rolling_std_30],

            "day_of_week": [day_of_week],
            "month": [month],
            "week_of_year": [week_of_year],
            "quarter": [quarter],

            "is_weekend": [is_weekend],
            "is_holiday": [is_holiday]

        })

        prediction = model.predict(features)[0]

        results.append({

            "date": str(future_date.date()),

            "predicted_sales": round(
                float(prediction),
                2
            )

        })

        new_row = pd.DataFrame({

            "Date": [future_date],
            "Total": [prediction]

        })

        temp_df = pd.concat(
            [temp_df, new_row],
            ignore_index=True
        )

    return {

        "model_used": "XGBoost",

        "forecast_days": request.days,

        "predictions": results

    }

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )