# End-to-End Time Series Forecasting System

## Project Overview

This project is a complete Time Series Forecasting backend system developed using Python and FastAPI.

The system trains multiple forecasting models, compares their performance, automatically selects the best model, and serves predictions through a REST API.

---

## Models Used

- ARIMA
- Prophet
- XGBoost
- LSTM

---

## Features

- Time series forecasting
- Feature engineering
- Lag features
- Rolling statistics
- Holiday handling
- Model comparison
- Automatic best model selection
- REST API using FastAPI
- Forecast endpoint

---

## Technologies Used

- Python
- Pandas
- NumPy
- XGBoost
- TensorFlow
- Prophet
- Statsmodels
- FastAPI
- Scikit-learn

---

## Run Training

```bash
python train.py
```

---

## Run API

```bash
python app.py
```

---

## API Documentation

```text
http://127.0.0.1:8000/docs
```

---

## Forecast API Example

### Request

```json
{
  "days": 7
}
```

### Response

```json
{
  "model_used": "XGBoost",
  "forecast_days": 7,
  "predictions": [
    {
      "date": "2026-05-10",
      "predicted_sales": 523118293.44
    }
  ]
}
```
