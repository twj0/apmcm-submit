# Q1 Forecast Model Comparison

This report compares a simple statistical baseline against the LSTM model.

## Metrics by target variable

| Target | Model | MAE | RMSE | MAPE (%) | sMAPE (%) |
|--------|-------|-----|------|----------|-----------|
| import_quantity | Statistical (lag-1 baseline) | 909176.18 | 1379610.04 | 55558952780.57 | 62.42 |
| import_quantity | LSTM | 1790650.25 | 2522518.99 | 244.32 | 103.46 |
| unit_price | Statistical (lag-1 baseline) | 570.92 | 4040.67 | 110271017.32 | 8.70 |
| unit_price | LSTM | 101.97 | 117.41 | 18.93 | 19.18 |