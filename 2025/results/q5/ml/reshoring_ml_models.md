# Reshoring Ml Models

```json
{
  "models": {
    "random_forest": {
      "rmse": 0.8903463021612373,
      "mae": 0.7288548892439195,
      "r2": -7.699473953239055
    },
    "gradient_boosting": {
      "rmse": 0.7566517341374105,
      "mae": 0.5355769113543829,
      "r2": -5.28300111873619
    }
  },
  "feature_importance": {
    "random_forest": {
      "sp500_index": 0.1258457573798114,
      "tariff_index_total_lag1": 0.21471746048879367,
      "dollar_index": 0.659436782131395
    },
    "gradient_boosting": {
      "sp500_index": 0.16323754146639322,
      "tariff_index_total_lag1": 0.13692287344993684,
      "dollar_index": 0.6998395850836701
    }
  },
  "top_features": {
    "random_forest": [
      {
        "feature": "dollar_index",
        "importance": 0.659436782131395
      },
      {
        "feature": "tariff_index_total_lag1",
        "importance": 0.21471746048879367
      },
      {
        "feature": "sp500_index",
        "importance": 0.1258457573798114
      }
    ],
    "gradient_boosting": [
      {
        "feature": "dollar_index",
        "importance": 0.6998395850836701
      },
      {
        "feature": "sp500_index",
        "importance": 0.16323754146639322
      },
      {
        "feature": "tariff_index_total_lag1",
        "importance": 0.13692287344993684
      }
    ]
  }
}
```
