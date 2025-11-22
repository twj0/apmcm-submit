# Reshoring Ml Models

```json
{
  "models": {
    "random_forest": {
      "rmse": 0.4472466670482285,
      "mae": 0.4363818367956398,
      "r2": -1.1951757639536273
    },
    "gradient_boosting": {
      "rmse": 0.3237026954211205,
      "mae": 0.23861977988380811,
      "r2": -0.1499202051144648
    }
  },
  "feature_importance": {
    "random_forest": {
      "sp500_index": 0.27665889184433506,
      "tariff_index_total": 0.45263916667940723,
      "tariff_retaliation_interaction": 0.2707019414762577
    },
    "gradient_boosting": {
      "sp500_index": 0.4343707833423996,
      "tariff_index_total": 0.2570664453827458,
      "tariff_retaliation_interaction": 0.3085627712748547
    }
  },
  "top_features": {
    "random_forest": [
      {
        "feature": "tariff_index_total",
        "importance": 0.45263916667940723
      },
      {
        "feature": "sp500_index",
        "importance": 0.27665889184433506
      },
      {
        "feature": "tariff_retaliation_interaction",
        "importance": 0.2707019414762577
      }
    ],
    "gradient_boosting": [
      {
        "feature": "sp500_index",
        "importance": 0.4343707833423996
      },
      {
        "feature": "tariff_retaliation_interaction",
        "importance": 0.3085627712748547
      },
      {
        "feature": "tariff_index_total",
        "importance": 0.2570664453827458
      }
    ]
  }
}
```
