# Model Comparison

```json
{
  "econometric_models": {
    "VAR": {
      "aic": -5.493823859532288,
      "bic": -4.888653673544196,
      "lag_order": 1
    },
    "OLS_Regressions": {
      "num_models": 4,
      "avg_rsquared": 0.25877557735830137
    }
  },
  "ml_models": {
    "VAR_LSTM": {
      "rmse": 0.7077031416838759,
      "training_samples": 7
    },
    "Reshoring_ML": {
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
    }
  },
  "summary": {
    "total_econometric_models": 2,
    "total_ml_models": 2,
    "hybrid_approach": "VAR + LSTM for impulse responses, RF/GB for reshoring prediction",
    "recommendation": "Use VAR for interpretability, ML for prediction accuracy"
  }
}
```
