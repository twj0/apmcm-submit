# Q5 Econometric vs ML Model Comparison

High-level comparison between traditional econometric models and ML enhancements.

## Summary Table

| Component | Model | Key Metric | Value | Notes |
|-----------|-------|------------|-------|-------|
| Macro relationships | OLS regressions | Avg R² | 0.259 | 4 regressions |
| Joint dynamics | VAR | Stable | False | lag/order from var_results |
| Dynamic residuals | VAR-LSTM | RMSE | 0.708 | samples=7 |
| Reshoring | RF | Test R² | -7.699 | from reshoring_ml_models.json |
| Reshoring | GB | Test R² | -5.283 | from reshoring_ml_models.json |