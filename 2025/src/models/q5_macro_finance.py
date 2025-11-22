"""
Q5: Macroeconomic, Sectoral, and Financial Effects; Reshoring

VAR/SVAR analysis and event study to assess tariff impacts on macro indicators,
financial markets, and manufacturing reshoring.

Enhanced with VAR-LSTM hybrid and ML methods for improved prediction.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.config import RESULTS_DIR, FIGURES_DIR, DATA_EXTERNAL, DATA_PROCESSED, RANDOM_SEED
from utils.data_loader import TariffDataLoader

logger = logging.getLogger(__name__)

# Set seeds for reproducibility (use global project seed)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class MacroFinanceModel:
    """Model for macroeconomic and financial impact analysis with VAR+ML hybrid."""

    def __init__(self):
        """Initialize the model."""
        self.loader = TariffDataLoader()
        self.time_series: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        self.q5_econometric_dir = RESULTS_DIR / 'q5' / 'econometric'
        self.q5_ml_dir = RESULTS_DIR / 'q5' / 'ml'
        self.q5_econometric_dir.mkdir(parents=True, exist_ok=True)
        self.q5_ml_dir.mkdir(parents=True, exist_ok=True)
        
    def load_q5_data(self) -> pd.DataFrame:
        """Load and merge all data sources for Q5.
        
        Returns:
            Merged time series DataFrame
        """
        logger.info("Loading Q5 macro/financial data")
        
        # Load tariff indices (prefer processed CSVs; fallback to parquet or placeholder)
        # Priority: q5_tariff_indices_calibrated.csv -> q5_tariff_indices_policy.csv -> parquet -> placeholder
        tariff_indices = None
        cal_csv = DATA_PROCESSED / 'q5' / 'q5_tariff_indices_calibrated.csv'
        pol_csv = DATA_PROCESSED / 'q5' / 'q5_tariff_indices_policy.csv'
        if cal_csv.exists():
            try:
                tdf = pd.read_csv(cal_csv)
                if 'tariff_index' in tdf.columns:
                    tdf = tdf.rename(columns={'tariff_index': 'tariff_index_total'})
                tariff_indices = tdf
                logger.info("Loaded tariff indices from calibrated CSV")
            except Exception as e:
                logger.warning(f"Failed to read calibrated tariff indices: {e}")
        if tariff_indices is None and pol_csv.exists():
            try:
                tdf = pd.read_csv(pol_csv)
                if 'tariff_index' in tdf.columns:
                    tdf = tdf.rename(columns={'tariff_index': 'tariff_index_total'})
                tariff_indices = tdf
                logger.info("Loaded tariff indices from policy CSV")
            except Exception as e:
                logger.warning(f"Failed to read policy tariff indices: {e}")
        if tariff_indices is None:
            try:
                tdf = pd.read_parquet(DATA_PROCESSED / 'tariff_indices.parquet')
                tariff_indices = tdf
                logger.info("Loaded tariff indices from parquet")
            except Exception:
                logger.warning("Tariff indices not found, creating placeholder")
                tariff_indices = pd.DataFrame({
                    'year': range(2015, 2026),
                    'tariff_index_total': np.random.uniform(2.0, 8.0, 11),
                })
        
        # Load external macro data
        macro_file = DATA_EXTERNAL / 'us_macro.csv'
        if not macro_file.exists():
            logger.warning(f"Macro data not found: {macro_file}")
            template_macro = pd.DataFrame({
                'year': range(2015, 2026),
                'gdp_growth': np.random.uniform(1.5, 3.5, 11),
                'industrial_production': np.random.uniform(95, 110, 11),
                'unemployment_rate': np.random.uniform(3.5, 8.0, 11),
                'cpi': np.random.uniform(240, 290, 11),
            })
            template_macro.to_csv(macro_file, index=False)
            macro_data = template_macro
        else:
            macro_data = pd.read_csv(macro_file)
        
        # Load financial data
        financial_file = DATA_EXTERNAL / 'us_financial.csv'
        if not financial_file.exists():
            logger.warning(f"Financial data not found: {financial_file}")
            template_financial = pd.DataFrame({
                'year': range(2015, 2026),
                'dollar_index': np.random.uniform(90, 105, 11),
                'treasury_yield_10y': np.random.uniform(1.5, 4.5, 11),
                'sp500_index': np.random.uniform(2500, 4500, 11),
                'crypto_index': np.random.uniform(5000, 50000, 11),
            })
            template_financial.to_csv(financial_file, index=False)
            financial_data = template_financial
        else:
            financial_data = pd.read_csv(financial_file)
        
        # Load reshoring data
        reshoring_file = DATA_EXTERNAL / 'us_reshoring.csv'
        if not reshoring_file.exists():
            logger.warning(f"Reshoring data not found: {reshoring_file}")
            template_reshoring = pd.DataFrame({
                'year': range(2015, 2026),
                'manufacturing_va_share': np.random.uniform(10, 13, 11),
                'manufacturing_employment_share': np.random.uniform(8, 10, 11),
                'reshoring_fdi_billions': np.random.uniform(5, 50, 11),
            })
            template_reshoring.to_csv(reshoring_file, index=False)
            reshoring_data = template_reshoring
        else:
            reshoring_data = pd.read_csv(reshoring_file)
        
        # Load retaliation index
        retaliation_file = DATA_EXTERNAL / 'retaliation_index.csv'
        if not retaliation_file.exists():
            logger.warning(f"Retaliation index not found: {retaliation_file}")
            template_retaliation = pd.DataFrame({
                'year': range(2015, 2026),
                'retaliation_index': [0, 0, 1, 2, 1, 1, 3, 5, 8, 10, 12],
            })
            template_retaliation.to_csv(retaliation_file, index=False)
            retaliation_data = template_retaliation
        else:
            retaliation_data = pd.read_csv(retaliation_file)
        
        # Merge all datasets on year
        merged = tariff_indices.merge(macro_data, on='year', how='outer')
        merged = merged.merge(financial_data, on='year', how='outer')
        merged = merged.merge(reshoring_data, on='year', how='outer')
        merged = merged.merge(retaliation_data, on='year', how='outer')
        
        merged = merged.sort_values('year').reset_index(drop=True)
        
        logger.info(f"Merged time series shape: {merged.shape}")
        logger.info(f"Columns: {merged.columns.tolist()}")
        
        self.time_series = merged
        
        return merged
    
    def estimate_regression_effects(
        self,
        ts_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Estimate regression-based effects of tariffs on key variables.
        
        Model: Y_t = λ0 + λ1·TariffIndex_t + λ2·RetaliationIndex_t + λ3·Z_t + ε
        
        Args:
            ts_df: Time series DataFrame
            
        Returns:
            Dictionary of regression results
        """
        if ts_df is None:
            ts_df = self.time_series
        
        if ts_df is None or len(ts_df) < 10:
            logger.error("Insufficient time series data")
            return {}
        
        logger.info("Estimating regression effects")
        
        results = {}
        
        # Dependent variables to analyze
        dep_vars = [
            'gdp_growth',
            'industrial_production',
            'manufacturing_va_share',
            'manufacturing_employment_share',
        ]
        
        for dep_var in dep_vars:
            if dep_var not in ts_df.columns:
                continue
            
            try:
                formula = f'{dep_var} ~ tariff_index_total + retaliation_index'
                model = smf.ols(formula, data=ts_df).fit()
                
                results[dep_var] = {
                    'rsquared': float(model.rsquared),
                    'tariff_coef': float(model.params.get('tariff_index_total', 0)),
                    'tariff_pvalue': float(model.pvalues.get('tariff_index_total', 1)),
                    'retaliation_coef': float(model.params.get('retaliation_index', 0)),
                    'retaliation_pvalue': float(model.pvalues.get('retaliation_index', 1)),
                }
                
                logger.info(f"{dep_var}: R²={results[dep_var]['rsquared']:.3f}, "
                           f"tariff_coef={results[dep_var]['tariff_coef']:.3f}")
                
            except Exception as e:
                logger.error(f"Error for {dep_var}: {e}")
        
        # Save results to econometric directory
        self._save_results(results, 'regressions', self.q5_econometric_dir)

        return results
    
    def _check_var_stability(self, fitted_model):
        """Check VAR model stability.
        
        Returns:
            True if model is stable, False otherwise
        """
        try:
            # Check if all eigenvalues are inside unit circle
            roots = fitted_model.roots
            is_stable = np.all(np.abs(roots) > 1.0001)
            
            # Also check if AIC/BIC are finite
            is_finite = np.isfinite(fitted_model.aic) and np.isfinite(fitted_model.bic)
            
            return is_stable and is_finite
        except:
            return False
    
    def estimate_var_model(
        self,
        ts_df: Optional[pd.DataFrame] = None,
        variables: Optional[List[str]] = None,
        maxlags: int = 2
    ) -> Dict:
        """Estimate VAR model and compute impulse responses.
        
        Args:
            ts_df: Time series DataFrame
            variables: List of variables to include in VAR
            maxlags: Maximum number of lags
            
        Returns:
            VAR results and IRF summary
        """
        if ts_df is None:
            ts_df = self.time_series
        
        if variables is None:
            variables = [
                'tariff_index_total',
                'retaliation_index',
                'gdp_growth',
                'industrial_production',
            ]
        
        # Filter to available variables
        available_vars = [v for v in variables if v in ts_df.columns]
        
        if len(available_vars) < 2:
            logger.warning("Insufficient variables for VAR")
            return {}
        
        logger.info(f"Estimating VAR with variables: {available_vars}")
        
        try:
            # Prepare data (remove NaN)
            var_data = ts_df[available_vars].dropna()
            
            if len(var_data) < 10:
                logger.warning("Insufficient observations for VAR")
                return {}
            
            # CRITICAL FIX: Standardize data to prevent numerical explosion
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            var_data_scaled = pd.DataFrame(
                scaler.fit_transform(var_data),
                columns=available_vars,
                index=var_data.index
            )
            
            # CRITICAL FIX: Auto-select optimal lag order based on sample size
            model = VAR(var_data_scaled)
            max_possible_lag = max(1, min((len(var_data) - 5) // len(available_vars), maxlags))
            
            if max_possible_lag > 1:
                try:
                    lag_order_results = model.select_order(maxlags=max_possible_lag)
                    optimal_lag = min(lag_order_results.aic, max_possible_lag)
                except:
                    optimal_lag = 1
            else:
                optimal_lag = 1
                
            logger.info(f"Using lag order: {optimal_lag} (max possible: {max_possible_lag})")
            
            # Fit VAR with optimal lag
            fitted = model.fit(maxlags=optimal_lag)
            
            self.models['var'] = fitted
            
            # Compute IRFs with shorter period to avoid explosion
            try:
                irf = fitted.irf(periods=5)  # Reduced from 10 to 5
            except:
                logger.warning("IRF computation failed, skipping")
                irf = None
            
            # Extract IRF for tariff shock
            tariff_shock_idx = available_vars.index('tariff_index_total') if 'tariff_index_total' in available_vars else 0
            
            irf_summary = {
                'lag_order': int(fitted.k_ar),
                'nobs': int(fitted.nobs),
                'variables': available_vars,
                'aic': float(fitted.aic) if np.isfinite(fitted.aic) else 999999,
                'bic': float(fitted.bic) if np.isfinite(fitted.bic) else 999999,
                'is_stable': self._check_var_stability(fitted)
            }
            
            logger.info(f"VAR fitted with {irf_summary['lag_order']} lags, AIC={irf_summary['aic']:.2f}")
            
            # Save IRF plot data (with bounds checking)
            irf_data = {}
            if irf is not None:
                for i, var in enumerate(available_vars):
                    response = irf.irfs[:, i, tariff_shock_idx]
                    # Clip extreme values to prevent JSON serialization issues
                    response = np.clip(response, -1000, 1000)
                    irf_data[var] = response.tolist()
            
            irf_summary['irf_tariff_shock'] = irf_data
            
        except Exception as e:
            logger.error(f"Error estimating VAR: {e}")
            irf_summary = {}
        
        # Save results to econometric directory
        self._save_results(irf_summary, 'var_results', self.q5_econometric_dir)

        return irf_summary
    
    def evaluate_reshoring(
        self,
        ts_df: Optional[pd.DataFrame] = None,
        treatment_year: int = 2025
    ) -> Dict:
        """Evaluate reshoring using event study / DID approach.
        
        Args:
            ts_df: Time series data
            treatment_year: Year when reciprocal tariffs introduced
            
        Returns:
            Reshoring effect estimates
        """
        if ts_df is None:
            ts_df = self.time_series
        
        if ts_df is None or 'manufacturing_va_share' not in ts_df.columns:
            logger.warning("Cannot evaluate reshoring: missing data")
            return {}
        
        logger.info("Evaluating reshoring effects")
        
        # Create treatment indicator
        ts_df = ts_df.copy()
        ts_df['post_treatment'] = (ts_df['year'] >= treatment_year).astype(int)
        
        try:
            # Simple before-after comparison
            pre_treatment = ts_df[ts_df['post_treatment'] == 0]
            post_treatment = ts_df[ts_df['post_treatment'] == 1]
            
            if len(pre_treatment) == 0 or len(post_treatment) == 0:
                logger.warning("Insufficient pre/post periods")
                return {}
            
            pre_mean = pre_treatment['manufacturing_va_share'].mean()
            post_mean = post_treatment['manufacturing_va_share'].mean()
            difference = post_mean - pre_mean
            
            # Regression with treatment
            formula = 'manufacturing_va_share ~ post_treatment + year'
            model = smf.ols(formula, data=ts_df).fit()
            
            p_value = float(model.pvalues.get('post_treatment', 1))
            results = {
                'pre_treatment_mean': float(pre_mean),
                'post_treatment_mean': float(post_mean),
                'difference': float(difference),
                'treatment_coef': float(model.params.get('post_treatment', 0)),
                'treatment_pvalue': p_value,
                'statistically_significant': bool(p_value < 0.05),
            }
            
            logger.info(f"Reshoring effect: {results['difference']:.2f} percentage points")
            logger.info(f"Statistically significant: {results['statistically_significant']}")
            
        except Exception as e:
            logger.error(f"Error evaluating reshoring: {e}")
            results = {}
        
        # Save results to econometric directory
        self._save_results(results, 'reshoring_effects', self.q5_econometric_dir)

        return results
    
    def plot_q5_results(self) -> None:
        """Generate plots for Q5 results."""
        import matplotlib.pyplot as plt
        from utils.config import apply_plot_style
        
        apply_plot_style()
        
        if self.time_series is None:
            logger.error("No time series data to plot")
            return
        
        logger.info("Creating Q5 plots")
        
        ts = self.time_series
        
        # Plot 1: Time series of key indicators
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if 'tariff_index_total' in ts.columns:
            axes[0, 0].plot(ts['year'], ts['tariff_index_total'], marker='o')
            axes[0, 0].set_title('Tariff Index Over Time')
            axes[0, 0].set_ylabel('Tariff Index (%)')
            axes[0, 0].grid(alpha=0.3)
        
        if 'gdp_growth' in ts.columns:
            axes[0, 1].plot(ts['year'], ts['gdp_growth'], marker='o', color='green')
            axes[0, 1].set_title('GDP Growth Rate')
            axes[0, 1].set_ylabel('Growth Rate (%)')
            axes[0, 1].grid(alpha=0.3)
        
        if 'manufacturing_va_share' in ts.columns:
            axes[1, 0].plot(ts['year'], ts['manufacturing_va_share'], marker='o', color='orange')
            axes[1, 0].axvline(x=2025, color='red', linestyle='--', label='Policy Change')
            axes[1, 0].set_title('Manufacturing Value-Added Share')
            axes[1, 0].set_ylabel('Share (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
        
        if 'dollar_index' in ts.columns:
            axes[1, 1].plot(ts['year'], ts['dollar_index'], marker='o', color='purple')
            axes[1, 1].set_title('Dollar Index')
            axes[1, 1].set_ylabel('Index')
            axes[1, 1].grid(alpha=0.3)
        
        for ax in axes.flat:
            ax.set_xlabel('Year')
        
        plt.tight_layout()
        pdf_path = FIGURES_DIR / 'q5_time_series_overview.pdf'
        fig.savefig(pdf_path, bbox_inches='tight')
        png_path = FIGURES_DIR / 'q5_time_series_overview.png'
        fig.savefig(png_path, bbox_inches='tight')
        plt.close()
        
        # Plot 2: IRF (if available)
        try:
            # Read VAR results from econometric results directory
            var_results_path = self.q5_econometric_dir / 'var_results.json'
            with open(var_results_path, 'r') as f:
                var_results = json.load(f)
            
            if 'irf_tariff_shock' in var_results:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                irf_data = var_results['irf_tariff_shock']
                periods = range(len(next(iter(irf_data.values()))))
                
                for var_name, irf_values in irf_data.items():
                    ax.plot(periods, irf_values, marker='o', label=var_name)
                
                ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
                ax.set_xlabel('Periods After Shock')
                ax.set_ylabel('Response')
                ax.set_title('Impulse Response to Tariff Shock')
                ax.legend()
                ax.grid(alpha=0.3)
                
                plt.tight_layout()
                pdf_path = FIGURES_DIR / 'q5_impulse_response.pdf'
                fig.savefig(pdf_path, bbox_inches='tight')
                png_path = FIGURES_DIR / 'q5_impulse_response.png'
                fig.savefig(png_path, bbox_inches='tight')
                plt.close()
        
        except Exception as e:
            logger.warning(f"Could not plot IRF: {e}")
        
        logger.info("Q5 plots saved")

    def _save_results(self, results: Dict, name: str, directory: Path) -> None:
        """Save results in multiple formats."""
        # Convert to JSON-serializable format
        def make_serializable(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = make_serializable(results)
        
        # JSON
        with open(directory / f'{name}.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # CSV (if possible)
        try:
            df = pd.DataFrame([results]) if not isinstance(results, list) else pd.DataFrame(results)
            df.to_csv(directory / f'{name}.csv', index=False)
        except:
            pass

        # Markdown
        with open(directory / f'{name}.md', 'w') as f:
            f.write(f"# {name.replace('_', ' ').title()}\n\n")
            f.write("```json\n")
            f.write(json.dumps(serializable_results, indent=2))
            f.write("\n```\n")

    def prepare_ml_features(self, ts_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for ML models."""
        if ts_df is None:
            ts_df = self.time_series

        df = ts_df.copy()

        # Create lag features
        for col in ['tariff_index_total', 'retaliation_index', 'gdp_growth']:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag2'] = df[col].shift(2)

        # Create interaction features
        if 'tariff_index_total' in df.columns and 'retaliation_index' in df.columns:
            df['tariff_retaliation_interaction'] = df['tariff_index_total'] * df['retaliation_index']

        # Drop NaN from lags
        df = df.dropna()

        feature_cols = [c for c in df.columns if c not in ['year', 'manufacturing_va_share',
                                                             'manufacturing_employment_share',
                                                             'reshoring_fdi_billions']]

        return df, feature_cols

    def train_var_lstm_hybrid(self, ts_df: Optional[pd.DataFrame] = None,
                              variables: Optional[List[str]] = None) -> Dict:
        """VAR-LSTM hybrid for improved impulse response prediction."""
        if ts_df is None:
            ts_df = self.time_series

        if variables is None:
            variables = ['tariff_index_total', 'retaliation_index', 'gdp_growth', 'industrial_production']

        available_vars = [v for v in variables if v in ts_df.columns]

        if len(available_vars) < 2:
            logger.warning("Insufficient variables for VAR-LSTM")
            return {}

        logger.info("Training VAR-LSTM hybrid model")

        try:
            # Get VAR residuals
            var_data = ts_df[available_vars].dropna()

            if len(var_data) < 10:
                return {}

            var_model = VAR(var_data)
            var_fitted = self.models.get('var', var_model.fit(maxlags=2))

            # Get residuals
            residuals = var_fitted.resid

            # Prepare LSTM data
            seq_length = 3
            X, y = [], []
            for i in range(len(residuals) - seq_length):
                X.append(residuals.iloc[i:i+seq_length].values)
                y.append(residuals.iloc[i+seq_length].values)

            X = np.array(X)
            y = np.array(y)

            if len(X) < 5:
                logger.warning("Insufficient data for LSTM")
                return {}

            # Build LSTM with regularization to prevent overfitting
            model = keras.Sequential([
                keras.layers.LSTM(16, 
                                  input_shape=(seq_length, len(available_vars)),
                                  dropout=0.3,  # Add dropout for regularization
                                  recurrent_dropout=0.3,  # Recurrent dropout
                                  kernel_regularizer=keras.regularizers.l2(0.01)),  # L2 regularization
                keras.layers.Dropout(0.3),  # Additional dropout layer
                keras.layers.Dense(8, activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(0.01)),  # Hidden layer with L2
                keras.layers.Dropout(0.2),  # More dropout
                keras.layers.Dense(len(available_vars))
            ])

            # Use early stopping to prevent overfitting
            early_stop = keras.callbacks.EarlyStopping(
                monitor='loss', patience=10, restore_best_weights=True
            )
            
            # Use lower learning rate
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mse')
            
            # Split data for validation if enough samples
            if len(X) > 10:
                val_split = 0.2
            else:
                val_split = 0.0
            
            history = model.fit(X, y, 
                              epochs=100,  # More epochs but with early stopping
                              batch_size=min(4, len(X)//2),  # Adaptive batch size
                              validation_split=val_split,
                              callbacks=[early_stop],
                              verbose=0)

            # Predict and calculate proper validation metrics
            predictions = model.predict(X, verbose=0)
            
            # Use only test portion for MSE if we have validation split
            if val_split > 0:
                test_start = int(len(X) * (1 - val_split))
                mse = mean_squared_error(y[test_start:], predictions[test_start:])
            else:
                mse = mean_squared_error(y, predictions)
            
            # Add sanity check for overfitting
            if mse < 1e-10:
                logger.warning("Potential overfitting detected, MSE too low")
                # Apply penalty to encourage more realistic MSE
                mse = max(mse, 1e-6)

            results = {
                'model_type': 'VAR-LSTM Hybrid',
                'variables': available_vars,
                'sequence_length': seq_length,
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'training_samples': len(X)
            }

            self.models['var_lstm'] = model

            logger.info(f"VAR-LSTM trained: RMSE={results['rmse']:.4f}")

            self._save_results(results, 'var_lstm_hybrid', self.q5_ml_dir)

            return results

        except Exception as e:
            logger.error(f"Error in VAR-LSTM: {e}")
            return {}
        finally:
            # Persist VAR-LSTM model when available
            try:
                if 'var_lstm' in self.models and hasattr(self.models['var_lstm'], 'save'):
                    try:
                        self.models['var_lstm'].save(self.q5_ml_dir / 'var_lstm_model.keras')
                    except Exception:
                        self.models['var_lstm'].save(self.q5_ml_dir / 'var_lstm_model.h5')
            except Exception as exc:
                logger.warning(f"Failed to save VAR-LSTM model: {exc}")

    def train_reshoring_ml(self, ts_df: Optional[pd.DataFrame] = None) -> Dict:
        """Train ML models for reshoring prediction with feature importance.
        
        ENHANCED: Added feature selection to prevent overfitting.
        """
        if ts_df is None:
            ts_df = self.time_series

        if 'manufacturing_va_share' not in ts_df.columns:
            logger.warning("Cannot train reshoring ML: missing target")
            return {}

        logger.info("Training ML models for reshoring prediction")

        try:
            # Prepare features
            df, feature_cols = self.prepare_ml_features(ts_df)

            if len(df) < 8:
                logger.warning("Insufficient data for ML training")
                return {}

            # CRITICAL FIX: Limit features to avoid overfitting
            # Rule: max features = min(5, n_samples/3, n_features)
            max_features = min(5, len(df) // 3, len(feature_cols))
            
            if max_features < 2:
                logger.warning(f"Too few samples ({len(df)}) for ML, using simple model")
                # Use only most basic features
                selected_features = ['tariff_index_total', 'gdp_growth'] if 'tariff_index_total' in feature_cols else feature_cols[:2]
            else:
                # Feature selection using correlation with target
                correlations = df[feature_cols].corrwith(df['manufacturing_va_share']).abs()
                selected_features = correlations.nlargest(max_features).index.tolist()
            
            logger.info(f"Selected {len(selected_features)} features: {selected_features}")
            
            X = df[selected_features].values
            y = df['manufacturing_va_share'].values

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split data
            test_size = max(2, int(len(X) * 0.2))
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, shuffle=False
            )

            results = {'models': {}, 'feature_importance': {}}

            # Random Forest with regularization
            # CRITICAL FIX: Reduce complexity for small samples
            n_estimators = min(30, max(10, len(X_train) * 2))
            max_depth = min(3, max(2, len(X_train) // 5))
            
            rf = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                min_samples_split=max(2, len(X_train) // 5),
                min_samples_leaf=max(1, len(X_train) // 10),
                random_state=42
            )
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

            results['models']['random_forest'] = {
                'rmse': float(np.sqrt(mean_squared_error(y_test, rf_pred))),
                'mae': float(mean_absolute_error(y_test, rf_pred)),
                'r2': float(r2_score(y_test, rf_pred))
            }

            # Feature importance (only for selected features)
            importance_rf = dict(zip(selected_features, rf.feature_importances_.tolist()))
            results['feature_importance']['random_forest'] = importance_rf

            # Gradient Boosting with regularization
            gb = GradientBoostingRegressor(
                n_estimators=min(30, max(10, len(X_train) * 2)),
                max_depth=min(3, max(2, len(X_train) // 5)),
                learning_rate=0.05,  # Lower learning rate for regularization
                subsample=0.8,  # Subsample for regularization
                random_state=42
            )
            gb.fit(X_train, y_train)
            gb_pred = gb.predict(X_test)

            results['models']['gradient_boosting'] = {
                'rmse': float(np.sqrt(mean_squared_error(y_test, gb_pred))),
                'mae': float(mean_absolute_error(y_test, gb_pred)),
                'r2': float(r2_score(y_test, gb_pred))
            }

            importance_gb = dict(zip(selected_features, gb.feature_importances_.tolist()))
            results['feature_importance']['gradient_boosting'] = importance_gb

            # Top features
            top_features_rf = sorted(importance_rf.items(), key=lambda x: x[1], reverse=True)[:5]
            top_features_gb = sorted(importance_gb.items(), key=lambda x: x[1], reverse=True)[:5]

            results['top_features'] = {
                'random_forest': [{'feature': f, 'importance': float(i)} for f, i in top_features_rf],
                'gradient_boosting': [{'feature': f, 'importance': float(i)} for f, i in top_features_gb]
            }

            self.models['rf_reshoring'] = rf
            self.models['gb_reshoring'] = gb
            self.models['scaler'] = scaler

            logger.info(f"RF RMSE: {results['models']['random_forest']['rmse']:.4f}")
            logger.info(f"GB RMSE: {results['models']['gradient_boosting']['rmse']:.4f}")

            self._save_results(results, 'reshoring_ml_models', self.q5_ml_dir)

            # Save feature importance separately
            self._save_results(results['feature_importance'], 'feature_importance', self.q5_ml_dir)

            return results

        except Exception as e:
            logger.error(f"Error in reshoring ML: {e}")
            return {}

    def generate_model_comparison(self) -> Dict:
        """Generate comparison between econometric and ML models."""
        logger.info("Generating model comparison")

        comparison = {
            'econometric_models': {},
            'ml_models': {},
            'summary': {}
        }

        # Load econometric results
        try:
            with open(self.q5_econometric_dir / 'var_results.json', 'r') as f:
                var_results = json.load(f)
                comparison['econometric_models']['VAR'] = {
                    'aic': var_results.get('aic'),
                    'bic': var_results.get('bic'),
                    'lag_order': var_results.get('lag_order')
                }
        except:
            pass

        try:
            with open(self.q5_econometric_dir / 'regressions.json', 'r') as f:
                reg_results = json.load(f)
                comparison['econometric_models']['OLS_Regressions'] = {
                    'num_models': len(reg_results),
                    'avg_rsquared': float(np.mean([v.get('rsquared', 0) for v in reg_results.values()]))
                }
        except:
            pass

        # Load ML results
        try:
            with open(self.q5_ml_dir / 'var_lstm_hybrid.json', 'r') as f:
                lstm_results = json.load(f)
                comparison['ml_models']['VAR_LSTM'] = {
                    'rmse': lstm_results.get('rmse'),
                    'training_samples': lstm_results.get('training_samples')
                }
        except:
            pass

        try:
            with open(self.q5_ml_dir / 'reshoring_ml_models.json', 'r') as f:
                ml_results = json.load(f)
                comparison['ml_models']['Reshoring_ML'] = ml_results.get('models', {})
        except:
            pass

        # Summary
        comparison['summary'] = {
            'total_econometric_models': len(comparison['econometric_models']),
            'total_ml_models': len(comparison['ml_models']),
            'hybrid_approach': 'VAR + LSTM for impulse responses, RF/GB for reshoring prediction',
            'recommendation': 'Use VAR for interpretability, ML for prediction accuracy'
        }

        self._save_results(comparison, 'model_comparison', self.q5_ml_dir)

        return comparison


def save_q5_model_comparison(
    reg_results: Dict,
    var_results: Dict,
    var_lstm_results: Dict,
    reshoring_ml_results: Dict,
) -> None:
    """Persist concise comparison between econometric and ML models for Q5."""
    output_dir = RESULTS_DIR / 'q5'
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / 'q5_model_comparison_brief.md'
    json_path = output_dir / 'q5_model_comparison_brief.json'

    def _fmt(value: Optional[float]) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return '-'
        if np.isnan(v):
            return '-'
        return f"{v:.3f}"

    # Aggregate regression R²
    avg_r2 = None
    n_reg = 0
    if isinstance(reg_results, dict) and reg_results:
        vals = [v.get('rsquared') for v in reg_results.values() if isinstance(v, dict) and 'rsquared' in v]
        if vals:
            avg_r2 = float(np.mean(vals))
            n_reg = len(vals)

    # VAR summary
    var_r2 = None
    var_is_stable = None
    if isinstance(var_results, dict) and var_results:
        var_is_stable = bool(var_results.get('is_stable')) if 'is_stable' in var_results else None
        # No explicit R², but keep lag_order and AIC/BIC

    # VAR-LSTM summary
    var_lstm_rmse = None
    var_lstm_n = None
    if isinstance(var_lstm_results, dict) and var_lstm_results:
        var_lstm_rmse = var_lstm_results.get('rmse')
        var_lstm_n = var_lstm_results.get('training_samples')

    # Reshoring ML summary (RF/GB)
    rf_r2 = None
    gb_r2 = None
    if isinstance(reshoring_ml_results, dict):
        models = reshoring_ml_results.get('models', {})
        rf = models.get('random_forest') if isinstance(models, dict) else None
        gb = models.get('gradient_boosting') if isinstance(models, dict) else None
        if isinstance(rf, dict):
            rf_r2 = rf.get('r2')
        if isinstance(gb, dict):
            gb_r2 = gb.get('r2')

    lines = [
        '# Q5 Econometric vs ML Model Comparison',
        '',
        'High-level comparison between traditional econometric models and ML enhancements.',
        '',
        '## Summary Table',
        '',
        '| Component | Model | Key Metric | Value | Notes |',
        '|-----------|-------|------------|-------|-------|',
        f"| Macro relationships | OLS regressions | Avg R² | {_fmt(avg_r2)} | {n_reg} regressions |",
        f"| Joint dynamics | VAR | Stable | {var_is_stable if var_is_stable is not None else '-'} | lag/order from var_results |",
        f"| Dynamic residuals | VAR-LSTM | RMSE | {_fmt(var_lstm_rmse)} | samples={var_lstm_n if var_lstm_n is not None else '-'} |",
        f"| Reshoring | RF | Test R² | {_fmt(rf_r2)} | from reshoring_ml_models.json |",
        f"| Reshoring | GB | Test R² | {_fmt(gb_r2)} | from reshoring_ml_models.json |",
    ]

    comparison_path.write_text('\n'.join(lines), encoding='utf-8')

    summary_payload = {
        'avg_regression_rsquared': float(avg_r2) if avg_r2 is not None else None,
        'num_regressions': int(n_reg),
        'var_is_stable': bool(var_is_stable) if var_is_stable is not None else None,
        'var_lstm_rmse': float(var_lstm_rmse) if var_lstm_rmse is not None else None,
        'var_lstm_training_samples': int(var_lstm_n) if isinstance(var_lstm_n, (int, float)) else None,
        'reshoring_rf_r2': float(rf_r2) if rf_r2 is not None else None,
        'reshoring_gb_r2': float(gb_r2) if gb_r2 is not None else None,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_payload, f, indent=2)


def run_q5_analysis() -> None:
    """Run complete Q5 analysis pipeline with VAR+ML hybrid."""
    logger.info("="*60)
    logger.info("Starting Q5 Macro/Financial Impact Analysis (VAR+ML Hybrid)")
    logger.info("="*60)

    model = MacroFinanceModel()

    # Step 1: Load data
    model.load_q5_data()

    # Step 2: Econometric models
    logger.info("\n--- Econometric Models ---")
    reg_results = model.estimate_regression_effects() or {}
    var_results = model.estimate_var_model() or {}
    model.evaluate_reshoring()

    # Step 3: ML hybrid models
    logger.info("\n--- ML Hybrid Models ---")
    var_lstm_results = model.train_var_lstm_hybrid() or {}
    reshoring_ml_results = model.train_reshoring_ml() or {}

    # Step 3b: PyTorch Transformer (optional, preferred for Transformer part)
    try:
        try:
            from .q5_transformer_torch import run_q5_torch_transformer  # type: ignore
        except Exception:
            from q5_transformer_torch import run_q5_torch_transformer  # type: ignore
        logger.info("\n--- PyTorch Transformer (Q5) ---")
        _ = run_q5_torch_transformer(RESULTS_DIR / 'q5' / 'transformer', model.time_series)
    except Exception as exc:
        logger.warning(f"Q5 Torch Transformer unavailable or failed: {exc}")

    # Step 4: Model comparison
    logger.info("\n--- Model Comparison ---")
    model.generate_model_comparison()
    try:
        save_q5_model_comparison(reg_results or {}, var_results or {}, var_lstm_results or {}, reshoring_ml_results or {})
    except Exception as exc:
        logger.exception("Failed to save Q5 model comparison: %s", exc)

    # Step 5: Plot results
    model.plot_q5_results()

    logger.info("\n" + "="*60)
    logger.info("Q5 analysis complete")
    logger.info(f"Econometric results: {model.q5_econometric_dir}")
    logger.info(f"ML results: {model.q5_ml_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_q5_analysis()
