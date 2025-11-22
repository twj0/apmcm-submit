"""
Q4: Tariff Revenue - Short and Medium-Term Analysis

Dynamic "Laffer curve" analysis with quadratic tariff-revenue relationship
and lagged import response to predict revenue over Trump's second term.

Enhanced with ML models for dynamic prediction.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, List
import logging
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.config import RESULTS_DIR, FIGURES_DIR, DATA_EXTERNAL, DATA_PROCESSED
from utils.data_loader import TariffDataLoader

logger = logging.getLogger(__name__)


class TariffRevenueModel:
    """Model for tariff revenue analysis with Laffer curve dynamics."""
    
    def __init__(self):
        """Initialize the model."""
        self.loader = TariffDataLoader()
        self.panel_data: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        self.elasticities: Dict = {}
        self.ml_models: Dict = {}

        # Create output directories
        self.econometric_dir = RESULTS_DIR / 'q4' / 'econometric'
        self.ml_dir = RESULTS_DIR / 'q4' / 'ml'
        self.econometric_dir.mkdir(parents=True, exist_ok=True)
        self.ml_dir.mkdir(parents=True, exist_ok=True)
        
    def load_q4_data(self) -> pd.DataFrame:
        """Load and aggregate data for revenue analysis.
        
        Returns:
            Panel DataFrame with revenue, imports, and tariffs
        """
        logger.info("Loading Q4 tariff revenue data")
        
        # Prefer processed annual revenue panel if available
        processed_panel = DATA_PROCESSED / 'q4' / 'q4_0_tariff_revenue_panel.csv'
        if processed_panel.exists():
            try:
                df = pd.read_csv(processed_panel)
                # Ensure required columns exist and rename for internal compatibility
                if 'year' in df.columns and 'total_tariff_revenue_usd' in df.columns:
                    panel = df.copy()
                    panel = panel.sort_values('year').reset_index(drop=True)
                    panel.rename(columns={'total_tariff_revenue_usd': 'total_revenue'}, inplace=True)
                    # Coerce numeric types
                    panel['year'] = panel['year'].astype(int)
                    panel['total_revenue'] = panel['total_revenue'].astype(float)
                    self.panel_data = panel
                    logger.info(f"Loaded processed revenue panel with {len(panel)} rows (years {panel['year'].min()}-{panel['year'].max()})")
                    return panel
            except Exception as e:
                logger.warning(f"Failed to read processed Q4 panel, falling back to raw imports: {e}")
        
        # Fallback: aggregate from raw imports with duty collected
        imports = self.loader.load_imports()
        revenue_panel = imports.groupby('year').agg({'duty_collected': 'sum'}).reset_index()
        revenue_panel.rename(columns={'duty_collected': 'total_revenue'}, inplace=True)
        revenue_panel = revenue_panel.sort_values('year').reset_index(drop=True)
        
        logger.info(f"Loaded {len(revenue_panel)} years of revenue data from raw imports")
        logger.info(f"Years: {revenue_panel['year'].tolist()}")
        logger.info(f"Total revenue range: {revenue_panel['total_revenue'].min():.0f} - {revenue_panel['total_revenue'].max():.0f}")
        
        self.panel_data = revenue_panel
        return revenue_panel
    
    def estimate_static_revenue_model(
        self,
        panel_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Estimate static Laffer-style revenue model.
        
        Model: ln(R) = α + β1·τ + β2·τ² + β3·X + ε
        
        Args:
            panel_df: Revenue panel data
            
        Returns:
            Model coefficients and revenue-maximizing tariff
        """
        if panel_df is None:
            panel_df = self.panel_data
        
        if panel_df is None or len(panel_df) < 5:
            logger.error("Insufficient data for estimation")
            return {}
        
        logger.info("Estimating static Laffer model")
        
        panel_df = panel_df.copy()
        
        # Prefer processed revenue panel for average tariff by year; fallback to external CSV
        processed_panel = DATA_PROCESSED / 'q4' / 'q4_0_tariff_revenue_panel.csv'
        avg_df: Optional[pd.DataFrame] = None
        if processed_panel.exists():
            try:
                tmp = pd.read_csv(processed_panel)
                if 'year' in tmp.columns and 'effective_tariff_rate' in tmp.columns:
                    avg_df = tmp[['year', 'effective_tariff_rate']].dropna().copy()
                    # Normalize to proportion if provided in percent
                    avg_df.rename(columns={'effective_tariff_rate': 'avg_tariff'}, inplace=True)
                    if avg_df['avg_tariff'].max() > 1.0:
                        avg_df['avg_tariff'] = avg_df['avg_tariff'] / 100.0
            except Exception as e:
                logger.warning(f"Failed reading processed revenue panel for avg tariff: {e}")

        if avg_df is None:
            avg_tariff_file = DATA_EXTERNAL / 'q4_avg_tariff_by_year.csv'
            if not avg_tariff_file.exists():
                logger.warning(f"Average tariff config not found: {avg_tariff_file}")
                template = panel_df[['year']].drop_duplicates().sort_values('year')
                template['avg_tariff'] = np.nan
                template.to_csv(avg_tariff_file, index=False)
                logger.warning(
                    "Created template q4_avg_tariff_by_year.csv. "
                    "Please fill 'avg_tariff' for each year before re-running."
                )
                return {}
            try:
                avg_df = pd.read_csv(avg_tariff_file)
            except Exception as e:
                logger.error(f"Error reading average tariff config: {e}")
                return {}
            if 'year' not in avg_df.columns or 'avg_tariff' not in avg_df.columns:
                logger.error("q4_avg_tariff_by_year.csv must contain 'year' and 'avg_tariff' columns")
                return {}

        panel_df = panel_df.merge(avg_df[['year', 'avg_tariff']], on='year', how='left')
        panel_df = panel_df.dropna(subset=['avg_tariff'])
        
        if len(panel_df) < 5:
            logger.error("Insufficient observations with avg_tariff for estimation")
            return {}
        
        panel_df['avg_tariff_sq'] = panel_df['avg_tariff'] ** 2
        panel_df['ln_revenue'] = np.log(panel_df['total_revenue'] + 1)
        
        try:
            # Estimate model
            formula = 'ln_revenue ~ avg_tariff + avg_tariff_sq'
            model = smf.ols(formula, data=panel_df).fit()
            
            self.models['static_laffer'] = model
            
            # Find revenue-maximizing tariff: -β1 / (2*β2)
            beta1 = model.params.get('avg_tariff', 0)
            beta2 = model.params.get('avg_tariff_sq', -1)
            
            if beta2 < 0:
                optimal_tariff = -beta1 / (2 * beta2)
            else:
                optimal_tariff = np.nan
            
            results = {
                'rsquared': float(model.rsquared),
                'nobs': int(model.nobs),
                'beta1_tariff': float(beta1),
                'beta2_tariff_sq': float(beta2),
                'optimal_tariff_pct': float(optimal_tariff * 100) if not np.isnan(optimal_tariff) else None,
            }
            
            logger.info(f"Static model R²: {results['rsquared']:.3f}")
            if results['optimal_tariff_pct']:
                logger.info(f"Revenue-maximizing tariff: {results['optimal_tariff_pct']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error estimating static model: {e}")
            results = {}
        
        # Save results to econometric directory
        with open(self.econometric_dir / 'static_laffer.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Also save to old location for compatibility
        with open(RESULTS_DIR / 'q4_static_laffer.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results
    
    def estimate_dynamic_import_response(
        self,
        panel_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Estimate dynamic import response to tariff changes.
        
        Model: Δln(M) = φ0 + φ1·Δτ_t + φ2·Δτ_{t-1} + ... + u
        
        Args:
            panel_df: Panel data with imports and tariffs
            
        Returns:
            Dynamic elasticity parameters
        """
        logger.info("Estimating dynamic import response")
        
        # Prefer estimating from processed revenue panel; fallback to external JSON
        processed_panel = DATA_PROCESSED / 'q4' / 'q4_0_tariff_revenue_panel.csv'
        if processed_panel.exists():
            try:
                df = pd.read_csv(processed_panel).sort_values('year')
                # Use total imports and effective tariff to estimate simple dynamics
                if {'total_imports_usd', 'effective_tariff_rate'}.issubset(df.columns):
                    tau = df['effective_tariff_rate'].astype(float).copy()
                    if tau.max() > 1.0:
                        tau = tau / 100.0
                    m = df['total_imports_usd'].astype(float).replace(0, np.nan)
                    dlog_m = np.log(m).diff()
                    dtau = tau.diff()
                    dtau_l1 = dtau.shift(1)
                    reg = pd.DataFrame({'dlog_m': dlog_m, 'dtau': dtau, 'dtau_l1': dtau_l1}).dropna()
                    if len(reg) >= 5:
                        X = sm.add_constant(reg[['dtau', 'dtau_l1']])
                        model = sm.OLS(reg['dlog_m'], X).fit()
                        phi1 = float(model.params.get('dtau', 0.0))
                        phi2 = float(model.params.get('dtau_l1', 0.0))
                        params = {
                            'short_run_elasticity': phi1,
                            'medium_run_elasticity': phi1 + phi2,
                            'adjustment_speed': 0.3,
                        }
                        with open(self.econometric_dir / 'dynamic_import.json', 'w') as f:
                            json.dump(params, f, indent=2)
                        with open(RESULTS_DIR / 'q4_dynamic_import.json', 'w') as f:
                            json.dump(params, f, indent=2)
                        return params
            except Exception as e:
                logger.warning(f"Failed to estimate dynamic import response from processed data: {e}")

        # Fallback: read external JSON parameters
        params_file = DATA_EXTERNAL / 'q4_dynamic_import_params.json'
        if not params_file.exists():
            logger.warning(f"Dynamic import parameter file not found: {params_file}")
            template = {
                'short_run_elasticity': None,
                'medium_run_elasticity': None,
                'adjustment_speed': None,
            }
            with open(params_file, 'w') as f:
                json.dump(template, f, indent=2)
            logger.warning(
                "Created template q4_dynamic_import_params.json. "
                "Please fill numeric values before relying on dynamic effects."
            )
            return {}
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
        except Exception as e:
            logger.error(f"Error reading dynamic import parameters: {e}")
            return {}
        required_keys = ['short_run_elasticity', 'medium_run_elasticity', 'adjustment_speed']
        for k in required_keys:
            if k not in params or params[k] is None:
                logger.error(f"Dynamic import parameter '{k}' is missing or None")
                return {}
        with open(self.econometric_dir / 'dynamic_import.json', 'w') as f:
            json.dump(params, f, indent=2)
        with open(RESULTS_DIR / 'q4_dynamic_import.json', 'w') as f:
            json.dump(params, f, indent=2)
        return params
    
    def simulate_second_term_revenue(
        self,
        static_model: Optional[Dict] = None,
        dynamic_model: Optional[Dict] = None,
        tariff_scenarios: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Simulate revenue over second term (4-5 years).
        
        Args:
            static_model: Static Laffer model results
            dynamic_model: Dynamic import response parameters
            tariff_scenarios: Dict defining baseline vs policy tariff paths
            
        Returns:
            DataFrame with projected revenue paths
        """
        logger.info("Simulating second-term revenue")
        
        # Load tariff scenarios: prefer processed CSV, fallback to external JSON
        scenarios_csv = DATA_PROCESSED / 'q4' / 'q4_1_tariff_scenarios.csv'
        if tariff_scenarios is None and scenarios_csv.exists():
            try:
                csv = pd.read_csv(scenarios_csv)
                if {'year', 'scenario', 'avg_tariff_rate'}.issubset(csv.columns):
                    years = sorted(csv['year'].unique().tolist())
                    tariff_scenarios = {}
                    for name, grp in csv.groupby('scenario'):
                        series = grp.sort_values('year')['avg_tariff_rate'].tolist()
                        tariff_scenarios[name] = series
                    # Derive base_import_value from processed revenue panel if available
                    base_import_value = None
                    processed_panel = DATA_PROCESSED / 'q4' / 'q4_0_tariff_revenue_panel.csv'
                    if processed_panel.exists():
                        try:
                            rp = pd.read_csv(processed_panel).sort_values('year')
                            if 'total_imports_usd' in rp.columns:
                                base_import_value = float(rp['total_imports_usd'].iloc[-1])
                        except Exception:
                            base_import_value = None
                else:
                    years = None
            except Exception as e:
                logger.warning(f"Failed to read processed scenarios CSV: {e}")
                years = None

        if tariff_scenarios is None or not tariff_scenarios:
            scenarios_file = DATA_EXTERNAL / 'q4_tariff_scenarios.json'
            if tariff_scenarios is None and not scenarios_file.exists():
                logger.warning(f"Tariff scenarios file not found: {scenarios_file}")
                # Create comprehensive scenarios (7 different policy paths)
                template = {
                    'base_import_value': 3200000000000,  # $3.2T baseline
                    'years': [2025, 2026, 2027, 2028, 2029],
                    'scenarios': {
                        # Scenario 1: Baseline (current policy continuation)
                        'baseline': [0.025, 0.026, 0.027, 0.028, 0.029],
                        
                        # Scenario 2: Aggressive reciprocal tariffs
                        'reciprocal_aggressive': [0.10, 0.15, 0.20, 0.20, 0.20],
                        
                        # Scenario 3: Moderate reciprocal tariffs
                        'reciprocal_moderate': [0.05, 0.075, 0.10, 0.10, 0.10],
                        
                        # Scenario 4: Gradual escalation
                        'gradual_escalation': [0.03, 0.05, 0.08, 0.12, 0.15],
                        
                        # Scenario 5: Trade war (high tariffs on China)
                        'trade_war_china': [0.15, 0.25, 0.30, 0.25, 0.20],
                        
                        # Scenario 6: Selective tariffs (targeted sectors)
                        'selective_sectors': [0.04, 0.06, 0.08, 0.07, 0.06],
                        
                        # Scenario 7: De-escalation path
                        'de_escalation': [0.08, 0.06, 0.04, 0.03, 0.025],
                    },
                }
                with open(scenarios_file, 'w') as f:
                    json.dump(template, f, indent=2)
                logger.info(
                    "Created comprehensive q4_tariff_scenarios.json with 7 policy scenarios."
                )
                tariff_scenarios = template.get('scenarios', {})
                base_import_value = template.get('base_import_value')
                years = template.get('years')
            if tariff_scenarios is None:
                try:
                    with open(scenarios_file, 'r') as f:
                        config = json.load(f)
                except Exception as e:
                    logger.error(f"Error reading tariff scenarios: {e}")
                    return pd.DataFrame()
                base_import_value = config.get('base_import_value')
                years = config.get('years')
                tariff_scenarios = config.get('scenarios', {})
            else:
                # Provided dict
                years = [2025 + i for i in range(len(next(iter(tariff_scenarios.values()))))]
                base_import_value = None
        
        if base_import_value is None:
            logger.error("base_import_value must be specified in q4_tariff_scenarios.json")
            return pd.DataFrame()
        
        if dynamic_model is None:
            dynamic_model = self.estimate_dynamic_import_response()
        
        use_dynamic = bool(dynamic_model)
        
        results = []
        
        for scenario_name, tariff_path in tariff_scenarios.items():
            if len(tariff_path) != len(years):
                logger.error(f"Tariff path length mismatch for scenario {scenario_name}")
                continue
            
            import_value = base_import_value
            for year, tariff_rate in zip(years, tariff_path):
                if tariff_rate is None:
                    logger.error(f"Missing tariff rate for year {year} in scenario {scenario_name}")
                    continue
                
                if use_dynamic and results:
                    prev_tariff = results[-1]['tariff_rate']
                    tariff_change = tariff_rate - prev_tariff
                    elasticity = dynamic_model['short_run_elasticity']
                    import_change_pct = elasticity * tariff_change
                    import_value = results[-1]['import_value'] * (1 + import_change_pct)
                
                revenue = import_value * tariff_rate
                
                results.append({
                    'scenario': scenario_name,
                    'year': year,
                    'tariff_rate': tariff_rate,
                    'import_value': import_value,
                    'revenue': revenue,
                })
        
        if not results:
            logger.error("No valid tariff scenarios to simulate")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # Compute cumulative revenue difference for baseline vs reciprocal_tariff if both present
        scenario_names = results_df['scenario'].unique().tolist()
        summary = {}
        # Support both 'reciprocal_tariff' and 'reciprocal' naming
        policy_name = 'reciprocal_tariff' if 'reciprocal_tariff' in scenario_names else ('reciprocal' if 'reciprocal' in scenario_names else None)
        if 'baseline' in scenario_names and policy_name is not None:
            baseline_revenue = results_df[results_df['scenario'] == 'baseline']['revenue'].sum()
            policy_revenue = results_df[results_df['scenario'] == policy_name]['revenue'].sum()
            net_revenue_gain = policy_revenue - baseline_revenue
            
            logger.info(f"Baseline cumulative revenue: ${baseline_revenue/1e9:.1f}B")
            logger.info(f"Policy cumulative revenue: ${policy_revenue/1e9:.1f}B")
            logger.info(f"Net revenue gain: ${net_revenue_gain/1e9:.1f}B")
            
            summary = {
                'baseline_total': float(baseline_revenue),
                'policy_total': float(policy_revenue),
                'net_gain': float(net_revenue_gain),
            }
        
        # Save results to econometric directory
        results_df.to_csv(self.econometric_dir / 'revenue_scenarios.csv', index=False)

        if summary:
            with open(self.econometric_dir / 'revenue_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

        # Also save to old location for compatibility
        results_df.to_csv(RESULTS_DIR / 'q4_revenue_scenarios.csv', index=False)

        if summary:
            with open(RESULTS_DIR / 'q4_revenue_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

        return results_df
    
    def engineer_features(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML models."""
        df = panel_df.copy()
        df = df.sort_values('year').reset_index(drop=True)

        # Lagged features
        df['tariff_lag1'] = df['avg_tariff'].shift(1)
        df['tariff_lag2'] = df['avg_tariff'].shift(2)
        df['revenue_lag1'] = df['total_revenue'].shift(1)

        # Tariff changes
        df['tariff_change'] = df['avg_tariff'].diff()
        df['tariff_change_lag1'] = df['tariff_change'].shift(1)

        # Policy indicators
        df['high_tariff_regime'] = (df['avg_tariff'] > df['avg_tariff'].median()).astype(int)

        return df.dropna()

    def train_ml_models(self, panel_df: Optional[pd.DataFrame] = None) -> Dict:
        """Train ML models for revenue prediction."""
        if panel_df is None:
            panel_df = self.panel_data

        if panel_df is None or len(panel_df) < 10:
            logger.error("Insufficient data for ML training")
            return {}

        logger.info("Training ML models for revenue prediction")

        # Load tariff data: prefer processed revenue panel for avg_tariff
        processed_panel = DATA_PROCESSED / 'q4' / 'q4_0_tariff_revenue_panel.csv'
        avg_df: Optional[pd.DataFrame] = None
        if processed_panel.exists():
            try:
                tmp = pd.read_csv(processed_panel)
                if 'year' in tmp.columns and 'effective_tariff_rate' in tmp.columns:
                    avg_df = tmp[['year', 'effective_tariff_rate']].dropna().copy()
                    avg_df.rename(columns={'effective_tariff_rate': 'avg_tariff'}, inplace=True)
                    if avg_df['avg_tariff'].max() > 1.0:
                        avg_df['avg_tariff'] = avg_df['avg_tariff'] / 100.0
            except Exception as e:
                logger.warning(f"Failed reading processed revenue panel for avg tariff: {e}")
        if avg_df is None:
            avg_tariff_file = DATA_EXTERNAL / 'q4_avg_tariff_by_year.csv'
            if not avg_tariff_file.exists():
                logger.error("Average tariff data required for ML training")
                return {}
            avg_df = pd.read_csv(avg_tariff_file)
        panel_df = panel_df.merge(avg_df[['year', 'avg_tariff']], on='year', how='left')
        panel_df = panel_df.dropna(subset=['avg_tariff'])

        # Engineer features
        df = self.engineer_features(panel_df)

        if len(df) < 5:
            logger.error("Insufficient data after feature engineering")
            return {}

        # Prepare training data
        feature_cols = ['avg_tariff', 'tariff_lag1', 'tariff_lag2', 'tariff_change',
                       'tariff_change_lag1', 'high_tariff_regime']
        X = df[feature_cols].values
        y = df['total_revenue'].values

        # Train Gradient Boosting model
        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        gb_model.fit(X, y)
        y_pred = gb_model.predict(X)

        # Calculate metrics
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'mae': float(mean_absolute_error(y, y_pred)),
            'r2': float(r2_score(y, y_pred)),
            'feature_importance': {col: float(imp) for col, imp in zip(feature_cols, gb_model.feature_importances_)}
        }

        self.ml_models['gradient_boosting'] = gb_model
        logger.info(f"GB Model R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.0f}")

        # Save model metadata
        with open(self.ml_dir / 'gb_model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def train_arima_model(self, panel_df: Optional[pd.DataFrame] = None) -> Dict:
        """Train ARIMA model for time series forecasting."""
        if panel_df is None:
            panel_df = self.panel_data

        if panel_df is None or len(panel_df) < 10:
            logger.error("Insufficient data for ARIMA")
            return {}

        logger.info("Training ARIMA model for time series forecasting")

        revenue_series = panel_df.sort_values('year')['total_revenue'].values

        try:
            arima_model = ARIMA(revenue_series, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            self.ml_models['arima'] = arima_fit

            metrics = {
                'aic': float(arima_fit.aic),
                'bic': float(arima_fit.bic),
                'order': [1, 1, 1]
            }

            logger.info(f"ARIMA Model AIC: {metrics['aic']:.1f}")

            with open(self.ml_dir / 'arima_model_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)

            return metrics

        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            return {}

    def ml_forecast_revenue(self, tariff_scenarios: Dict, years: List[int]) -> pd.DataFrame:
        """Forecast revenue using ML models."""
        if 'gradient_boosting' not in self.ml_models:
            logger.error("ML models not trained")
            return pd.DataFrame()

        logger.info("Generating ML-based revenue forecasts")

        gb_model = self.ml_models['gradient_boosting']
        results = []

        for scenario_name, tariff_path in tariff_scenarios.items():
            for i, (year, tariff) in enumerate(zip(years, tariff_path)):
                # Create features
                tariff_lag1 = tariff_path[i-1] if i > 0 else tariff
                tariff_lag2 = tariff_path[i-2] if i > 1 else tariff_lag1
                tariff_change = tariff - tariff_lag1
                tariff_change_lag1 = tariff_lag1 - tariff_lag2 if i > 0 else 0
                high_tariff = 1 if tariff > 0.05 else 0

                features = np.array([[tariff, tariff_lag1, tariff_lag2, tariff_change,
                                    tariff_change_lag1, high_tariff]])

                revenue_pred = gb_model.predict(features)[0]

                results.append({
                    'scenario': scenario_name,
                    'year': year,
                    'tariff_rate': tariff,
                    'revenue_ml': revenue_pred
                })

        results_df = pd.DataFrame(results)

        # Save ML forecasts
        results_df.to_csv(self.ml_dir / 'ml_revenue_forecasts.csv', index=False)

        return results_df

    def compare_models(self, econometric_df: pd.DataFrame, ml_df: pd.DataFrame) -> Dict:
        """Compare econometric vs ML predictions."""
        logger.info("Comparing econometric and ML predictions")

        merged = econometric_df.merge(ml_df, on=['scenario', 'year', 'tariff_rate'], how='inner')

        comparison = {}
        for scenario in merged['scenario'].unique():
            data = merged[merged['scenario'] == scenario]
            econ_total = data['revenue'].sum()
            ml_total = data['revenue_ml'].sum()
            diff_pct = ((ml_total - econ_total) / econ_total) * 100

            comparison[scenario] = {
                'econometric_total': float(econ_total),
                'ml_total': float(ml_total),
                'difference_pct': float(diff_pct)
            }

        # Save comparison
        with open(self.ml_dir / 'model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)

        # Save merged data
        merged.to_csv(self.ml_dir / 'combined_forecasts.csv', index=False)

        # Create markdown report
        md_lines = ["# Model Comparison: Econometric vs ML\n"]
        for scenario, metrics in comparison.items():
            md_lines.append(f"## {scenario}\n")
            md_lines.append(f"- Econometric Total: ${metrics['econometric_total']/1e9:.2f}B\n")
            md_lines.append(f"- ML Total: ${metrics['ml_total']/1e9:.2f}B\n")
            md_lines.append(f"- Difference: {metrics['difference_pct']:.1f}%\n\n")

        with open(self.ml_dir / 'model_comparison.md', 'w') as f:
            f.writelines(md_lines)

        logger.info("Model comparison complete")
        return comparison

    def plot_q4_results(self) -> None:
        """Generate plots for Q4 results."""
        import matplotlib.pyplot as plt
        from utils.config import apply_plot_style

        apply_plot_style()

        try:
            revenue_df = pd.read_csv(RESULTS_DIR / 'q4_revenue_scenarios.csv')
        except FileNotFoundError:
            logger.error("Revenue scenarios not found")
            return

        logger.info("Creating Q4 plots")

        # Plot 1: Revenue over time by scenario
        fig, ax = plt.subplots(figsize=(12, 6))

        for scenario in revenue_df['scenario'].unique():
            data = revenue_df[revenue_df['scenario'] == scenario]
            ax.plot(data['year'], data['revenue'] / 1e9, marker='o', label=scenario)

        ax.set_xlabel('Year')
        ax.set_ylabel('Tariff Revenue ($ Billions)')
        ax.set_title('Projected Tariff Revenue: Baseline vs Reciprocal Tariff Policy')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        pdf_path = FIGURES_DIR / 'q4_revenue_time_path.pdf'
        fig.savefig(pdf_path, bbox_inches='tight')
        png_path = FIGURES_DIR / 'q4_revenue_time_path.png'
        fig.savefig(png_path, bbox_inches='tight')
        plt.close()

        # Plot 2: Laffer curve (stylized)
        fig, ax = plt.subplots(figsize=(10, 6))

        tariff_rates = np.linspace(0, 0.50, 100)
        # Stylized Laffer curve: R = t * (1 - t/t_max) * base
        t_max = 0.40
        base_revenue = 100
        revenues = tariff_rates * (1 - tariff_rates / t_max) * base_revenue

        ax.plot(tariff_rates * 100, revenues, linewidth=2)
        ax.axvline(x=20, color='red', linestyle='--', label='Revenue-maximizing rate (~20%)')
        ax.axvline(x=12, color='blue', linestyle='--', label='Proposed policy (~12%)')

        ax.set_xlabel('Tariff Rate (%)')
        ax.set_ylabel('Revenue (Stylized Index)')
        ax.set_title('Laffer Curve: Tariff Rate vs Revenue')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        pdf_path = FIGURES_DIR / 'q4_laffer_curve.pdf'
        fig.savefig(pdf_path, bbox_inches='tight')
        png_path = FIGURES_DIR / 'q4_laffer_curve.png'
        fig.savefig(png_path, bbox_inches='tight')
        plt.close()

        logger.info("Q4 plots saved")


def run_q4_analysis(use_ml: bool = True) -> None:
    """Run complete Q4 analysis pipeline with ML enhancements.

    Args:
        use_ml: Whether to include ML-based predictions
    """
    logger.info("="*60)
    logger.info("Starting Q4 Tariff Revenue Analysis")
    if use_ml:
        logger.info("ML enhancements: ENABLED")
    logger.info("="*60)

    model = TariffRevenueModel()

    # Step 1: Load data
    panel_data = model.load_q4_data()

    # Step 2: Estimate static model (econometric)
    static_results = model.estimate_static_revenue_model()

    # Step 3: Estimate dynamic model (econometric)
    dynamic_results = model.estimate_dynamic_import_response()

    # Step 4: Simulate revenue scenarios (econometric)
    econ_scenarios = model.simulate_second_term_revenue(static_results, dynamic_results)

    # Step 5: ML enhancements
    if use_ml and len(panel_data) >= 10:
        logger.info("-"*60)
        logger.info("ML Enhancement Phase")
        logger.info("-"*60)

        # Train ML models
        model.train_ml_models(panel_data)
        model.train_arima_model(panel_data)

        # Load tariff scenarios for ML forecasting
        scenarios_file = DATA_EXTERNAL / 'q4_tariff_scenarios.json'
        if scenarios_file.exists():
            with open(scenarios_file, 'r') as f:
                config = json.load(f)

            years = config.get('years', [2025, 2026, 2027, 2028, 2029])
            tariff_scenarios = config.get('scenarios', {})

            # Generate ML forecasts
            ml_forecasts = model.ml_forecast_revenue(tariff_scenarios, years)

            # Compare models
            if not econ_scenarios.empty and not ml_forecasts.empty:
                model.compare_models(econ_scenarios, ml_forecasts)
        else:
            logger.warning("Tariff scenarios file not found, skipping ML forecasts")

    # Step 6: Plot results
    model.plot_q4_results()

    logger.info("="*60)
    logger.info("Q4 analysis complete")
    logger.info(f"Results saved to: {model.econometric_dir}")
    if use_ml:
        logger.info(f"ML results saved to: {model.ml_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_q4_analysis()
