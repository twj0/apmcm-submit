"""
Visualization Template for Post-Processing Model Results.

This module provides templates for creating publication-quality visualizations
from the exported model results in 2025/results/q{N}/{method}/ directories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Base class for visualizing model results."""
    
    def __init__(self, results_dir: Path, figures_dir: Path, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize visualizer.
        
        Args:
            results_dir: Directory containing results (e.g., RESULTS_DIR / 'q2')
            figures_dir: Directory to save figures
            style: Matplotlib style
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set default parameters
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        
    def load_json(self, method: str, filename: str) -> Dict:
        """Load JSON results file.
        
        Args:
            method: Method name
            filename: Filename (with or without .json extension)
            
        Returns:
            Loaded data dictionary
        """
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = self.results_dir / method / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract data from metadata wrapper if present
        if 'data' in data:
            return data['data']
        return data
    
    def load_csv(self, method: str, filename: str) -> pd.DataFrame:
        """Load CSV results file.
        
        Args:
            method: Method name
            filename: Filename (with or without .csv extension)
            
        Returns:
            Loaded DataFrame
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        filepath = self.results_dir / method / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
        
        return pd.read_csv(filepath)
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300) -> Path:
        """Save figure to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution
            
        Returns:
            Path to saved PDF figure (PNG is also saved with same stem)
        """
        # Allow callers to pass names with or without extension; always save both PDF and PNG
        base = Path(filename)
        stem = base.stem  # drop any existing extension

        pdf_path = self.figures_dir / f"{stem}.pdf"
        png_path = self.figures_dir / f"{stem}.png"

        fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight')
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved figures: {pdf_path} and {png_path}")
        return pdf_path


class ModelComparisonVisualizer(ResultsVisualizer):
    """Visualizer for comparing different models/methods."""
    
    def plot_metrics_comparison(self, methods: List[str], metric_name: str = 'r2',
                               title: Optional[str] = None) -> Path:
        """Plot comparison of a specific metric across methods.
        
        Args:
            methods: List of method names
            metric_name: Metric to compare (e.g., 'r2', 'rmse', 'mae')
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        method_names = []
        metric_values = []
        
        for method in methods:
            # Try to load metrics file
            metrics = self.load_json(method, 'metrics')
            if metrics and metric_name in metrics:
                method_names.append(method)
                metric_values.append(metrics[metric_name])
        
        if not method_names:
            logger.warning(f"No {metric_name} data found for any method")
            plt.close(fig)
            return None
        
        # Create bar plot
        colors = sns.color_palette('husl', len(method_names))
        bars = ax.bar(method_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Method')
        ax.set_ylabel(metric_name.upper())
        ax.set_title(title or f'{metric_name.upper()} Comparison Across Methods')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15, ha='right')
        
        return self.save_figure(fig, f'comparison_{metric_name}.pdf')
    
    def plot_predictions_comparison(self, methods: List[str], 
                                   actual_col: str = 'actual',
                                   pred_col: str = 'predicted') -> Path:
        """Plot actual vs predicted for multiple methods.
        
        Args:
            methods: List of method names
            actual_col: Column name for actual values
            pred_col: Column name for predicted values
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
        if len(methods) == 1:
            axes = [axes]
        
        for ax, method in zip(axes, methods):
            df = self.load_csv(method, 'predictions')
            
            if df.empty or actual_col not in df.columns or pred_col not in df.columns:
                ax.text(0.5, 0.5, f'No data for {method}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Scatter plot
            ax.scatter(df[actual_col], df[pred_col], alpha=0.6, s=50)
            
            # Perfect prediction line
            min_val = min(df[actual_col].min(), df[pred_col].min())
            max_val = max(df[actual_col].max(), df[pred_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{method.title()}')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return self.save_figure(fig, 'predictions_comparison.pdf')


class TimeSeriesVisualizer(ResultsVisualizer):
    """Visualizer for time series results."""
    
    def plot_forecast(self, method: str, data_file: str,
                     historical_col: str, forecast_col: str,
                     time_col: str = 'year') -> Path:
        """Plot historical data with forecast.
        
        Args:
            method: Method name
            data_file: CSV filename
            historical_col: Column with historical data
            forecast_col: Column with forecast
            time_col: Time column
            
        Returns:
            Path to saved figure
        """
        df = self.load_csv(method, data_file)
        
        if df.empty:
            logger.error(f"No data found in {data_file}")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical
        if historical_col in df.columns:
            ax.plot(df[time_col], df[historical_col], 
                   'o-', label='Historical', linewidth=2, markersize=6)
        
        # Plot forecast
        if forecast_col in df.columns:
            forecast_mask = df[forecast_col].notna()
            ax.plot(df.loc[forecast_mask, time_col], 
                   df.loc[forecast_mask, forecast_col],
                   's--', label='Forecast', linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        ax.set_title(f'Time Series Forecast - {method.title()}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        return self.save_figure(fig, f'{method}_forecast.pdf')
    
    def plot_multiple_scenarios(self, method: str, scenarios_file: str,
                               value_col: str, scenario_col: str = 'scenario',
                               time_col: str = 'year') -> Path:
        """Plot multiple scenario forecasts.
        
        Args:
            method: Method name
            scenarios_file: CSV filename
            value_col: Column with values
            scenario_col: Column with scenario names
            time_col: Time column
            
        Returns:
            Path to saved figure
        """
        df = self.load_csv(method, scenarios_file)
        
        if df.empty:
            logger.error(f"No data found in {scenarios_file}")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        scenarios = df[scenario_col].unique()
        colors = sns.color_palette('husl', len(scenarios))
        
        for scenario, color in zip(scenarios, colors):
            scenario_data = df[df[scenario_col] == scenario]
            ax.plot(scenario_data[time_col], scenario_data[value_col],
                   'o-', label=scenario, color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Year')
        ax.set_ylabel(value_col.replace('_', ' ').title())
        ax.set_title(f'Scenario Comparison - {method.title()}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        return self.save_figure(fig, f'{method}_scenarios.pdf')


class NetworkVisualizer(ResultsVisualizer):
    """Visualizer for network/graph results."""
    
    def plot_risk_heatmap(self, method: str, risk_file: str,
                         segment_col: str = 'segment',
                         risk_col: str = 'supply_risk_index',
                         time_col: str = 'year') -> Path:
        """Plot risk heatmap across segments and time.
        
        Args:
            method: Method name
            risk_file: CSV filename
            segment_col: Column with segments
            risk_col: Column with risk values
            time_col: Time column
            
        Returns:
            Path to saved figure
        """
        df = self.load_csv(method, risk_file)
        
        if df.empty:
            logger.error(f"No data found in {risk_file}")
            return None
        
        # Pivot for heatmap
        pivot_df = df.pivot(index=segment_col, columns=time_col, values=risk_col)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   cbar_kws={'label': 'Risk Index'}, ax=ax)
        
        ax.set_title(f'Supply Chain Risk Heatmap - {method.title()}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Segment')
        
        return self.save_figure(fig, f'{method}_risk_heatmap.pdf')


# Convenience functions for common visualization tasks

def create_q2_visualizations(results_dir: Path, figures_dir: Path) -> List[Path]:
    """Create all visualizations for Q2.
    
    Args:
        results_dir: Q2 results directory
        figures_dir: Output figures directory
        
    Returns:
        List of paths to created figures
    """
    viz = ModelComparisonVisualizer(results_dir, figures_dir)
    ts_viz = TimeSeriesVisualizer(results_dir, figures_dir)
    
    figures = []
    
    # Method comparison
    methods = ['econometric', 'marl', 'transformer']
    fig_path = viz.plot_metrics_comparison(methods, 'r2', 'Q2 Model Performance Comparison')
    if fig_path:
        figures.append(fig_path)
    
    # Scenario forecasts
    fig_path = ts_viz.plot_multiple_scenarios('econometric', 'scenario_imports', 
                                             'japan_direct_imports')
    if fig_path:
        figures.append(fig_path)
    
    return figures


def create_q3_visualizations(results_dir: Path, figures_dir: Path) -> List[Path]:
    """Create all visualizations for Q3.
    
    Args:
        results_dir: Q3 results directory
        figures_dir: Output figures directory
        
    Returns:
        List of paths to created figures
    """
    viz = ModelComparisonVisualizer(results_dir, figures_dir)
    net_viz = NetworkVisualizer(results_dir, figures_dir)
    
    figures = []
    
    # Method comparison
    methods = ['econometric', 'gnn', 'ml']
    fig_path = viz.plot_metrics_comparison(methods, 'r2', 'Q3 Model Performance Comparison')
    if fig_path:
        figures.append(fig_path)
    
    # Risk heatmap
    fig_path = net_viz.plot_risk_heatmap('econometric', 'q3_security_metrics')
    if fig_path:
        figures.append(fig_path)
    
    return figures


def create_q4_visualizations(results_dir: Path, figures_dir: Path) -> List[Path]:
    """Create all visualizations for Q4.
    
    Args:
        results_dir: Q4 results directory
        figures_dir: Output figures directory
        
    Returns:
        List of paths to created figures
    """
    ts_viz = TimeSeriesVisualizer(results_dir, figures_dir)
    
    figures = []
    
    # Revenue scenarios
    fig_path = ts_viz.plot_multiple_scenarios('econometric', 'revenue_scenarios',
                                             'revenue')
    if fig_path:
        figures.append(fig_path)
    
    # ML forecasts
    fig_path = ts_viz.plot_multiple_scenarios('ml', 'ml_revenue_forecasts',
                                             'revenue_ml')
    if fig_path:
        figures.append(fig_path)
    
    return figures


def create_q5_visualizations(results_dir: Path, figures_dir: Path) -> List[Path]:
    """Create all visualizations for Q5.
    
    Args:
        results_dir: Q5 results directory
        figures_dir: Output figures directory
        
    Returns:
        List of paths to created figures
    """
    viz = ModelComparisonVisualizer(results_dir, figures_dir)
    ts_viz = TimeSeriesVisualizer(results_dir, figures_dir)
    
    figures = []
    
    # Method comparison
    methods = ['econometric', 'ml']
    fig_path = viz.plot_metrics_comparison(methods, 'r2', 'Q5 Model Performance Comparison')
    if fig_path:
        figures.append(fig_path)
    
    return figures


def create_all_visualizations(base_results_dir: Path, base_figures_dir: Path) -> Dict[str, List[Path]]:
    """Create all visualizations for all questions.
    
    Args:
        base_results_dir: Base results directory (RESULTS_DIR)
        base_figures_dir: Base figures directory (FIGURES_DIR)
        
    Returns:
        Dict mapping question to list of figure paths
    """
    all_figures = {}
    
    # Q2
    q2_results = base_results_dir / 'q2'
    if q2_results.exists():
        all_figures['q2'] = create_q2_visualizations(q2_results, base_figures_dir)
    
    # Q3
    q3_results = base_results_dir / 'q3'
    if q3_results.exists():
        all_figures['q3'] = create_q3_visualizations(q3_results, base_figures_dir)
    
    # Q4
    q4_results = base_results_dir / 'q4'
    if q4_results.exists():
        all_figures['q4'] = create_q4_visualizations(q4_results, base_figures_dir)
    
    # Q5
    q5_results = base_results_dir / 'q5'
    if q5_results.exists():
        all_figures['q5'] = create_q5_visualizations(q5_results, base_figures_dir)
    
    return all_figures
