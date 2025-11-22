"""
Q6 Data Visualization Module.

This module provides visualization functions for the Q6 integrated dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

def load_q6_data(data_path: Path) -> pd.DataFrame:
    """Load Q6 integrated dataset.
    
    Args:
        data_path: Path to the Q6 dataset CSV file
        
    Returns:
        Loaded DataFrame
    """
    if not data_path.exists():
        logger.error(f"Q6 data file not found: {data_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded Q6 data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading Q6 data: {e}")
        return pd.DataFrame()

def plot_automotive_sales_trends(df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot automotive sales trends by brand over time.
    
    Args:
        df: Q6 dataset DataFrame
        figures_dir: Directory to save figures
        
    Returns:
        Path to saved figure
    """
    if df.empty or 'year' not in df.columns or 'total_sales' not in df.columns:
        logger.warning("Required columns for automotive sales trends not found")
        return None
    
    # Filter rows with brand information
    auto_df = df[df['brand'].notna()].copy()
    
    if auto_df.empty:
        logger.warning("No automotive data found in Q6 dataset")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group by brand and plot sales trends
    brands = auto_df['brand'].unique()
    colors = sns.color_palette('husl', len(brands))
    
    for i, brand in enumerate(brands):
        brand_data = auto_df[auto_df['brand'] == brand]
        ax.plot(brand_data['year'], brand_data['total_sales'], 
               'o-', label=brand, color=colors[i], linewidth=2, markersize=6)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Sales')
    ax.set_title('Automotive Sales Trends by Brand (2015-2025)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure (PDF + PNG)
    output_path = figures_dir / 'q6_automotive_sales_trends.pdf'
    png_path = figures_dir / 'q6_automotive_sales_trends.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved automotive sales trends figures: {output_path} and {png_path}")
    
    return output_path

def plot_trade_metrics_correlation(df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot correlation heatmap of key trade metrics.
    
    Args:
        df: Q6 dataset DataFrame
        figures_dir: Directory to save figures
        
    Returns:
        Path to saved figure
    """
    if df.empty:
        logger.warning("Q6 dataset is empty")
        return None
    
    # Select key trade metrics
    trade_columns = [
        'total_imports_usd', 'total_tariff_revenue_usd', 'effective_tariff_rate',
        'china_imports_usd', 'china_tariff_revenue_usd', 'avg_tariff_rate',
        'expected_revenue_billions', 'gdp_growth', 'industrial_production'
    ]
    
    # Filter columns that exist in the dataset
    available_columns = [col for col in trade_columns if col in df.columns]
    
    if len(available_columns) < 2:
        logger.warning("Insufficient trade metrics columns found for correlation plot")
        return None
    
    # Use only numeric data for correlation
    corr_df = df[available_columns].select_dtypes(include=[np.number])
    
    if corr_df.empty or corr_df.shape[1] < 2:
        logger.warning("No numeric trade metrics found for correlation plot")
        return None
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
    
    ax.set_title('Correlation Matrix of Key Trade Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure (PDF + PNG)
    output_path = figures_dir / 'q6_trade_metrics_correlation.pdf'
    png_path = figures_dir / 'q6_trade_metrics_correlation.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved trade metrics correlation figures: {output_path} and {png_path}")
    
    return output_path

def plot_semiconductor_trends(df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot semiconductor industry trends.
    
    Args:
        df: Q6 dataset DataFrame
        figures_dir: Directory to save figures
        
    Returns:
        Path to saved figure
    """
    if df.empty or 'year' not in df.columns:
        logger.warning("Required columns for semiconductor trends not found")
        return None
    
    # Select semiconductor-related columns
    chip_columns = [
        'us_chip_output_index', 'us_chip_output_billions', 'global_chip_output_billions',
        'us_global_share_pct', 'global_chip_demand_index'
    ]
    
    # Filter columns that exist in the dataset
    available_columns = [col for col in chip_columns if col in df.columns]
    
    if not available_columns:
        logger.warning("No semiconductor columns found in Q6 dataset")
        return None
    
    # Aggregate data by year (mean values)
    yearly_df = df.groupby('year')[available_columns].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each semiconductor metric
    for col in available_columns:
        ax.plot(yearly_df['year'], yearly_df[col], 'o-', label=col.replace('_', ' ').title(), 
               linewidth=2, markersize=6)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title('Semiconductor Industry Trends (2010-2025)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure (PDF + PNG)
    output_path = figures_dir / 'q6_semiconductor_trends.pdf'
    png_path = figures_dir / 'q6_semiconductor_trends.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved semiconductor trends figures: {output_path} and {png_path}")
    
    return output_path

def plot_macro_economic_indicators(df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot macroeconomic indicators over time.
    
    Args:
        df: Q6 dataset DataFrame
        figures_dir: Directory to save figures
        
    Returns:
        Path to saved figure
    """
    if df.empty or 'year' not in df.columns:
        logger.warning("Required columns for macroeconomic indicators not found")
        return None
    
    # Select macroeconomic columns
    macro_columns = [
        'gdp_growth', 'industrial_production', 'unemployment_rate',
        'cpi', 'federal_funds_rate', 'dollar_index'
    ]
    
    # Filter columns that exist in the dataset
    available_columns = [col for col in macro_columns if col in df.columns]
    
    if not available_columns:
        logger.warning("No macroeconomic columns found in Q6 dataset")
        return None
    
    # Aggregate data by year (mean values)
    yearly_df = df.groupby('year')[available_columns].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each macroeconomic indicator
    for col in available_columns:
        ax.plot(yearly_df['year'], yearly_df[col], 'o-', label=col.replace('_', ' ').title(), 
               linewidth=2, markersize=6)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title('Macroeconomic Indicators Trends (2010-2025)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure (PDF + PNG)
    output_path = figures_dir / 'q6_macro_economic_indicators.pdf'
    png_path = figures_dir / 'q6_macro_economic_indicators.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved macroeconomic indicators figures: {output_path} and {png_path}")
    
    return output_path

def plot_manufacturing_reshoring(df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot manufacturing reshoring trends.
    
    Args:
        df: Q6 dataset DataFrame
        figures_dir: Directory to save figures
        
    Returns:
        Path to saved figure
    """
    if df.empty or 'year' not in df.columns:
        logger.warning("Required columns for manufacturing reshoring not found")
        return None
    
    # Select reshoring-related columns
    reshoring_columns = [
        'reshoring_fdi_billions', 'manufacturing_reshoring_index',
        'target_manufacturing_reshoring_index', 'z_reshoring_fdi_billions'
    ]
    
    # Filter columns that exist in the dataset
    available_columns = [col for col in reshoring_columns if col in df.columns]
    
    if not available_columns:
        logger.warning("No reshoring columns found in Q6 dataset")
        return None
    
    # Aggregate data by year (mean values)
    yearly_df = df.groupby('year')[available_columns].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each reshoring metric
    for col in available_columns:
        ax.plot(yearly_df['year'], yearly_df[col], 'o-', label=col.replace('_', ' ').title(), 
               linewidth=2, markersize=6)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title('Manufacturing Reshoring Trends (2010-2025)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure (PDF + PNG)
    output_path = figures_dir / 'q6_manufacturing_reshoring.pdf'
    png_path = figures_dir / 'q6_manufacturing_reshoring.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved manufacturing reshoring figures: {output_path} and {png_path}")
    
    return output_path

def create_q6_visualizations(data_path: Path, figures_dir: Path) -> List[Path]:
    """Create all visualizations for Q6.
    
    Args:
        data_path: Path to the Q6 dataset CSV file
        figures_dir: Directory to save figures
        
    Returns:
        List of paths to created figures
    """
    # Create figures directory if it doesn't exist
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_q6_data(data_path)
    
    if df.empty:
        logger.error("Cannot create Q6 visualizations: empty dataset")
        return []
    
    figures = []
    
    # Create visualizations
    fig_path = plot_automotive_sales_trends(df, figures_dir)
    if fig_path:
        figures.append(fig_path)
    
    fig_path = plot_trade_metrics_correlation(df, figures_dir)
    if fig_path:
        figures.append(fig_path)
    
    fig_path = plot_semiconductor_trends(df, figures_dir)
    if fig_path:
        figures.append(fig_path)
    
    fig_path = plot_macro_economic_indicators(df, figures_dir)
    if fig_path:
        figures.append(fig_path)
    
    fig_path = plot_manufacturing_reshoring(df, figures_dir)
    if fig_path:
        figures.append(fig_path)
    
    logger.info(f"Created {len(figures)} Q6 visualizations")
    return figures