import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sys.path.append(str(Path(__file__).parents[1]))
from utils.config import RESULTS_DIR, FIGURES_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_q1():
    """Generate Q1 visualizations from results."""
    logger.info("Q1: Soybean trade elasticities and scenarios")
    fig_dir = FIGURES_DIR / 'q1'
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / 'q1' / 'q1_scenario_exports.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Import quantity by scenario
        for exp in df['exporter'].unique():
            data = df[df['exporter'] == exp]
            ax1.plot(data['scenario'], data['simulated_import_quantity'], marker='o', label=exp, linewidth=2)
        ax1.set_xlabel('Scenario'); ax1.set_ylabel('Import Quantity'); ax1.set_title('Q1: Soybean Import Quantity')
        ax1.legend(); ax1.grid(alpha=0.3); ax1.tick_params(axis='x', rotation=15)

        # Market share
        for exp in df['exporter'].unique():
            data = df[df['exporter'] == exp]
            ax2.plot(data['scenario'], data['market_share'], marker='s', label=exp, linewidth=2)
        ax2.set_xlabel('Scenario'); ax2.set_ylabel('Market Share (%)'); ax2.set_title('Q1: Market Share by Exporter')
        ax2.legend(); ax2.grid(alpha=0.3); ax2.tick_params(axis='x', rotation=15)

        plt.tight_layout()
        fig.savefig(fig_dir / 'q1_scenarios.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_q2():
    """Generate Q2 visualizations from results."""
    logger.info("Q2: Auto trade scenarios")
    fig_dir = FIGURES_DIR / 'q2'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Scenario imports
    csv_path = RESULTS_DIR / 'q2' / 'econometric' / 'scenario_imports.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in df.columns[1:]:
            ax.plot(df.iloc[:, 0], df[col], marker='s', label=col, linewidth=2)
        ax.set_xlabel('Scenario'); ax.set_ylabel('Import Value'); ax.set_title('Q2: Japan Auto Import Scenarios')
        ax.legend(); ax.grid(alpha=0.3)
        fig.savefig(fig_dir / 'q2_scenarios.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_q3():
    """Generate Q3 visualizations from results."""
    logger.info("Q3: Semiconductor policy scenarios")
    fig_dir = FIGURES_DIR / 'q3'
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / 'q3_policy_scenarios.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Self-sufficiency by segment
        for pol in df['policy'].unique():
            data = df[df['policy'] == pol]
            ax1.plot(data['segment'], data['self_sufficiency_pct'], marker='o', label=pol, linewidth=2)
        ax1.set_xlabel('Segment'); ax1.set_ylabel('Self-Sufficiency (%)'); ax1.set_title('Q3: Self-Sufficiency by Policy')
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        # Security vs Cost tradeoff
        summary = df.groupby('policy').agg({'security_index': 'mean', 'cost_index': 'mean'}).reset_index()
        ax2.scatter(summary['cost_index'], summary['security_index'], s=200, alpha=0.7, c=range(len(summary)), cmap='viridis')
        for i, pol in enumerate(summary['policy']):
            ax2.annotate(pol.replace('Policy_', '').replace('_', ' '),
                        (summary['cost_index'].iloc[i], summary['security_index'].iloc[i]),
                        fontsize=8, ha='center')
        ax2.set_xlabel('Cost Index'); ax2.set_ylabel('Security Index'); ax2.set_title('Q3: Security-Cost Tradeoff')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(fig_dir / 'q3_policy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_q4():
    """Generate Q4 visualizations from results."""
    logger.info("Q4: Tariff revenue Laffer curve")
    fig_dir = FIGURES_DIR / 'q4'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Static Laffer curve
    json_path = RESULTS_DIR / 'q4' / 'econometric' / 'static_laffer.json'
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)

        fig, ax = plt.subplots(figsize=(10, 6))
        tariff_range = np.linspace(0, 0.5, 100)
        coef = data.get('coefficients', {})
        revenue = coef.get('const', 0) + coef.get('avg_tariff_rate', 0) * tariff_range + coef.get('avg_tariff_rate_sq', 0) * tariff_range**2

        ax.plot(tariff_range * 100, revenue, linewidth=3, color='darkblue')
        if 'optimal_tariff_rate' in data:
            opt = data['optimal_tariff_rate']
            ax.axvline(opt * 100, color='red', linestyle='--', linewidth=2, label=f'Optimal: {opt*100:.1f}%')
        ax.set_xlabel('Average Tariff Rate (%)'); ax.set_ylabel('Tariff Revenue'); ax.set_title('Q4: Laffer Curve')
        ax.legend(); ax.grid(alpha=0.3)
        fig.savefig(fig_dir / 'q4_laffer_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_q5():
    """Generate Q5 visualizations from results."""
    logger.info("Q5: Macro impacts and VAR")
    fig_dir = FIGURES_DIR / 'q5'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # VAR results
    json_path = RESULTS_DIR / 'q5' / 'econometric' / 'var_results.json'
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)

        if 'irf' in data:
            irf = data['irf']
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            for i, (var, ax) in enumerate(zip(['gdp_growth', 'industrial_production'], axes.flat[:2])):
                if var in irf:
                    periods = list(range(len(irf[var])))
                    ax.plot(periods, irf[var], marker='o', linewidth=2)
                    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
                    ax.set_xlabel('Periods'); ax.set_ylabel('Response'); ax.set_title(f'IRF: {var}')
                    ax.grid(alpha=0.3)

            plt.tight_layout()
            fig.savefig(fig_dir / 'q5_var_irf.png', dpi=300, bbox_inches='tight')
            plt.close()

def run_all():
    """Generate all visualizations."""
    logger.info("Starting visualization generation...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for plot_func in [plot_q1, plot_q2, plot_q3, plot_q4, plot_q5]:
        try:
            plot_func()
        except Exception as e:
            logger.error(f"{plot_func.__name__} failed: {e}")

    logger.info(f"Visualizations saved to {FIGURES_DIR}")

if __name__ == "__main__":
    run_all()
