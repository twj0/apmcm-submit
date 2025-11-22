"""
Main entry point for APMCM 2025 Problem C analysis.

This script runs all question analyses (Q1-Q5) in sequence.

Usage:
    uv run python 2025/src/main.py [--questions Q1 Q2 Q3]
"""

import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import (
    run_q1_analysis,
    run_q2_analysis,
    run_q3_analysis,
    run_q4_analysis,
    run_q5_analysis,
)
from utils.config import ensure_directories, set_random_seed, RESULTS_DIR, FIGURES_DIR
from utils.mapping import save_mapping_tables
from utils.data_loader import TariffDataLoader
from utils.external_data import ensure_all_external_data
from utils.data_exporter import ModelResultsManager
from visualization.viz_template import create_all_visualizations


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('2025/results/logs/analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the analysis environment."""
    logger.info("Setting up analysis environment")
    
    # Ensure directories exist
    ensure_directories()
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create mapping tables
    try:
        save_mapping_tables()
        logger.info("Mapping tables created")
    except Exception as e:
        logger.warning(f"Could not create mapping tables: {e}")

    # Ensure external CSV/JSON parameter files exist with structured sample data
    # so that each question can run even before empirical data are available.
    try:
        ensure_all_external_data()
        logger.info("External data/parameter files verified or created under 2025/data/external/")
    except Exception as e:
        logger.warning(f"Could not ensure external data files: {e}")

    # Validate Tariff Data directory before proceeding so downstream modules
    # can rely on official USITC exports instead of placeholder content.
    loader = TariffDataLoader()
    validation = loader.validate_data_sources()
    if not validation.get('healthy', False):
        raise RuntimeError(
            "Tariff Data validation failed. Please review the logs, ensure "
            "2025/problems/Tariff Data contains the official CSV exports, and rerun."
        )


def run_all_analyses(no_ml: bool = False):
    """Run all question analyses in sequence.
    
    Args:
        no_ml: If True, disable ML enhancements where applicable
    """
    analyses = {
        'Q1': ('Soybean Trade Analysis', lambda: run_q1_analysis()),
        'Q2': ('Auto Trade Analysis', lambda: run_q2_analysis(use_transformer=not no_ml)),
        'Q3': ('Semiconductor Analysis', lambda: run_q3_analysis()),
        'Q4': ('Tariff Revenue Analysis', lambda: run_q4_analysis(use_ml=not no_ml)),
        'Q5': ('Macro/Financial Impact Analysis', lambda: run_q5_analysis()),
    }
    
    for q_name, (description, analysis_func) in analyses.items():
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Starting {q_name}: {description}")
        logger.info("=" * 70)
        
        try:
            analysis_func()
            logger.info(f"✓ {q_name} completed successfully")
        except Exception as e:
            logger.error(f"✗ {q_name} failed with error: {e}", exc_info=True)
        
        logger.info("")


def run_selected_analyses(questions: list, no_ml: bool = False):
    """Run selected question analyses.
    
    Args:
        questions: List of question names or numbers (e.g., ['Q1', 'Q3'] or ['2','4'])
        no_ml: If True, disable ML enhancements where applicable
    """
    analysis_map = {
        'Q1': lambda: run_q1_analysis(),
        'Q2': lambda: run_q2_analysis(use_transformer=not no_ml),
        'Q3': lambda: run_q3_analysis(),
        'Q4': lambda: run_q4_analysis(use_ml=not no_ml),
        'Q5': lambda: run_q5_analysis(),
    }
    
    for q in questions:
        # Normalize to 'QX'
        key = None
        if isinstance(q, int):
            key = f'Q{q}'
        else:
            s = str(q)
            key = f'Q{s}' if s.isdigit() else s.upper()
        
        if key not in analysis_map:
            logger.warning(f"Unknown question: {q}. Skipping.")
            continue
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Running {key}")
        logger.info("=" * 70)
        
        try:
            analysis_map[key]()
            logger.info(f"✓ {key} completed")
        except Exception as e:
            logger.error(f"✗ {key} failed: {e}", exc_info=True)


def generate_visualizations() -> None:
    """Generate all visualizations using the standardized template."""
    try:
        all_figures = create_all_visualizations(RESULTS_DIR, FIGURES_DIR)
        for question, figures in all_figures.items():
            logger.info(f"{question.upper()}: Generated {len(figures)} figures")
            for fig_path in figures:
                logger.info(f"  - {Path(fig_path).name}")
    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}", exc_info=True)


def generate_summary_reports() -> None:
    """Generate per-question SUMMARY.md reports listing outputs."""
    for q_num in [1, 2, 3, 4, 5]:
        try:
            manager = ModelResultsManager(q_num, RESULTS_DIR)
            q_dir = RESULTS_DIR / f'q{q_num}'
            if q_dir.exists():
                for method_dir in q_dir.iterdir():
                    if method_dir.is_dir():
                        manager.register_method(method_dir.name)
            summary_path = manager.generate_summary()
            logger.info(f"Q{q_num} summary: {summary_path}")
        except Exception as e:
            logger.warning(f"Could not generate summary for Q{q_num}: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='APMCM 2025 Problem C Analysis (main entry, compatible with run_all_models)'
    )
    parser.add_argument(
        '--questions',
        nargs='+',
        help='Specific questions to run; accepts Q1..Q5 or 1..5 (default: all)'
    )
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations after analyses')
    parser.add_argument('--no-ml', action='store_true', help='Disable ML enhancements (affects Q2 Transformer, Q4 ML)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("APMCM 2025 Problem C - U.S. Tariff Policy Analysis")
    logger.info("=" * 70)
    
    # Setup
    # Respect dynamic log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    setup_environment()
    
    # Run analyses
    if args.questions:
        run_selected_analyses(args.questions, no_ml=args.no_ml)
    else:
        run_all_analyses(no_ml=args.no_ml)
    
    # Optional visualization and summary
    if args.visualize:
        generate_visualizations()
    generate_summary_reports()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Analysis pipeline completed")
    logger.info("Results saved to: 2025/results/")
    logger.info("Figures saved to: 2025/figures/")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
