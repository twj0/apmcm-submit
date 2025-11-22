"""Configuration and path management for APMCM 2025 Problem C."""

from pathlib import Path
import os
import random
import numpy as np


# === Project Root ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up to 2025/
WORKSPACE_ROOT = PROJECT_ROOT.parent
ENV_FILE = WORKSPACE_ROOT / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Load key=value pairs from .env without overriding existing vars."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

# === Data Paths ===
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"
DATA_INTERIM = DATA_DIR / "interim"

# Tariff Data (already provided)
TARIFF_DATA_DIR = PROJECT_ROOT / "problems" / "Tariff Data"

# === Output Paths ===
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PREDICTIONS = RESULTS_DIR / "predictions"
RESULTS_METRICS = RESULTS_DIR / "metrics"
RESULTS_TABLES = RESULTS_DIR / "tables"
RESULTS_LOGS = RESULTS_DIR / "logs"

FIGURES_DIR = PROJECT_ROOT / "figures"

# === Random Seed ===
RANDOM_SEED = 42


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    # If using PyTorch or TensorFlow, add their seed settings here


def ensure_directories() -> None:
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_RAW,
        DATA_PROCESSED,
        DATA_EXTERNAL,
        DATA_INTERIM,
        RESULTS_PREDICTIONS,
        RESULTS_METRICS,
        RESULTS_TABLES,
        RESULTS_LOGS,
        FIGURES_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# === Plotting Style ===
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}


def apply_plot_style() -> None:
    """Apply consistent plotting style."""
    try:
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams.update(PLOT_STYLE)
    except ImportError:
        pass  # matplotlib not installed yet


load_env_file()
set_random_seed()
ensure_directories()
