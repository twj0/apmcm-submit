"""Utility modules for APMCM 2025 Problem C."""

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    TARIFF_DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
    set_random_seed,
    ensure_directories,
    apply_plot_style,
)

from .data_loader import TariffDataLoader, load_processed_data

from .data_fetch import (
    DataFetchError,
    fetch_un_comtrade_soybeans,
    fetch_fred_series,
)

from .mapping import (
    HSMapper,
    CountryMapper,
    create_hs_sector_mapping,
    save_mapping_tables,
    SOYBEANS_HS,
    AUTOS_HS,
    SEMICONDUCTORS_HS,
)

from .external_data import (
    ensure_q1_external_data,
    ensure_q2_external_data,
    ensure_q3_external_data,
    ensure_q4_external_data,
    ensure_q5_external_data,
    ensure_all_external_data,
)

__all__ = [
    # Config
    'PROJECT_ROOT',
    'DATA_DIR',
    'TARIFF_DATA_DIR',
    'RESULTS_DIR',
    'FIGURES_DIR',
    'RANDOM_SEED',
    'set_random_seed',
    'ensure_directories',
    'apply_plot_style',
    # Data loader
    'TariffDataLoader',
    'load_processed_data',
    # Data fetch helpers
    'DataFetchError',
    'fetch_un_comtrade_soybeans',
    'fetch_fred_series',
    # Mapping
    'HSMapper',
    'CountryMapper',
    'create_hs_sector_mapping',
    'save_mapping_tables',
    'SOYBEANS_HS',
    'AUTOS_HS',
    'SEMICONDUCTORS_HS',
    # External data helpers
    'ensure_q1_external_data',
    'ensure_q2_external_data',
    'ensure_q3_external_data',
    'ensure_q4_external_data',
    'ensure_q5_external_data',
    'ensure_all_external_data',
]
