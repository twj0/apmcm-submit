"""
CU-1: Tariff & Trade Data Loader

Centralized loader to read and merge Tariff Data (imports, exports,
annual tariff schedules) into standardized DataFrames.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging
from .config import TARIFF_DATA_DIR, DATA_PROCESSED

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TariffDataLoader:
    """Loader for U.S. Tariff Data."""
    
    def __init__(self, tariff_data_dir: Path = TARIFF_DATA_DIR):
        """Initialize the loader.
        
        Args:
            tariff_data_dir: Path to the Tariff Data directory
        """
        self.tariff_data_dir = Path(tariff_data_dir)
        self.import_file = self.tariff_data_dir / "DataWeb-Query-Import__General_Import_Charges.csv"
        self.export_file = self.tariff_data_dir / "DataWeb-Query-Export__FAS_Value.csv"

    @staticmethod
    def _standardize_hs_codes(series: pd.Series) -> pd.Series:
        """Normalize HS codes to digit-only strings while preserving length."""
        hs_series = series.astype("string").str.strip()
        hs_series = hs_series.str.replace(r"\.0+$", "", regex=True)
        hs_series = hs_series.str.replace(r"[^0-9]", "", regex=True)
        hs_series = hs_series.replace("", pd.NA)
        return hs_series

    # ------------------------------------------------------------------
    # Validation utilities
    # ------------------------------------------------------------------

    def validate_data_sources(self, min_year_columns: int = 3) -> Dict[str, bool]:
        """Validate that required Tariff Data CSVs exist and look non-template.

        Args:
            min_year_columns: Minimum number of year columns expected in
                the wide-format import/export CSVs.

        Returns:
            Dict summarizing validation results with a "healthy" flag.
        """

        logger.info("Validating Tariff Data sources under %s", self.tariff_data_dir)

        results = {
            'imports_csv': self._validate_main_tariff_csv(
                self.import_file,
                label='imports',
                skiprows=2,
                min_year_columns=min_year_columns,
            ),
            'exports_csv': self._validate_main_tariff_csv(
                self.export_file,
                label='exports',
                skiprows=2,
                min_year_columns=min_year_columns,
            ),
            'tariff_schedules': self._validate_tariff_schedule_dirs(),
        }

        results['healthy'] = all(results.values())

        if results['healthy']:
            logger.info("Tariff Data sources validation completed successfully.")
        else:
            logger.warning(
                "Tariff Data validation detected potential issues. "
                "Please review the warnings above before running analyses."
            )

        return results

    def _validate_main_tariff_csv(
        self,
        file_path: Path,
        label: str,
        skiprows: int,
        min_year_columns: int,
    ) -> bool:
        """Validate the main wide-format import/export CSVs."""

        if not file_path.exists():
            logger.error("%s CSV not found at %s", label.capitalize(), file_path)
            return False

        try:
            sample = pd.read_csv(file_path, skiprows=skiprows, nrows=5)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Could not read %s CSV %s: %s", label, file_path, exc)
            return False

        normalized_cols = {
            str(col).lower().strip().replace(' ', '_'): str(col)
            for col in sample.columns
        }

        required = {'hts_number', 'country'}
        missing = [col for col in required if col not in normalized_cols]
        if missing:
            logger.warning(
                "File %s appears to be missing required columns: %s",
                file_path,
                missing,
            )

        year_cols: List[str] = []
        for col in sample.columns:
            col_str = str(col).strip()
            if col_str.isdigit():
                year_cols.append(col)

        if len(year_cols) < min_year_columns:
            logger.warning(
                "%s CSV %s contains only %d year columns (expected >= %d)",
                label.capitalize(),
                file_path,
                len(year_cols),
                min_year_columns,
            )

        has_non_zero_data = True
        if year_cols:
            numeric_sample = sample[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            if float(numeric_sample.abs().sum().sum()) == 0.0:
                logger.warning(
                    "Sample rows from %s look all-zero. If this is placeholder data, "
                    "please replace it with the official Tariff Data export.",
                    file_path,
                )
                has_non_zero_data = False

        return not missing and has_non_zero_data and len(year_cols) >= min_year_columns

    def _validate_tariff_schedule_dirs(self) -> bool:
        """Ensure at least one annual tariff schedule CSV can be read."""

        schedule_dirs = sorted(self.tariff_data_dir.glob("tariff_data_*"))
        if not schedule_dirs:
            logger.warning(
                "No tariff_data_*/ subdirectories found under %s. "
                "Annual tariff schedules are required for downstream analysis.",
                self.tariff_data_dir,
            )
            return False

        any_valid_file = False
        for directory in schedule_dirs:
            csv_files = sorted(directory.glob("*.csv"))
            if not csv_files:
                logger.warning("No CSV files found inside %s", directory)
                continue

            sample_file = csv_files[0]
            try:
                pd.read_csv(sample_file, nrows=1)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Unable to read %s: %s", sample_file, exc)
                continue

            any_valid_file = True
            break

        if not any_valid_file:
            logger.warning(
                "Unable to read any tariff schedule CSVs under %s."
                " Please verify the Tariff Data archive.",
                self.tariff_data_dir,
            )

        return any_valid_file
        
    def load_imports(self) -> pd.DataFrame:
        """Load U.S. import data with duty collected.
        
        Returns:
            DataFrame with standardized columns
        """
        logger.info(f"Loading import data from {self.import_file}")
        
        try:
            # Read CSV, skip the metadata rows
            df = pd.read_csv(
                self.import_file,
                skiprows=2,
                dtype={"HTS Number": "string"}
            )
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Rename key columns to standard names
            rename_map = {
                'data_type': 'data_type',
                'hts_number': 'hs_code',
                'description': 'product_description',
                'country': 'partner_country',
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            
            # Melt year columns to long format
            year_cols = [col for col in df.columns if col.isdigit()]
            if year_cols:
                id_vars = [col for col in df.columns if col not in year_cols]
                df = df.melt(
                    id_vars=id_vars,
                    value_vars=year_cols,
                    var_name='year',
                    value_name='duty_collected'
                )
                df['year'] = df['year'].astype(int)
            
            # Convert duty_collected to numeric
            df['duty_collected'] = pd.to_numeric(df['duty_collected'], errors='coerce')
            
            # Ensure HS code is string with leading zeros preserved
            if 'hs_code' in df.columns:
                df['hs_code'] = self._standardize_hs_codes(df['hs_code'])
            
            logger.info(f"Loaded {len(df)} import records")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"Years: {sorted(df['year'].unique())}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading import data: {e}")
            raise
    
    def load_exports(self) -> pd.DataFrame:
        """Load U.S. export data.
        
        Returns:
            DataFrame with standardized columns
        """
        logger.info(f"Loading export data from {self.export_file}")
        
        try:
            # Read CSV, skip the metadata rows
            df = pd.read_csv(
                self.export_file,
                skiprows=2,
                dtype={"HTS Number": "string"}
            )
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Rename key columns
            rename_map = {
                'data_type': 'data_type',
                'hts_number': 'hs_code',
                'description': 'product_description',
                'country': 'partner_country',
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            
            # Melt year columns to long format
            year_cols = [col for col in df.columns if col.isdigit()]
            if year_cols:
                id_vars = [col for col in df.columns if col not in year_cols]
                df = df.melt(
                    id_vars=id_vars,
                    value_vars=year_cols,
                    var_name='year',
                    value_name='export_value'
                )
                df['year'] = df['year'].astype(int)
            
            # Convert export_value to numeric
            df['export_value'] = pd.to_numeric(df['export_value'], errors='coerce')
            
            # Ensure HS code is string with leading zeros preserved
            if 'hs_code' in df.columns:
                df['hs_code'] = self._standardize_hs_codes(df['hs_code'])
            
            logger.info(f"Loaded {len(df)} export records")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading export data: {e}")
            raise
    
    def load_annual_tariff_schedule(self, year: int) -> Optional[pd.DataFrame]:
        """Load annual tariff schedule data.
        
        Args:
            year: Year to load (e.g., 2020)
            
        Returns:
            DataFrame with tariff schedule, or None if not available
        """
        tariff_dir = self.tariff_data_dir / f"tariff_data_{year}"
        csv_file = tariff_dir / f"tariff_database_{year}10.csv"
        
        if not csv_file.exists():
            logger.warning(f"Tariff schedule for {year} not found at {csv_file}")
            return None
        
        logger.info(f"Loading tariff schedule for {year}")
        
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            df['year'] = year
            
            logger.info(f"Loaded {len(df)} tariff schedule records for {year}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading tariff schedule for {year}: {e}")
            return None
    
    def compute_effective_tariff(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute effective tariff rate.
        
        Args:
            df: DataFrame with 'duty_collected' and 'import_value' columns
            
        Returns:
            DataFrame with 'effective_tariff' column added
        """
        if 'duty_collected' in df.columns and 'import_value' in df.columns:
            # Only compute where import_value > 0
            mask = df['import_value'] > 0
            df.loc[mask, 'effective_tariff'] = (
                df.loc[mask, 'duty_collected'] / df.loc[mask, 'import_value']
            )
            
            # Cap at reasonable values (e.g., 200% = 2.0)
            df.loc[df['effective_tariff'] > 2.0, 'effective_tariff'] = np.nan
            df.loc[df['effective_tariff'] < 0, 'effective_tariff'] = np.nan
            
        return df
    
    def create_imports_panel(
        self,
        years: Optional[List[int]] = None,
        save_to_disk: bool = True
    ) -> pd.DataFrame:
        """Create a comprehensive imports panel.
        
        Args:
            years: List of years to include (if None, use all available)
            save_to_disk: Whether to save processed data
            
        Returns:
            Panel DataFrame with imports, duties, and effective tariffs
        """
        # Load import data
        imports = self.load_imports()
        
        # Filter years if specified
        if years:
            imports = imports[imports['year'].isin(years)]
        
        # Compute effective tariff (if import_value is available)
        # Note: The import file may not have import_value, only duty_collected
        # We'll need to merge with tariff schedules or use approximations
        
        logger.info(f"Created imports panel with shape {imports.shape}")
        
        # Save to disk
        if save_to_disk:
            output_file = DATA_PROCESSED / "imports_panel.parquet"
            imports.to_parquet(output_file, index=False)
            logger.info(f"Saved imports panel to {output_file}")
        
        return imports
    
    def create_exports_panel(
        self,
        years: Optional[List[int]] = None,
        save_to_disk: bool = True
    ) -> pd.DataFrame:
        """Create a comprehensive exports panel.
        
        Args:
            years: List of years to include
            save_to_disk: Whether to save processed data
            
        Returns:
            Panel DataFrame with exports
        """
        # Load export data
        exports = self.load_exports()
        
        # Filter years if specified
        if years:
            exports = exports[exports['year'].isin(years)]
        
        logger.info(f"Created exports panel with shape {exports.shape}")
        
        # Save to disk
        if save_to_disk:
            output_file = DATA_PROCESSED / "exports_panel.parquet"
            exports.to_parquet(output_file, index=False)
            logger.info(f"Saved exports panel to {output_file}")
        
        return exports
    
    def compute_tariff_indices(
        self,
        imports: pd.DataFrame,
        groupby: List[str] = ['year']
    ) -> pd.DataFrame:
        """Compute trade-weighted average tariff indices.
        
        Args:
            imports: Imports DataFrame with effective_tariff and import_value
            groupby: Columns to group by
            
        Returns:
            DataFrame with tariff indices
        """
        if 'effective_tariff' not in imports.columns:
            logger.warning("effective_tariff not found, cannot compute indices")
            return pd.DataFrame()
        
        # Remove missing values
        valid = imports.dropna(subset=['effective_tariff', 'duty_collected'])
        
        # Compute trade-weighted average
        indices = valid.groupby(groupby).apply(
            lambda x: pd.Series({
                'tariff_index_weighted': (
                    (x['duty_collected'] * x['effective_tariff']).sum() / 
                    x['duty_collected'].sum()
                    if x['duty_collected'].sum() > 0 else np.nan
                ),
                'tariff_index_simple': x['effective_tariff'].mean(),
                'total_duty_collected': x['duty_collected'].sum(),
            })
        ).reset_index()
        
        return indices


def load_processed_data(data_type: str = 'imports') -> pd.DataFrame:
    """Convenience function to load processed data.
    
    Args:
        data_type: Type of data ('imports', 'exports', 'tariff_indices')
        
    Returns:
        Loaded DataFrame
    """
    file_map = {
        'imports': DATA_PROCESSED / 'imports_panel.parquet',
        'exports': DATA_PROCESSED / 'exports_panel.parquet',
        'tariff_indices': DATA_PROCESSED / 'tariff_indices.parquet',
    }
    
    file_path = file_map.get(data_type)
    if not file_path or not file_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {file_path}")
    
    return pd.read_parquet(file_path)
