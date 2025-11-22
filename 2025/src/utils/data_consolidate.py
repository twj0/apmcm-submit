"""Data consolidation utilities for merging and cleaning external datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def consolidate_fred_data(
    data_dir: Path,
    output_filename: str = "fred_consolidated.csv",
    wide_format: bool = False,
) -> pd.DataFrame:
    """Consolidate multiple FRED CSV files into a single DataFrame.
    
    Args:
        data_dir: Directory containing *_official.csv files from FRED
        output_filename: Name for consolidated output file
        wide_format: If True, pivot to wide format with series as columns
        
    Returns:
        Consolidated DataFrame with all FRED series
    """
    fred_files = list(data_dir.glob("*_official.csv"))
    
    if not fred_files:
        logger.warning("No FRED official files found in %s", data_dir)
        return pd.DataFrame()
    
    logger.info("Found %d FRED files to consolidate", len(fred_files))
    
    dfs = []
    for file_path in fred_files:
        try:
            df = pd.read_csv(file_path)
            if 'series_id' in df.columns:
                # Extract meaningful name from filename
                name = file_path.stem.replace("_official", "").replace("us_", "")
                df['indicator_name'] = name
                dfs.append(df)
                logger.debug("Loaded %s: %d rows", file_path.name, len(df))
        except Exception as e:
            logger.error("Failed to load %s: %s", file_path.name, e)
    
    if not dfs:
        logger.error("No valid FRED files could be loaded")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined = pd.concat(dfs, ignore_index=True)
    
    # Clean and standardize
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    combined['year'] = combined['year'].astype('Int64')
    combined = combined.sort_values(['year', 'series_id']).reset_index(drop=True)
    
    if wide_format:
        # Pivot to wide format: year as index, series_id as columns
        pivot = combined.pivot_table(
            index='year',
            columns='indicator_name',
            values='value',
            aggfunc='first'
        ).reset_index()
        output_df = pivot
        logger.info("Created wide format with %d years x %d indicators", 
                   len(pivot), len(pivot.columns) - 1)
    else:
        output_df = combined
        logger.info("Created long format with %d rows", len(combined))
    
    # Save consolidated data
    output_path = data_dir / output_filename
    output_df.to_csv(output_path, index=False)
    logger.info("Saved consolidated FRED data to %s", output_path)
    
    return output_df


def merge_macro_datasets(
    data_dir: Path,
    output_filename: str = "us_macro_consolidated.csv",
) -> pd.DataFrame:
    """Merge multiple macro indicators into a single wide-format table.
    
    This creates a comprehensive macro dataset suitable for Q5 analysis.
    
    Args:
        data_dir: Directory containing external data files
        output_filename: Name for output file
        
    Returns:
        Wide-format DataFrame with year as index and all indicators as columns
    """
    # Define expected indicators with their file patterns
    indicators = {
        'gdp_real': 'us_real_gdp_official.csv',
        'cpi': 'us_cpi_official.csv',
        'unemployment': 'us_unemployment_rate_official.csv',
        'industrial_production': 'us_industrial_production_official.csv',
        'fed_funds_rate': 'us_federal_funds_rate_official.csv',
        'treasury_10y': 'us_treasury_10y_yield_official.csv',
        'sp500': 'us_sp500_index_official.csv',
    }
    
    merged_data = {}
    
    for name, filename in indicators.items():
        file_path = data_dir / filename
        if not file_path.exists():
            logger.warning("Missing file: %s", filename)
            continue
        
        try:
            df = pd.read_csv(file_path)
            if 'year' in df.columns and 'value' in df.columns:
                # Extract year-value pairs
                year_values = df.set_index('year')['value'].to_dict()
                merged_data[name] = year_values
                logger.debug("Loaded %s: %d years", name, len(year_values))
        except Exception as e:
            logger.error("Failed to process %s: %s", filename, e)
    
    if not merged_data:
        logger.error("No macro data could be loaded")
        return pd.DataFrame()
    
    # Create wide-format DataFrame
    result = pd.DataFrame(merged_data).reset_index()
    result.rename(columns={'index': 'year'}, inplace=True)
    result = result.sort_values('year').reset_index(drop=True)
    
    # Save
    output_path = data_dir / output_filename
    result.to_csv(output_path, index=False)
    logger.info("Saved consolidated macro data to %s", output_path)
    logger.info("Shape: %d years x %d indicators", len(result), len(result.columns) - 1)
    
    return result


def check_data_quality(data_dir: Path) -> dict:
    """Generate a data quality report for all CSV files in the directory.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dictionary with quality metrics for each file
    """
    report = {}
    csv_files = list(data_dir.glob("*.csv"))
    
    logger.info("Checking data quality for %d CSV files", len(csv_files))
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # Calculate quality metrics
            total_rows = len(df)
            total_cols = len(df.columns)
            missing_values = df.isnull().sum().sum()
            missing_pct = (missing_values / (total_rows * total_cols) * 100) if total_rows > 0 else 0
            
            # Check for duplicate rows
            duplicates = df.duplicated().sum()
            
            # Check year coverage if year column exists
            year_info = {}
            if 'year' in df.columns:
                years = df['year'].dropna().unique()
                year_info = {
                    'min_year': int(years.min()) if len(years) > 0 else None,
                    'max_year': int(years.max()) if len(years) > 0 else None,
                    'year_count': len(years),
                }
            
            report[file_path.name] = {
                'rows': total_rows,
                'columns': total_cols,
                'missing_values': int(missing_values),
                'missing_pct': round(missing_pct, 2),
                'duplicates': int(duplicates),
                'year_info': year_info,
                'columns_list': list(df.columns),
            }
            
        except Exception as e:
            report[file_path.name] = {'error': str(e)}
            logger.error("Failed to check %s: %s", file_path.name, e)
    
    return report


def print_quality_report(report: dict) -> None:
    """Print a formatted data quality report.
    
    Args:
        report: Quality report dictionary from check_data_quality
    """
    print("\n" + "=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)
    
    for filename, metrics in sorted(report.items()):
        print(f"\n📄 {filename}")
        
        if 'error' in metrics:
            print(f"   ❌ Error: {metrics['error']}")
            continue
        
        print(f"   Dimensions: {metrics['rows']:,} rows × {metrics['columns']} columns")
        print(f"   Missing: {metrics['missing_values']:,} values ({metrics['missing_pct']}%)")
        
        if metrics['duplicates'] > 0:
            print(f"   ⚠️  Duplicates: {metrics['duplicates']}")
        
        if metrics['year_info']:
            yi = metrics['year_info']
            print(f"   Time range: {yi.get('min_year')} - {yi.get('max_year')} ({yi.get('year_count')} years)")
        
        # Show completeness indicator
        if metrics['missing_pct'] == 0 and metrics['duplicates'] == 0:
            print("   ✅ Data quality: EXCELLENT")
        elif metrics['missing_pct'] < 5 and metrics['duplicates'] == 0:
            print("   ✓ Data quality: GOOD")
        else:
            print("   ⚠️  Data quality: NEEDS REVIEW")
    
    print("\n" + "=" * 80)


def main():
    """CLI entry point for data consolidation utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Consolidate and check external data quality")
    parser.add_argument(
        'command',
        choices=['consolidate', 'merge-macro', 'check'],
        help='Operation to perform'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('2025/data/external'),
        help='Directory containing data files'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output filename (optional)'
    )
    parser.add_argument(
        '--wide',
        action='store_true',
        help='Use wide format for consolidate command'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.command == 'consolidate':
        output = args.output or 'fred_consolidated.csv'
        consolidate_fred_data(args.data_dir, output, args.wide)
    
    elif args.command == 'merge-macro':
        output = args.output or 'us_macro_consolidated.csv'
        merge_macro_datasets(args.data_dir, output)
    
    elif args.command == 'check':
        report = check_data_quality(args.data_dir)
        print_quality_report(report)


if __name__ == '__main__':
    main()
