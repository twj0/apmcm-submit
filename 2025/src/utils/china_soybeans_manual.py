"""Manual data preparation for China soybean imports.

Since UN Comtrade API is failing, this script provides tools to:
1. Process manually downloaded data from GACC or other sources
2. Search for and download data from alternative sources
3. Generate template files for manual data entry
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def create_template_csv(output_path: Path | str) -> pd.DataFrame:
    """Create a template CSV for manual data entry of China soybean imports.
    
    Args:
        output_path: Where to save the template CSV
        
    Returns:
        Empty template DataFrame
    """
    template = pd.DataFrame(columns=[
        'year',
        'exporter',  # US, Brazil, Argentina
        'import_value_usd',  # Total import value in USD
        'import_quantity_tonnes',  # Import quantity in metric tonnes
        'tariff_cn_on_exporter',  # China's tariff rate on this exporter
    ])
    
    # Add example rows as guidance
    examples = [
        {
            'year': 2015,
            'exporter': 'US',
            'import_value_usd': None,
            'import_quantity_tonnes': None,
            'tariff_cn_on_exporter': 0.03,  # Normal MFN rate
        },
        {
            'year': 2015,
            'exporter': 'Brazil',
            'import_value_usd': None,
            'import_quantity_tonnes': None,
            'tariff_cn_on_exporter': 0.03,
        },
        {
            'year': 2015,
            'exporter': 'Argentina',
            'import_value_usd': None,
            'import_quantity_tonnes': None,
            'tariff_cn_on_exporter': 0.03,
        },
    ]
    
    template = pd.concat([template, pd.DataFrame(examples)], ignore_index=True)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    
    logger.info("Created template file at %s", output_path)
    logger.info("Fill in import_value_usd and import_quantity_tonnes for years 2015-2024")
    logger.info("Data sources:")
    logger.info("  - GACC (China Customs): http://www.customs.gov.cn/")
    logger.info("  - UN Comtrade web interface: https://comtradeplus.un.org/")
    logger.info("  - WITS: https://wits.worldbank.org/")
    
    return template


def process_gacc_export(
    input_path: Path | str,
    output_path: Path | str,
    year_col: str = 'Year',
    partner_col: str = 'Country',
    value_col: str = 'Value',
    quantity_col: str = 'Quantity',
) -> pd.DataFrame:
    """Process a CSV exported from GACC data portal.
    
    Args:
        input_path: Path to raw GACC CSV
        output_path: Where to save processed data
        year_col: Name of year column in input
        partner_col: Name of partner country column
        value_col: Name of trade value column (in USD or CNY)
        quantity_col: Name of quantity column (in kg or tonnes)
        
    Returns:
        Processed DataFrame in standard format
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info("Loading GACC data from %s", input_path)
    df = pd.read_csv(input_path)
    
    logger.info("Input columns: %s", df.columns.tolist())
    
    # Try to find columns by name matching
    def find_column(df: pd.DataFrame, hints: list[str]) -> Optional[str]:
        for hint in hints:
            for col in df.columns:
                if hint.lower() in str(col).lower():
                    return col
        return None
    
    year_col_found = find_column(df, [year_col, 'year', 'period', '年份'])
    partner_col_found = find_column(df, [partner_col, 'partner', 'country', '国家', '来源国'])
    value_col_found = find_column(df, [value_col, 'value', 'import value', '进口额'])
    quantity_col_found = find_column(df, [quantity_col, 'quantity', 'import quantity', '进口量'])
    
    if not all([year_col_found, partner_col_found, value_col_found, quantity_col_found]):
        logger.error("Could not identify all required columns")
        logger.error("Found: year=%s, partner=%s, value=%s, quantity=%s",
                    year_col_found, partner_col_found, value_col_found, quantity_col_found)
        raise ValueError("Cannot process GACC data - column mapping failed")
    
    # Extract and standardize
    processed = pd.DataFrame({
        'year': pd.to_numeric(df[year_col_found], errors='coerce'),
        'exporter': df[partner_col_found].astype(str).str.strip(),
        'import_value_usd': pd.to_numeric(df[value_col_found], errors='coerce'),
        'import_quantity_tonnes': pd.to_numeric(df[quantity_col_found], errors='coerce'),
    })
    
    # Standardize country names
    country_mapping = {
        'united states': 'US',
        'usa': 'US',
        'america': 'US',
        '美国': 'US',
        'brazil': 'Brazil',
        '巴西': 'Brazil',
        'argentina': 'Argentina',
        '阿根廷': 'Argentina',
    }
    
    processed['exporter'] = processed['exporter'].str.lower().map(
        lambda x: country_mapping.get(x, x.title())
    )
    
    # Filter to main exporters
    main_exporters = ['US', 'Brazil', 'Argentina']
    processed = processed[processed['exporter'].isin(main_exporters)]
    
    # Add default tariff rates (can be updated manually)
    # Note: US had higher tariffs during trade war (2018-2020)
    def get_tariff(row):
        if row['exporter'] == 'US':
            if 2018 <= row['year'] <= 2020:
                return 0.25  # Retaliatory tariff
            else:
                return 0.03  # Normal MFN rate
        else:
            return 0.03  # Normal MFN rate for Brazil/Argentina
    
    processed['tariff_cn_on_exporter'] = processed.apply(get_tariff, axis=1)
    
    # Sort and save
    processed = processed.sort_values(['year', 'exporter']).reset_index(drop=True)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    
    logger.info("Processed %d rows and saved to %s", len(processed), output_path)
    logger.info("Year range: %d - %d", processed['year'].min(), processed['year'].max())
    logger.info("Exporters: %s", processed['exporter'].unique().tolist())
    
    return processed


def validate_soybean_data(data_path: Path | str) -> dict:
    """Validate China soybean import data for completeness and consistency.
    
    Args:
        data_path: Path to soybean import CSV
        
    Returns:
        Dictionary with validation results
    """
    data_path = Path(data_path)
    if not data_path.exists():
        return {'valid': False, 'error': f'File not found: {data_path}'}
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return {'valid': False, 'error': f'Cannot read CSV: {e}'}
    
    required_cols = ['year', 'exporter', 'import_value_usd', 'import_quantity_tonnes']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return {'valid': False, 'error': f'Missing columns: {missing_cols}'}
    
    # Check year coverage
    expected_years = set(range(2015, 2025))  # 2015-2024
    actual_years = set(df['year'].dropna().unique())
    missing_years = expected_years - actual_years
    
    # Check exporter coverage
    expected_exporters = {'US', 'Brazil', 'Argentina'}
    actual_exporters = set(df['exporter'].unique())
    missing_exporters = expected_exporters - actual_exporters
    
    # Check for missing values in critical columns
    value_missing = df['import_value_usd'].isnull().sum()
    quantity_missing = df['import_quantity_tonnes'].isnull().sum()
    
    # Data quality checks
    warnings = []
    if missing_years:
        warnings.append(f"Missing years: {sorted(missing_years)}")
    if missing_exporters:
        warnings.append(f"Missing exporters: {missing_exporters}")
    if value_missing > 0:
        warnings.append(f"{value_missing} rows missing import value")
    if quantity_missing > 0:
        warnings.append(f"{quantity_missing} rows missing import quantity")
    
    # Check for unrealistic values
    if 'import_value_usd' in df.columns:
        max_value = df['import_value_usd'].max()
        if max_value > 100_000_000_000:  # > 100 billion USD
            warnings.append(f"Suspiciously large import value: ${max_value:,.0f}")
    
    is_valid = len(warnings) == 0
    
    return {
        'valid': is_valid,
        'warnings': warnings,
        'year_coverage': {
            'expected': sorted(expected_years),
            'actual': sorted(actual_years),
            'missing': sorted(missing_years),
        },
        'exporter_coverage': {
            'expected': sorted(expected_exporters),
            'actual': sorted(actual_exporters),
            'missing': sorted(missing_exporters),
        },
        'data_completeness': {
            'total_rows': len(df),
            'value_missing': int(value_missing),
            'quantity_missing': int(quantity_missing),
        },
    }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Tools for manual China soybean import data preparation"
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Template command
    template_parser = subparsers.add_parser(
        'template',
        help='Create a template CSV for manual data entry'
    )
    template_parser.add_argument(
        '--output',
        type=Path,
        default=Path('2025/data/raw/china_soybeans_template.csv'),
        help='Output path for template'
    )
    
    # Process command
    process_parser = subparsers.add_parser(
        'process',
        help='Process manually downloaded GACC data'
    )
    process_parser.add_argument(
        'input',
        type=Path,
        help='Path to raw GACC CSV'
    )
    process_parser.add_argument(
        '--output',
        type=Path,
        default=Path('2025/data/external/china_imports_soybeans_official.csv'),
        help='Output path'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate soybean import data'
    )
    validate_parser.add_argument(
        'input',
        type=Path,
        help='Path to soybean CSV to validate'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    if args.command == 'template':
        create_template_csv(args.output)
    
    elif args.command == 'process':
        process_gacc_export(args.input, args.output)
    
    elif args.command == 'validate':
        result = validate_soybean_data(args.input)
        
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        
        if result.get('valid'):
            print("✅ Data is valid and complete")
        else:
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print("⚠️  Data has issues:")
                for warning in result.get('warnings', []):
                    print(f"  - {warning}")
        
        if 'year_coverage' in result:
            yc = result['year_coverage']
            print(f"\nYear coverage: {len(yc['actual'])}/{len(yc['expected'])} years")
            if yc['missing']:
                print(f"  Missing: {yc['missing']}")
        
        if 'exporter_coverage' in result:
            ec = result['exporter_coverage']
            print(f"\nExporter coverage: {len(ec['actual'])}/{len(ec['expected'])} exporters")
            if ec['missing']:
                print(f"  Missing: {ec['missing']}")
        
        if 'data_completeness' in result:
            dc = result['data_completeness']
            print(f"\nData completeness:")
            print(f"  Total rows: {dc['total_rows']}")
            print(f"  Missing values: {dc['value_missing']}")
            print(f"  Missing quantities: {dc['quantity_missing']}")
        
        print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
