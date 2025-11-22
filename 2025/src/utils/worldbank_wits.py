"""Helpers for fetching data from World Bank / WITS-related sources.

This module implements two practical workflows based on your summary:

1) World Bank macro / trade indicators via the `wbdata` library
   (no registration required, JSON API under the hood).

2) Working with WITS bulk-download CSV files (which you obtain via the
   WITS web UI "Advanced Query" and email links). Python code here helps
   you load and standardize those CSVs into tidy DataFrames for analysis.

Important:
- For HS-level trade flows (e.g. China soybean imports by HS code), the
  recommended programmatic source remains UN Comtrade's API. WITS bulk
  download is still the easiest way to obtain very large extracts, but it
  normally requires manual job submission in a browser.
- This module therefore does NOT try to automate logging into WITS or
  submitting jobs. It focuses on:
  * Calling the public World Bank API via `wbdata`.
  * Loading/standardizing the CSV files you download from WITS.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

try:  # Optional dependency
    import wbdata  # type: ignore
except Exception:  # pragma: no cover - optional
    wbdata = None


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# World Bank (wbdata) helpers
# ---------------------------------------------------------------------------


def fetch_worldbank_indicator_to_csv(
    indicator: str,
    *,
    country: str = "CHN",
    start_year: int = 2015,
    end_year: int = 2024,
    out_path: Path | str = Path("worldbank_indicator.csv"),
    clean_output: bool = True,
) -> pd.DataFrame:
    """Fetch a World Bank indicator via `wbdata` and save as CSV.

    This is a thin wrapper around `wbdata.get_data` that:
    - Requests values for a given country and year window.
    - Converts the returned list-of-dicts into a DataFrame.
    - Optionally cleans nested dictionary columns for better usability.
    - Writes it to CSV for inspection and downstream analysis.

    Args:
        indicator: World Bank indicator code, e.g. "TM.TAX.MANF.SM.AR.ZS".
        country: ISO2 / ISO3 code accepted by `wbdata`, e.g. "CHN".
        start_year: First year (inclusive).
        end_year: Last year (inclusive).
        out_path: Where to save the CSV.
        clean_output: If True, extracts 'value'/'id' from nested dict columns.

    Returns:
        pandas.DataFrame of the fetched data.
    """

    if wbdata is None:
        raise ImportError(
            "wbdata is not installed. Install it first with 'pip install wbdata'."
        )

    start = _dt.datetime(start_year, 1, 1)
    end = _dt.datetime(end_year, 12, 31)

    LOGGER.info(
        "Fetching World Bank indicator %s for %s, %s-%s",
        indicator,
        country,
        start_year,
        end_year,
    )

    # wbdata 1.1.0 expects the "date" argument instead of the older
    # "data_date" keyword. It accepts strings ("YYYY") or datetime objects.
    rows: List[Dict[str, Any]] = wbdata.get_data(
        indicator,
        country=country,
        date=(start.strftime("%Y"), end.strftime("%Y")),
    )

    if not rows:
        LOGGER.warning("World Bank API returned no data for given query.")

    df = pd.DataFrame(rows)
    
    # Clean nested dictionary columns if requested
    if clean_output and not df.empty:
        for col in df.columns:
            # Check if column contains dict-like strings or actual dicts
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(sample, dict):
                # Extract 'value' or 'id' if available
                if 'value' in sample:
                    df[f"{col}_name"] = df[col].apply(lambda x: x.get('value') if isinstance(x, dict) else x)
                if 'id' in sample:
                    df[f"{col}_id"] = df[col].apply(lambda x: x.get('id') if isinstance(x, dict) else x)
                # Keep the most useful field and drop the dict column
                if 'value' in sample:
                    df[col] = df[f"{col}_name"]
                    df.drop(f"{col}_name", axis=1, inplace=True, errors='ignore')
            elif isinstance(sample, str) and sample.startswith("{'"):
                # Handle dict strings (malformed output from wbdata)
                LOGGER.warning(f"Column '{col}' contains dict strings, attempting to parse")
                try:
                    import ast
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x).get('value') if isinstance(x, str) and x.startswith("{'") else x)
                except Exception as e:
                    LOGGER.warning(f"Could not parse dict strings in column '{col}': {e}")
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    LOGGER.info("Saved World Bank indicator data to %s", out_path)
    return df


# ---------------------------------------------------------------------------
# WITS bulk-download helpers (local CSV)
# ---------------------------------------------------------------------------


def load_wits_csv(path: Path | str) -> pd.DataFrame:
    """Load a WITS bulk-download CSV into a DataFrame.

    This assumes you have already:
    - Logged into https://wits.worldbank.org/
    - Submitted an Advanced Query (e.g. UN Comtrade, TRAINS)
    - Downloaded the resulting CSV to disk.

    The function simply reads the CSV with pandas and returns it. It does
    not assume a particular schema; you can then inspect columns and, if
    desired, build specialized preprocessors (like the soybean pipeline).
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"WITS CSV not found: {path}")

    LOGGER.info("Loading WITS CSV from %s", path)
    df = pd.read_csv(path)
    LOGGER.info("Loaded WITS CSV with shape %s and columns: %s", df.shape, df.columns.tolist())
    return df


def standardize_wits_trade_csv(
    in_path: Path | str,
    out_path: Path | str,
    year_column_hint: str = "Year",
    partner_column_hint: str = "Partner",
    value_column_hint: str = "Trade Value",
) -> pd.DataFrame:
    """Example helper to normalize a WITS trade CSV into a simpler schema.

    This is intentionally minimal and meant as a starting point. It:
    - Attempts to locate columns whose names contain the given hints
      (case-insensitive).
    - Keeps only those key columns and writes them to a new CSV.

    Args:
        in_path: Input WITS CSV path.
        out_path: Output normalized CSV path.
        year_column_hint: Substring used to find the year column.
        partner_column_hint: Substring used to find the partner column.
        value_column_hint: Substring used to find the trade value column.

    Returns:
        Simplified DataFrame with columns: year, partner, trade_value.
    """

    df = load_wits_csv(in_path)
    lower_to_orig = {str(c).lower(): str(c) for c in df.columns}

    def _find_col(hint: str) -> str:
        hint_l = hint.lower()
        # exact lower-case match
        if hint_l in lower_to_orig:
            return lower_to_orig[hint_l]
        # substring search
        for lower, orig in lower_to_orig.items():
            if hint_l in lower:
                return orig
        raise ValueError(
            f"Could not find column containing '{hint}' in WITS CSV. "
            f"Available columns: {list(df.columns)}"
        )

    year_col = _find_col(year_column_hint)
    partner_col = _find_col(partner_column_hint)
    value_col = _find_col(value_column_hint)

    out = pd.DataFrame(
        {
            "year": pd.to_numeric(df[year_col], errors="coerce").astype("Int64"),
            "partner": df[partner_col].astype("string"),
            "trade_value": pd.to_numeric(df[value_col], errors="coerce"),
        }
    )

    out = out.dropna(subset=["year", "partner", "trade_value"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    LOGGER.info("Saved standardized WITS trade data to %s", out_path)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "World Bank / WITS data helpers: fetch World Bank indicators "
            "and standardize WITS bulk-download CSVs."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: wb (World Bank)
    wb_parser = subparsers.add_parser(
        "wb",
        help="Fetch a World Bank indicator via wbdata and save to CSV.",
    )
    wb_parser.add_argument("indicator", type=str, help="World Bank indicator code.")
    wb_parser.add_argument("--country", type=str, default="CHN", help="Country code (default: CHN).")
    wb_parser.add_argument("--start-year", type=int, default=2015, help="Start year (inclusive).")
    wb_parser.add_argument("--end-year", type=int, default=2024, help="End year (inclusive).")
    wb_parser.add_argument(
        "--output",
        type=str,
        default=str(Path("2025/data/external/worldbank_indicator.csv")),
        help="Output CSV path.",
    )

    # Subcommand: wits (standardize local WITS CSV)
    wits_parser = subparsers.add_parser(
        "wits",
        help="Standardize a WITS bulk-download CSV into a simple schema.",
    )
    wits_parser.add_argument("input", type=str, help="Input WITS CSV path.")
    wits_parser.add_argument(
        "--output",
        type=str,
        default=str(Path("2025/data/external/wits_trade_standardized.csv")),
        help="Output CSV path.",
    )
    wits_parser.add_argument(
        "--year-hint",
        type=str,
        default="Year",
        help="Substring used to detect the year column (default: 'Year').",
    )
    wits_parser.add_argument(
        "--partner-hint",
        type=str,
        default="Partner",
        help="Substring used to detect the partner column (default: 'Partner').",
    )
    wits_parser.add_argument(
        "--value-hint",
        type=str,
        default="Trade Value",
        help=(
            "Substring used to detect the trade value column "
            "(default: 'Trade Value')."
        ),
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "wb":
        fetch_worldbank_indicator_to_csv(
            indicator=args.indicator,
            country=args.country,
            start_year=args.start_year,
            end_year=args.end_year,
            out_path=args.output,
        )
    elif args.command == "wits":
        standardize_wits_trade_csv(
            in_path=args.input,
            out_path=args.output,
            year_column_hint=args.year_hint,
            partner_column_hint=args.partner_hint,
            value_column_hint=args.value_hint,
        )
    else:  # pragma: no cover - defensive
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
