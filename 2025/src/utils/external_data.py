"""Helpers for external/parameter data used in Q1–Q5.

This module creates and maintains CSV/JSON files under
``2025/data/external`` so that all question pipelines can run with
structured sample data when empirical data are not yet available.

The functions are **idempotent** and will not overwrite non-template
user data (heuristically detected by non-zero values).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import logging

import pandas as pd

from .config import DATA_EXTERNAL

logger = logging.getLogger(__name__)


YEARS_2015_2025: List[int] = list(range(2015, 2026))
YEARS_2020_2024: List[int] = [2020, 2021, 2022, 2023, 2024]


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read CSV if it exists, otherwise return empty DataFrame."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def _is_all_zero(df: pd.DataFrame, cols: List[str]) -> bool:
    """Return True if all specified columns are zero or missing."""
    if df.empty:
        return True
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].fillna(0)
        if (s != 0).any():
            return False
    return True


# ---------------------------------------------------------------------------
# Q1 – Soybeans
# ---------------------------------------------------------------------------


def ensure_q1_external_data() -> pd.DataFrame:
    """Ensure ``china_imports_soybeans.csv`` exists with structured data.

    File: ``2025/data/external/china_imports_soybeans.csv``

    Columns:
        year, exporter, import_value_usd, import_quantity_tonnes,
        tariff_cn_on_exporter

    Returns:
        DataFrame with soybean imports by exporter and year.
    """
    path = DATA_EXTERNAL / "china_imports_soybeans.csv"
    df = _read_csv_if_exists(path)

    is_template = _is_all_zero(df, ["import_value_usd", "import_quantity_tonnes"]) or (
        "year" in df.columns and df["year"].nunique() <= 2
    )
    if not df.empty and not is_template:
        return df

    rows: List[Dict] = []

    value_billion: Dict[int, Dict[str, float]] = {
        2020: {"US": 11.0, "Brazil": 25.0, "Argentina": 6.0},
        2021: {"US": 13.0, "Brazil": 24.5, "Argentina": 6.5},
        2022: {"US": 10.0, "Brazil": 28.0, "Argentina": 7.0},
        2023: {"US": 12.0, "Brazil": 27.5, "Argentina": 7.5},
        2024: {"US": 13.0, "Brazil": 27.0, "Argentina": 8.0},
    }
    qty_million: Dict[int, Dict[str, float]] = {
        2020: {"US": 28.0, "Brazil": 70.0, "Argentina": 15.0},
        2021: {"US": 30.0, "Brazil": 69.0, "Argentina": 16.0},
        2022: {"US": 25.0, "Brazil": 75.0, "Argentina": 17.0},
        2023: {"US": 27.0, "Brazil": 74.0, "Argentina": 17.5},
        2024: {"US": 28.0, "Brazil": 73.0, "Argentina": 18.0},
    }
    tariff: Dict[int, Dict[str, float]] = {
        2020: {"US": 0.25, "Brazil": 0.03, "Argentina": 0.03},
        2021: {"US": 0.20, "Brazil": 0.03, "Argentina": 0.03},
        2022: {"US": 0.30, "Brazil": 0.03, "Argentina": 0.03},
        2023: {"US": 0.28, "Brazil": 0.03, "Argentina": 0.03},
        2024: {"US": 0.25, "Brazil": 0.03, "Argentina": 0.03},
    }

    for year in YEARS_2020_2024:
        for exporter in ("US", "Brazil", "Argentina"):
            rows.append(
                {
                    "year": year,
                    "exporter": exporter,
                    "import_value_usd": value_billion[year][exporter] * 1e9,
                    "import_quantity_tonnes": qty_million[year][exporter] * 1e6,
                    "tariff_cn_on_exporter": tariff[year][exporter],
                }
            )

    df_new = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(path, index=False)
    logger.warning(
        "Created structured SAMPLE data for Q1 at %s. "
        "Replace with empirical China Customs data when available.",
        path,
    )
    return df_new


# ---------------------------------------------------------------------------
# Q2 – Autos
# ---------------------------------------------------------------------------


def ensure_q2_external_data() -> Dict[str, pd.DataFrame]:
    """Ensure Q2 auto external files exist with structured data.

    Files:
        us_auto_sales_by_brand.csv
        us_auto_indicators.csv

    Returns:
        Dict with keys "sales" and "indicators".
    """
    sales_path = DATA_EXTERNAL / "us_auto_sales_by_brand.csv"
    ind_path = DATA_EXTERNAL / "us_auto_indicators.csv"

    sales_df = _read_csv_if_exists(sales_path)
    if sales_df.empty or _is_all_zero(sales_df, ["total_sales"]):
        rows: List[Dict] = []
        brands = ["Toyota", "Honda", "Nissan", "Ford"]
        base_sales = {"Toyota": 1.6, "Honda": 1.3, "Nissan": 1.0, "Ford": 1.2}
        for i, year in enumerate(YEARS_2020_2024):
            growth = 1.0 + 0.03 * max(i - 1, 0)  # slight recovery after 2020
            for brand in brands:
                total = base_sales[brand] * growth  # millions of units
                if brand in {"Toyota", "Honda", "Nissan"}:
                    us_share = 0.20 + 0.05 * i / 4.0
                    mx_share = 0.35
                else:
                    us_share = 0.75 + 0.02 * i / 4.0
                    mx_share = 0.10
                jp_share = max(0.0, 1.0 - us_share - mx_share)
                rows.append(
                    {
                        "year": year,
                        "brand": brand,
                        "total_sales": total * 1e6,
                        "us_produced": total * us_share * 1e6,
                        "mexico_produced": total * mx_share * 1e6,
                        "japan_imported": total * jp_share * 1e6,
                    }
                )
        sales_df = pd.DataFrame(rows)
        sales_path.parent.mkdir(parents=True, exist_ok=True)
        sales_df.to_csv(sales_path, index=False)
        logger.warning(
            "Created structured SAMPLE data for Q2 sales at %s. Replace with "
            "brand-level empirical data when available.",
            sales_path,
        )

    ind_df = _read_csv_if_exists(ind_path)
    if ind_df.empty or _is_all_zero(ind_df, ["us_auto_production"]):
        rows2: List[Dict] = []
        prod_million = [8.0, 8.6, 8.9, 9.2, 9.4]
        emp_thousand = [900, 915, 925, 935, 945]
        price_index = [100, 102, 105, 108, 111]
        gdp = [21000, 22000, 23000, 23800, 24500]
        fuel_idx = [90, 95, 120, 110, 105]  # oil price spike around 2022
        for year, p, e, pi, g, f in zip(
            YEARS_2020_2024, prod_million, emp_thousand, price_index, gdp, fuel_idx
        ):
            rows2.append(
                {
                    "year": year,
                    "us_auto_production": p * 1e6,
                    "us_auto_employment": e * 1e3,
                    "us_auto_price_index": pi,
                    "us_gdp_billions": g,
                    "fuel_price_index": f,
                }
            )
        ind_df = pd.DataFrame(rows2)
        ind_path.parent.mkdir(parents=True, exist_ok=True)
        ind_df.to_csv(ind_path, index=False)
        logger.warning(
            "Created structured SAMPLE data for Q2 indicators at %s.", ind_path
        )

    return {"sales": sales_df, "indicators": ind_df}


# ---------------------------------------------------------------------------
# Q3 – Semiconductors
# ---------------------------------------------------------------------------


def ensure_q3_external_data() -> Dict[str, pd.DataFrame]:
    """Ensure Q3 semiconductor external files exist with structured data."""
    out_path = DATA_EXTERNAL / "us_semiconductor_output.csv"
    pol_path = DATA_EXTERNAL / "us_chip_policies.csv"

    out_df = _read_csv_if_exists(out_path)
    if out_df.empty or _is_all_zero(out_df, ["us_chip_output_billions"]):
        rows: List[Dict] = []
        for year in YEARS_2020_2024:
            demand_index = 100 + (year - 2020) * 5
            base = {"high": 20.0, "mid": 30.0, "low": 15.0}
            growth_factor = 1.0 + 0.04 * (year - 2020)
            for seg in ("high", "mid", "low"):
                rows.append(
                    {
                        "year": year,
                        "segment": seg,
                        "us_chip_output_billions": base[seg] * growth_factor,
                        "global_chip_demand_index": demand_index,
                    }
                )
        out_df = pd.DataFrame(rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        logger.warning(
            "Created structured SAMPLE data for Q3 output at %s.", out_path
        )

    pol_df = _read_csv_if_exists(pol_path)
    if pol_df.empty or _is_all_zero(pol_df, ["subsidy_index"]):
        pol_df = pd.DataFrame(
            {
                "year": YEARS_2020_2024,
                "subsidy_index": [0, 0, 5, 10, 15],
                "export_control_china": [0, 0, 1, 1, 1],
            }
        )
        pol_path.parent.mkdir(parents=True, exist_ok=True)
        pol_df.to_csv(pol_path, index=False)
        logger.warning(
            "Created structured SAMPLE data for Q3 policies at %s.", pol_path
        )

    return {"output": out_df, "policies": pol_df}


# ---------------------------------------------------------------------------
# Q4 – Tariff Revenue
# ---------------------------------------------------------------------------


def ensure_q4_external_data() -> None:
    """Ensure Q4 parameter files (CSV/JSON) exist with structured values."""
    # Average tariff by year
    avg_path = DATA_EXTERNAL / "q4_avg_tariff_by_year.csv"
    avg_df = _read_csv_if_exists(avg_path)
    if avg_df.empty or _is_all_zero(avg_df, ["avg_tariff"]):
        start, end = 0.0244, 0.2011
        years = YEARS_2015_2025
        steps = len(years) - 1
        rows: List[Dict] = []
        for i, year in enumerate(years):
            t = start + (end - start) * i / steps
            rows.append({"year": year, "avg_tariff": round(t, 4)})
        avg_df = pd.DataFrame(rows)
        avg_path.parent.mkdir(parents=True, exist_ok=True)
        avg_df.to_csv(avg_path, index=False)
        logger.warning("Created SAMPLE avg_tariff path at %s.", avg_path)

    # Dynamic import elasticities
    dyn_path = DATA_EXTERNAL / "q4_dynamic_import_params.json"
    dyn: Dict[str, float]
    if dyn_path.exists():
        try:
            with open(dyn_path, "r") as f:
                dyn = json.load(f)
        except Exception:  # pragma: no cover - defensive
            dyn = {}
    else:
        dyn = {}
    keys = {"short_run_elasticity", "medium_run_elasticity", "adjustment_speed"}
    if not keys.issubset(dyn.keys()):
        dyn = {
            "short_run_elasticity": -1.2,
            "medium_run_elasticity": -2.0,
            "adjustment_speed": 0.6,
        }
        dyn_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dyn_path, "w") as f:
            json.dump(dyn, f, indent=2)
        logger.warning("Created SAMPLE dynamic import elasticities at %s.", dyn_path)

    # Tariff scenarios
    scen_path = DATA_EXTERNAL / "q4_tariff_scenarios.json"
    scen: Dict
    if scen_path.exists():
        try:
            with open(scen_path, "r") as f:
                scen = json.load(f)
        except Exception:  # pragma: no cover
            scen = {}
    else:
        scen = {}
    if "base_import_value" not in scen or "scenarios" not in scen:
        years = [2025, 2026, 2027, 2028, 2029]
        scen = {
            "base_import_value": 3_000_000_000_000.0,
            "years": years,
            "scenarios": {
                "baseline": [0.025, 0.026, 0.027, 0.028, 0.029],
                "reciprocal_tariff": [0.025, 0.10, 0.12, 0.13, 0.14],
            },
        }
        scen_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scen_path, "w") as f:
            json.dump(scen, f, indent=2)
        logger.warning("Created SAMPLE tariff scenarios at %s.", scen_path)


# ---------------------------------------------------------------------------
# Q5 – Macro / Finance / Reshoring
# ---------------------------------------------------------------------------


def ensure_q5_external_data() -> Dict[str, pd.DataFrame]:
    """Ensure Q5 macro/financial/reshoring external data exist."""
    macro_path = DATA_EXTERNAL / "us_macro.csv"
    fin_path = DATA_EXTERNAL / "us_financial.csv"
    resh_path = DATA_EXTERNAL / "us_reshoring.csv"
    ret_path = DATA_EXTERNAL / "retaliation_index.csv"

    macro_df = _read_csv_if_exists(macro_path)
    if macro_df.empty or _is_all_zero(macro_df, ["gdp_growth"]):
        rows: List[Dict] = []
        gdp = [2.5, 2.4, 2.3, 2.2, 2.1, -3.0, 5.5, 2.8, 2.3, 2.0, 1.8]
        ip = [100, 101, 102, 103, 104, 96, 102, 104, 105, 106, 107]
        u = [5.5, 5.3, 5.0, 4.5, 3.8, 8.0, 5.5, 4.5, 4.2, 4.0, 3.9]
        cpi = [240, 244, 248, 252, 256, 259, 264, 276, 283, 289, 295]
        for year, g, ipi, ur, cp in zip(YEARS_2015_2025, gdp, ip, u, cpi):
            rows.append(
                {
                    "year": year,
                    "gdp_growth": g,
                    "industrial_production": ipi,
                    "unemployment_rate": ur,
                    "cpi": cp,
                }
            )
        macro_df = pd.DataFrame(rows)
        macro_path.parent.mkdir(parents=True, exist_ok=True)
        macro_df.to_csv(macro_path, index=False)
        logger.warning("Created structured SAMPLE macro data at %s.", macro_path)

    fin_df = _read_csv_if_exists(fin_path)
    if fin_df.empty or _is_all_zero(fin_df, ["dollar_index"]):
        rows2: List[Dict] = []
        dxy = [95, 96, 97, 95, 93, 99, 92, 94, 96, 98, 99]
        y10 = [2.2, 2.4, 2.6, 2.9, 2.5, 1.0, 1.5, 2.0, 2.8, 3.5, 3.8]
        spx = [2100, 2200, 2350, 2500, 2700, 2600, 2900, 3200, 3500, 3800, 4000]
        crypto = [400, 600, 900, 1500, 3000, 8000, 40000, 20000, 25000, 28000, 30000]
        for year, dx, y, s, c in zip(YEARS_2015_2025, dxy, y10, spx, crypto):
            rows2.append(
                {
                    "year": year,
                    "dollar_index": dx,
                    "treasury_yield_10y": y,
                    "sp500_index": s,
                    "crypto_index": c,
                }
            )
        fin_df = pd.DataFrame(rows2)
        fin_path.parent.mkdir(parents=True, exist_ok=True)
        fin_df.to_csv(fin_path, index=False)
        logger.warning("Created structured SAMPLE financial data at %s.", fin_path)

    resh_df = _read_csv_if_exists(resh_path)
    if resh_df.empty or _is_all_zero(resh_df, ["manufacturing_va_share"]):
        rows3: List[Dict] = []
        va = [12.0, 11.9, 11.8, 11.7, 11.6, 11.5, 11.6, 11.7, 11.8, 11.9, 12.1]
        emp = [9.5, 9.4, 9.3, 9.2, 9.1, 9.0, 9.0, 9.1, 9.2, 9.3, 9.5]
        fdi = [10, 11, 12, 13, 12, 11, 15, 18, 20, 23, 26]
        for year, v, e, f in zip(YEARS_2015_2025, va, emp, fdi):
            rows3.append(
                {
                    "year": year,
                    "manufacturing_va_share": v,
                    "manufacturing_employment_share": e,
                    "reshoring_fdi_billions": f,
                }
            )
        resh_df = pd.DataFrame(rows3)
        resh_path.parent.mkdir(parents=True, exist_ok=True)
        resh_df.to_csv(resh_path, index=False)
        logger.warning("Created structured SAMPLE reshoring data at %s.", resh_path)

    ret_df = _read_csv_if_exists(ret_path)
    if ret_df.empty or _is_all_zero(ret_df, ["retaliation_index"]):
        ret_df = pd.DataFrame(
            {
                "year": YEARS_2015_2025,
                "retaliation_index": [
                    0,
                    0,
                    1,
                    2,
                    1,
                    1,
                    3,
                    5,
                    6,
                    8,
                    10,
                ],
            }
        )
        ret_path.parent.mkdir(parents=True, exist_ok=True)
        ret_df.to_csv(ret_path, index=False)
        logger.warning("Created structured SAMPLE retaliation index at %s.", ret_path)

    return {
        "macro": macro_df,
        "financial": fin_df,
        "reshoring": resh_df,
        "retaliation": ret_df,
    }


# ---------------------------------------------------------------------------
# Unified entry
# ---------------------------------------------------------------------------


def ensure_all_external_data() -> None:
    """Ensure all Q1–Q5 external datasets and parameters exist.

    This is safe to call multiple times; it will only create or overwrite
    files that appear to be empty/template data.
    """
    ensure_q1_external_data()
    ensure_q2_external_data()
    ensure_q3_external_data()
    ensure_q4_external_data()
    ensure_q5_external_data()
