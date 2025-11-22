"""HTTP helpers to download official datasets for the Q1–Q5 pipelines."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd
import requests

# Ensure relative imports work when running the file directly
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils.config import DATA_EXTERNAL  # type: ignore
else:
    from .config import DATA_EXTERNAL

logger = logging.getLogger(__name__)

COMTRADE_BASE_URL = "https://comtradeplus.un.org/api/v1/getComtradeData"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_DEFAULT_MAX_RETRIES = 3
FRED_DEFAULT_RETRY_SLEEP = 1.5
DEFAULT_STATUS_FORCELIST = (429, 500, 502, 503, 504, 520, 521, 522, 524)
DEFAULT_REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "SPEC-data-fetch/1.0 (data_fetch.py)",
}


class DataFetchError(RuntimeError):
    """Raised when an HTTP download fails or returns empty data."""


def _resolve_output_path(filename: str | Path, data_dir: Path = DATA_EXTERNAL) -> Path:
    path = data_dir / filename if not Path(filename).is_absolute() else Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _require_api_key(explicit_key: Optional[str], env_var: str) -> str:
    key = explicit_key or os.getenv(env_var)
    if not key:
        raise DataFetchError(
            f"API key missing: pass `api_key=` or set the {env_var} environment variable."
        )
    return key


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _log_failure(name: str, payload: Mapping[str, Any]) -> None:
    log_path = DATA_EXTERNAL / "failed_downloads.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"dataset": name, **payload}, ensure_ascii=False) + "\n")


def _http_get_with_retry(
    url: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 30,
    max_retries: int = 3,
    retry_sleep: float = 1.5,
    status_forcelist: Sequence[int] = DEFAULT_STATUS_FORCELIST,
) -> requests.Response:
    attempt = 0
    last_exc: Optional[requests.RequestException] = None
    while attempt < max_retries:
        attempt += 1
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code in status_forcelist:
                raise requests.HTTPError(
                    f"HTTP {response.status_code} for {response.url}", response=response
                )
            return response
        except requests.RequestException as exc:  # pragma: no cover - network
            last_exc = exc
            if attempt >= max_retries:
                break
            logger.warning(
                "HTTP request to %s failed (%s/%s): %s; retrying in %.1fs",
                url,
                attempt,
                max_retries,
                exc,
                retry_sleep,
            )
            time.sleep(retry_sleep)

    assert last_exc is not None
    raise last_exc


def fetch_un_comtrade_soybeans(
    *,
    start_year: int = 2015,
    end_year: int = 2024,
    reporter_code: str | int = "156",  # China
    partner_code: str = "all",
    hs_code: str = "1201",
    flow_code: str = "M",  # Imports into reporter
    out_filename: str | Path = "china_imports_soybeans_official.csv",
    include_desc: bool = True,
    max_records: int = 5000,
    max_pages: int = 5,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Download soybean imports (HS 1201) from UN Comtrade+ API.

    Args:
        start_year: First year to include.
        end_year: Last year to include.
        reporter_code: Comtrade reporter code (156 for China).
        partner_code: Partner code ("all" downloads all exporters).
        hs_code: Commodity code (HS4/HS6). 1201 = soybeans.
        flow_code: "M" (imports) or "X" (exports).
        out_filename: Relative/absolute path for the CSV output.
        include_desc: Whether to request verbose text descriptions.
        max_records: Safety guard for API response size.
        api_key: Optional override for the UN_COMTRADE_API_KEY env variable.

    Returns:
        DataFrame with columns year/importer/exporter/value/quantity/hs_code.
    """

    subscription_key = _require_api_key(api_key, "UN_COMTRADE_API_KEY")

    params: MutableMapping[str, str | int] = {
        "typeCode": "C",
        "freqCode": "A",
        "clCode": "HS",
        "reporterCode": reporter_code,
        "partnerCode": partner_code,
        "cmdCode": hs_code,
        "flowCode": flow_code,
        "startYear": start_year,
        "endYear": end_year,
        "fmt": "JSON",
        "countOnly": "false",
        "includeDesc": str(include_desc).lower(),
        "pageNumber": 1,
        "pageSize": max_records,
    }

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    request_headers = {**DEFAULT_REQUEST_HEADERS, **headers}

    rows: List[MutableMapping[str, object]] = []
    page = 0
    while True:
        page += 1
        if page > max_pages:
            break
        params["pageNumber"] = page

        try:
            response = _http_get_with_retry(
                COMTRADE_BASE_URL,
                params=params,
                headers=request_headers,
                timeout=60,
                max_retries=5,
                retry_sleep=2.0,
            )
        except requests.RequestException as exc:  # pragma: no cover - network0
            _log_failure(
                "q1_china_soybean_imports",
                {"error": str(exc), "page": page},
            )
            raise DataFetchError(f"Comtrade request failed: {exc}") from exc

        content_type = response.headers.get("Content-Type", "").lower()
        if "json" not in content_type:
            snippet = response.text[:400]
            _log_failure(
                "q1_china_soybean_imports",
                {
                    "error": "non_json_response",
                    "snippet": snippet,
                    "page": page,
                    "status": response.status_code,
                    "content_type": content_type,
                },
            )
            raise DataFetchError(
                "Comtrade response was not JSON (likely gateway redirect). "
                "Check failed_downloads.jsonl for details."
            )

        try:
            payload = response.json()
        except ValueError:  # pragma: no cover - network
            snippet = response.text[:400]
            _log_failure(
                "q1_china_soybean_imports",
                {
                    "error": "invalid_json",
                    "snippet": snippet,
                    "page": page,
                    "status": response.status_code,
                },
            )
            raise DataFetchError(
                "Comtrade response contained invalid JSON. Check failed_downloads.jsonl for details."
            )

        dataset: List[Mapping[str, object]] = payload.get("dataset", [])
        if not dataset:
            if page == 1:
                raise DataFetchError(
                    "Comtrade returned no rows. Check parameters or API quota."
                )
            break

        for item in dataset:
            qty = _safe_float(item.get("qty"))
            qty_unit = (item.get("qtyUnitAbbr") or "").lower()
            qty_tonnes = qty / 1000.0 if qty and qty_unit in {"kg", "kilograms"} else qty

            rows.append(
                {
                    "year": int(item.get("period")),
                    "importer": item.get("rtTitle") or item.get("rt3ISO") or reporter_code,
                    "exporter": item.get("ptTitle") or item.get("pt3ISO"),
                    "import_value_usd": _safe_float(item.get("primaryValue")),
                    "import_quantity_tonnes": qty_tonnes,
                    "hs_code": item.get("cmdCode"),
                    "flow": item.get("flowDesc") or item.get("flowName"),
                }
            )

        meta = payload.get("meta", {}) or {}
        if not meta or not meta.get("nextPage"):
            break

    if not rows:
        raise DataFetchError("Comtrade produced zero records after pagination.")

    df = pd.DataFrame(rows).sort_values(["year", "exporter"]).reset_index(drop=True)
    output_path = _resolve_output_path(out_filename)
    df.to_csv(output_path, index=False)
    logger.info("Saved Comtrade soybeans data → %s", output_path)
    return df


def fetch_fred_series(
    series_id: str,
    *,
    start_year: int,
    end_year: int,
    frequency: Optional[str] = "a",
    aggregation_method: str = "avg",
    output_filename: str | Path | None = None,
    units: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = FRED_DEFAULT_MAX_RETRIES,
    retry_sleep: float = FRED_DEFAULT_RETRY_SLEEP,
) -> pd.DataFrame:
    """Download a FRED time series for the requested window.

    Args:
        series_id: FRED series identifier (e.g., "TOTALSA").
        start_year: First year to request.
        end_year: Final year to request.
        frequency: Override frequency ("a", "q", "m", etc.). None keeps native.
        aggregation_method: Aggregation scheme when frequency is overridden.
        output_filename: Optional CSV name; defaults to `<series_id>_fred.csv`.
        units: Optional FRED unit transform (e.g., "pc1" for pct change).
        api_key: Optional override for the FRED_API_KEY env variable.
        max_retries: Number of HTTP retry attempts on failure (default 3).
        retry_sleep: Seconds to wait between retries (default 1.5s).

    Returns:
        DataFrame of observations saved under data/external/.
    """

    params: MutableMapping[str, str] = {
        "series_id": series_id,
        "observation_start": f"{start_year}-01-01",
        "observation_end": f"{end_year}-12-31",
        "file_type": "json",
    }

    if frequency:
        params["frequency"] = frequency
        params["aggregation_method"] = aggregation_method
    if units:
        params["units"] = units

    fred_key = api_key or os.getenv("FRED_API_KEY")
    if fred_key:
        params["api_key"] = fred_key

    try:
        response = _http_get_with_retry(
            FRED_BASE_URL,
            params=params,
            headers=DEFAULT_REQUEST_HEADERS,
            timeout=30,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
    except requests.RequestException as exc:  # pragma: no cover - network
        raise DataFetchError(f"FRED request failed: {exc}") from exc

    payload = response.json()
    if "error_message" in payload:
        raise DataFetchError(payload["error_message"])

    observations = payload.get("observations", [])
    if not observations:
        raise DataFetchError("FRED returned no observations for the given window.")

    records = []
    for obs in observations:
        value = obs.get("value", "")
        records.append(
            {
                "series_id": series_id,
                "date": obs.get("date"),
                "year": int(obs.get("date", "0000")[:4]),
                "value": None if value in {"", "."} else float(value),
            }
        )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    filename = output_filename or f"{series_id.lower()}_fred.csv"
    output_path = _resolve_output_path(filename)
    df.to_csv(output_path, index=False)
    logger.info("Saved FRED series %s → %s", series_id, output_path)
    return df


DEFAULT_DATASETS: List[dict[str, Any]] = [
    {
        "name": "q1_china_soybean_imports",
        "group": "Q1",
        "source": "UN Comtrade",
        "description": "China soybean imports by partner (HS1201)",
        "output": "china_imports_soybeans_official.csv",
        "runner": partial(
            fetch_un_comtrade_soybeans,
            start_year=2015,
            end_year=2024,
            out_filename="china_imports_soybeans_official.csv",
        ),
    },
    {
        "name": "q2_total_light_vehicle_sales",
        "group": "Q2",
        "source": "FRED",
        "description": "Total light vehicle sales (TOTALSA)",
        "output": "us_total_light_vehicle_sales_official.csv",
        "runner": partial(
            fetch_fred_series,
            series_id="TOTALSA",
            start_year=2015,
            end_year=2024,
            frequency="a",
            aggregation_method="avg",
            output_filename="us_total_light_vehicle_sales_official.csv",
        ),
    },
    {
        "name": "q2_motor_vehicle_retail_sales",
        "group": "Q2",
        "source": "FRED",
        "description": "Motor vehicle & parts retail sales (MRTSSM441USN)",
        "output": "us_motor_vehicle_retail_sales_official.csv",
        "runner": partial(
            fetch_fred_series,
            series_id="MRTSSM441USN",
            start_year=2015,
            end_year=2024,
            frequency="a",
            aggregation_method="sum",
            output_filename="us_motor_vehicle_retail_sales_official.csv",
        ),
    },
    {
        "name": "q3_semiconductor_output_index",
        "group": "Q3",
        "source": "FRED",
        "description": "Sectoral output index NAICS 3344 (IPUEN3344T300000000)",
        "output": "us_semiconductor_output_index_official.csv",
        "runner": partial(
            fetch_fred_series,
            series_id="IPUEN3344T300000000",
            start_year=2015,
            end_year=2024,
            frequency="a",
            aggregation_method="avg",
            output_filename="us_semiconductor_output_index_official.csv",
        ),
    },
    {
        "name": "q5_real_gdp",
        "group": "Q5",
        "source": "FRED",
        "description": "Real GDP (GDPC1)",
        "output": "us_real_gdp_official.csv",
        "runner": partial(
            fetch_fred_series,
            series_id="GDPC1",
            start_year=2015,
            end_year=2024,
            frequency="a",
            aggregation_method="avg",
            output_filename="us_real_gdp_official.csv",
        ),
    },
    {
        "name": "q5_cpi",
        "group": "Q5",
        "source": "FRED",
        "description": "Consumer Price Index (CPIAUCSL)",
        "output": "us_cpi_official.csv",
        "runner": partial(
            fetch_fred_series,
            series_id="CPIAUCSL",
            start_year=2015,
            end_year=2024,
            frequency="a",
            aggregation_method="avg",
            output_filename="us_cpi_official.csv",
        ),
    },
    {
        "name": "q5_unemployment_rate",
        "group": "Q5",
        "source": "FRED",
        "description": "Unemployment rate (UNRATE)",
        "output": "us_unemployment_rate_official.csv",
        "runner": partial(
            fetch_fred_series,
            series_id="UNRATE",
            start_year=2015,
            end_year=2024,
            frequency="a",
            aggregation_method="avg",
            output_filename="us_unemployment_rate_official.csv",
        ),
    },
    {
        "name": "q5_industrial_production",
        "group": "Q5",
        "source": "FRED",
        "description": "Industrial Production index (INDPRO)",
        "output": "us_industrial_production_official.csv",
        "runner": partial(
            fetch_fred_series,
            series_id="INDPRO",
            start_year=2015,
            end_year=2024,
            frequency="a",
            aggregation_method="avg",
            output_filename="us_industrial_production_official.csv",
        ),
    },
    {
        "name": "q5_federal_funds_rate",
        "group": "Q5",
        "source": "FRED",
        "description": "Effective federal funds rate (FEDFUNDS)",
        "output": "us_federal_funds_rate_official.csv",
        "runner": partial(
            fetch_fred_series,
            series_id="FEDFUNDS",
            start_year=2015,
            end_year=2024,
            frequency="a",
            aggregation_method="avg",
            output_filename="us_federal_funds_rate_official.csv",
        ),
    },
    {
        "name": "q5_treasury_10y_yield",
        "group": "Q5",
        "source": "FRED",
        "description": "10Y Treasury yield (DGS10)",
        "output": "us_treasury_10y_yield_official.csv",
        "runner": partial(
            fetch_fred_series,
            series_id="DGS10",
            start_year=2015,
            end_year=2024,
            frequency="a",
            aggregation_method="avg",
            output_filename="us_treasury_10y_yield_official.csv",
        ),
    },
    {
        "name": "q5_sp500_index",
        "group": "Q5",
        "source": "FRED",
        "description": "S&P 500 index level (SP500)",
        "output": "us_sp500_index_official.csv",
        "runner": partial(
            fetch_fred_series,
            series_id="SP500",
            start_year=2015,
            end_year=2024,
            frequency="a",
            aggregation_method="avg",
            output_filename="us_sp500_index_official.csv",
        ),
    },
]

DATASET_NAMES = [item["name"] for item in DEFAULT_DATASETS]
DATASET_GROUPS = sorted({item["group"] for item in DEFAULT_DATASETS})


def _render_dataset_catalog() -> str:
    lines = ["name | group | source | output | description", "-" * 80]
    for ds in DEFAULT_DATASETS:
        lines.append(
            f"{ds['name']:<32} | {ds['group']:<3} | {ds['source']:<11} | "
            f"{ds['output']:<38} | {ds['description']}"
        )
    return "\n".join(lines)


def _select_dataset_names(
    dataset_args: Optional[Sequence[str]],
    groups: Optional[Sequence[str]],
) -> List[str]:
    if not dataset_args and not groups:
        return list(DATASET_NAMES)

    selected: List[str] = []
    if dataset_args:
        if any(name == "all" for name in dataset_args):
            selected.extend(DATASET_NAMES)
        else:
            for name in dataset_args:
                if name in DATASET_NAMES and name not in selected:
                    selected.append(name)
    if groups:
        for group in groups:
            for ds in DEFAULT_DATASETS:
                if ds["group"] == group and ds["name"] not in selected:
                    selected.append(ds["name"])
    return selected


def run_cli(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download official external datasets for Q1–Q5"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(set(DATASET_NAMES + ["all"])),
        help="Dataset names to download (default: all)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=DATASET_GROUPS,
        help="Filter by question groups (e.g., Q1 Q5)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print dataset catalog and exit",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at first failure",
    )

    args = parser.parse_args(argv)

    if args.list:
        print(_render_dataset_catalog())
        return

    selected = _select_dataset_names(args.datasets, args.groups)
    if not selected:
        parser.error("No datasets selected. Use --list to inspect availability.")

    for ds in DEFAULT_DATASETS:
        if ds["name"] not in selected:
            continue
        logger.info("Downloading %s (%s)", ds["name"], ds["description"])
        try:
            ds["runner"]()
            logger.info("✅ Saved to %s", DATA_EXTERNAL / ds["output"])
        except Exception as exc:  # pragma: no cover - network I/O
            logger.error("❌ %s failed: %s", ds["name"], exc)
            if args.fail_fast:
                raise


__all__ = [
    "DataFetchError",
    "fetch_un_comtrade_soybeans",
    "fetch_fred_series",
]


if __name__ == "__main__":
    run_cli()

