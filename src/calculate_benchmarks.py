"""Build Seoul same-industry benchmark series from commercial-area data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def annual_benchmarks(annual: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete commercial-area records to Seoul industry benchmarks."""
    complete = annual.loc[annual["complete_year"]].copy()
    group_cols = ["year", "industry_code"]
    benchmark = complete.groupby(group_cols, as_index=False).agg(
        seoul_sales_amount=("sales_amount", "sum"),
        seoul_sales_transactions=("sales_transactions", "sum"),
        seoul_avg_store_count=("store_count", "sum"),
        seoul_year_end_store_count=("year_end_store_count", "sum"),
        seoul_open_count=("open_count", "sum"),
        seoul_close_count=("close_count", "sum"),
        source_area_count=("area_code", "nunique"),
    )
    benchmark["seoul_sales_per_store"] = np.where(
        benchmark["seoul_avg_store_count"] > 0, benchmark["seoul_sales_amount"] / benchmark["seoul_avg_store_count"], np.nan
    )
    benchmark["seoul_net_entry_rate"] = np.where(
        benchmark["seoul_avg_store_count"] > 0,
        (benchmark["seoul_open_count"] - benchmark["seoul_close_count"]) / benchmark["seoul_avg_store_count"], np.nan,
    )
    return benchmark


def quarterly_benchmarks(quarter: pd.DataFrame) -> pd.DataFrame:
    """Aggregate quarter records for an optional non-annualized early-warning score."""
    benchmark = quarter.groupby(["quarter", "industry_code"], as_index=False).agg(
        seoul_sales_amount=("sales_amount", "sum"),
        seoul_sales_transactions=("sales_transactions", "sum"),
        seoul_store_count=("store_count", "sum"),
        seoul_open_count=("open_count", "sum"),
        seoul_close_count=("close_count", "sum"),
    )
    benchmark["seoul_sales_per_store"] = np.where(benchmark["seoul_store_count"] > 0, benchmark["seoul_sales_amount"] / benchmark["seoul_store_count"], np.nan)
    benchmark["seoul_net_entry_rate"] = np.where(
        benchmark["seoul_store_count"] > 0,
        (benchmark["seoul_open_count"] - benchmark["seoul_close_count"]) / benchmark["seoul_store_count"], np.nan,
    )
    return benchmark
