"""Build quarter and annual food-service commercial-area panels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ANALYSIS_YEARS, KEY_COLUMNS
from .load_data import quarter_to_year


def remove_duplicate_keys(data: pd.DataFrame, measures: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove only exact duplicate primary keys; quarantine conflicting duplicates.

    The raw sources remain untouched. Exact duplicate observations are retained
    once for analysis; conflicting duplicates are excluded rather than averaged.
    """
    duplicated = data[data.duplicated(KEY_COLUMNS, keep=False)].copy()
    if duplicated.empty:
        return data.copy(), pd.DataFrame(columns=[*KEY_COLUMNS, "duplicate_resolution"])
    exact_keys: list[tuple[str, str, str]] = []
    conflict_keys: list[tuple[str, str, str]] = []
    for key, group in duplicated.groupby(KEY_COLUMNS, dropna=False):
        compare = group[["area_name", "area_type", "industry_name", *measures]].fillna("<NA>").astype(str)
        (exact_keys if len(compare.drop_duplicates()) == 1 else conflict_keys).append(key)
    exact = pd.DataFrame(exact_keys, columns=KEY_COLUMNS)
    conflict = pd.DataFrame(conflict_keys, columns=KEY_COLUMNS)
    unique = data.loc[~data.duplicated(KEY_COLUMNS, keep=False)].copy()
    if not exact.empty:
        first_exact = data.merge(exact, on=KEY_COLUMNS, how="inner").drop_duplicates(KEY_COLUMNS, keep="first")
        unique = pd.concat([unique, first_exact], ignore_index=True)
    audit = pd.concat([
        exact.assign(duplicate_resolution="exact_duplicate_keep_one"),
        conflict.assign(duplicate_resolution="conflicting_duplicate_excluded"),
    ], ignore_index=True)
    return unique, audit


def build_quarter_panel(sales: pd.DataFrame, stores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join normalized sales and store data at the mandated quarterly key."""
    sales_clean, sales_dupes = remove_duplicate_keys(sales, ["sales_amount", "sales_transactions"])
    stores_clean, stores_dupes = remove_duplicate_keys(stores, ["store_count", "open_count", "close_count"])
    panel = sales_clean.merge(
        stores_clean[[*KEY_COLUMNS, "store_count", "open_count", "close_count", "source_file"]],
        on=KEY_COLUMNS, how="inner", suffixes=("_sales", "_stores"), validate="one_to_one",
    )
    panel["year"] = quarter_to_year(panel["quarter"])
    panel["quarter_no"] = pd.to_numeric(panel["quarter"].astype(str).str[-1], errors="coerce")
    panel["sales_per_store"] = np.where(panel["store_count"] > 0, panel["sales_amount"] / panel["store_count"], np.nan)
    panel["net_entry_rate"] = np.where(panel["store_count"] > 0, (panel["open_count"] - panel["close_count"]) / panel["store_count"], np.nan)
    return panel, pd.concat([sales_dupes.assign(dataset="sales"), stores_dupes.assign(dataset="stores")], ignore_index=True)


def build_annual_panel(quarter_panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete quarterly observations to the specified annual grain."""
    source = quarter_panel.loc[quarter_panel["year"].isin(ANALYSIS_YEARS)].copy()
    group_cols = ["year", "area_code", "industry_code"]
    aggregations = {
        "area_name": "first", "area_type": "first", "industry_name": "first",
        "sales_amount": "sum", "sales_transactions": "sum", "store_count": "mean",
        "open_count": "sum", "close_count": "sum", "quarter": "nunique",
    }
    annual = source.groupby(group_cols, as_index=False).agg(aggregations).rename(columns={"quarter": "observed_quarters"})
    q4 = source.loc[source["quarter_no"].eq(4), [*group_cols, "store_count"]].rename(columns={"store_count": "year_end_store_count"})
    annual = annual.merge(q4, on=group_cols, how="left")
    annual["complete_year"] = annual["observed_quarters"].eq(4)
    annual.loc[~annual["complete_year"], ["sales_amount", "sales_transactions", "store_count", "open_count", "close_count", "year_end_store_count"]] = np.nan
    annual["sales_per_store"] = np.where(annual["store_count"] > 0, annual["sales_amount"] / annual["store_count"], np.nan)
    annual["net_entry_count"] = annual["open_count"] - annual["close_count"]
    annual["net_entry_rate"] = np.where(annual["store_count"] > 0, annual["net_entry_count"] / annual["store_count"], np.nan)
    return annual.sort_values(group_cols).reset_index(drop=True)


def build_early_warning_panel(quarter_panel: pd.DataFrame) -> pd.DataFrame:
    """Create quarter-level input for the optional 2025Q1→2026Q1 check."""
    columns = ["quarter", "area_code", "industry_code", "area_name", "area_type", "industry_name", "sales_amount", "sales_transactions", "store_count", "open_count", "close_count", "sales_per_store", "net_entry_rate"]
    return quarter_panel.loc[quarter_panel["quarter"].isin(["20251", "20261"]), columns].copy()
