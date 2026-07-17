"""Data-quality checks used by the analysis pipeline."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import ANALYSIS_QUARTERS, FOOD_CODES, KEY_COLUMNS


def _issue(issue_type: str, frame: pd.DataFrame, detail: str, columns: Iterable[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["issue_type", "detail", *columns])
    output = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    output.insert(0, "detail", detail)
    output.insert(0, "issue_type", issue_type)
    return output


def validate_sources(sales: pd.DataFrame, stores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return detailed quality issues and an auditable summary.

    No observation is altered here; checks only identify exceptions for reports.
    """
    issues: list[pd.DataFrame] = []
    summary: dict[str, object] = {}
    for label, data, numeric_columns in (
        ("sales", sales, ["sales_amount", "sales_transactions"]),
        ("stores", stores, ["store_count", "open_count", "close_count"]),
    ):
        duplicate_mask = data.duplicated(KEY_COLUMNS, keep=False)
        issues.append(_issue("primary_key_duplicate", data.loc[duplicate_mask], label, [*KEY_COLUMNS, "source_file"]))
        for column in numeric_columns:
            issues.append(_issue("negative_value", data.loc[data[column].lt(0, fill_value=False)], f"{label}.{column}", [*KEY_COLUMNS, column, "source_file"]))
            issues.append(_issue("missing_value", data.loc[data[column].isna()], f"{label}.{column}", [*KEY_COLUMNS, column, "source_file"]))
        summary[f"{label}_rows"] = int(len(data))
        summary[f"{label}_duplicate_rows"] = int(duplicate_mask.sum())
    expected = set(ANALYSIS_QUARTERS)
    actual = set(sales["quarter"].astype(str)).union(stores["quarter"].astype(str))
    summary["missing_analysis_quarters"] = sorted(expected - actual)
    summary["observed_analysis_quarters"] = sorted(expected & actual)
    for quarter in sorted(expected):
        present_codes = set(sales.loc[sales["quarter"].eq(quarter), "industry_code"]).union(
            stores.loc[stores["quarter"].eq(quarter), "industry_code"]
        )
        absent = sorted(set(FOOD_CODES) - present_codes)
        if absent:
            issues.append(pd.DataFrame({"issue_type": "food_industry_absent", "detail": [quarter], "industry_code": [", ".join(absent)]}))
    outer = sales[KEY_COLUMNS].merge(stores[KEY_COLUMNS], on=KEY_COLUMNS, how="outer", indicator=True)
    issues.append(_issue("sales_without_store", outer.loc[outer["_merge"].eq("left_only")], "결합 누락", KEY_COLUMNS))
    issues.append(_issue("store_without_sales", outer.loc[outer["_merge"].eq("right_only")], "결합 누락", KEY_COLUMNS))
    merged = sales.merge(stores[[*KEY_COLUMNS, "store_count"]], on=KEY_COLUMNS, how="inner")
    issues.append(_issue("sales_with_zero_store", merged.loc[merged["store_count"].eq(0) & merged["sales_amount"].gt(0)], "매출 존재·점포수 0", [*KEY_COLUMNS, "sales_amount", "store_count"]))
    area_name_count = pd.concat([sales[["area_code", "area_name"]], stores[["area_code", "area_name"]]]).drop_duplicates().groupby("area_code")["area_name"].nunique()
    conflicting_names = area_name_count[area_name_count.gt(1)].index
    if len(conflicting_names):
        names = pd.concat([sales[["area_code", "area_name"]], stores[["area_code", "area_name"]]]).drop_duplicates()
        issues.append(_issue("area_name_changed", names.loc[names["area_code"].isin(conflicting_names)], "동일 상권코드의 명칭 다수", ["area_code", "area_name"]))
    type_count = pd.concat([sales[["area_code", "area_type"]], stores[["area_code", "area_type"]]]).drop_duplicates().groupby("area_code")["area_type"].nunique()
    conflicting_types = type_count[type_count.gt(1)].index
    if len(conflicting_types):
        types = pd.concat([sales[["area_code", "area_type"]], stores[["area_code", "area_type"]]]).drop_duplicates()
        issues.append(_issue("area_type_changed", types.loc[types["area_code"].isin(conflicting_types)], "동일 상권코드의 유형 다수", ["area_code", "area_type"]))
    result = pd.concat([issue for issue in issues if not issue.empty], ignore_index=True, sort=False) if any(not x.empty for x in issues) else pd.DataFrame(columns=["issue_type", "detail"])
    summary["issue_rows"] = int(len(result))
    summary["sales_store_match_rate"] = float((outer["_merge"] == "both").mean()) if len(outer) else 0.0
    return result, summary
