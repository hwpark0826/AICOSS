"""Relative-performance calculation and industry-weighted area aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import METRIC_WEIGHTS, PERIODS


MEASURES = {
    "sales_rel": ("sales_amount", "seoul_sales_amount", "sales_amount"),
    "transactions_rel": ("sales_transactions", "seoul_sales_transactions", "sales_amount"),
    "sales_per_store_rel": ("sales_per_store", "seoul_sales_per_store", "sales_amount"),
    "store_rel": ("year_end_store_count", "seoul_year_end_store_count", "store_count"),
}


def _log_relative(start: pd.Series, end: pd.Series, benchmark_start: pd.Series, benchmark_end: pd.Series) -> pd.Series:
    valid = (start > 0) & (end > 0) & (benchmark_start > 0) & (benchmark_end > 0)
    output = pd.Series(np.nan, index=start.index, dtype=float)
    output.loc[valid] = np.log(end.loc[valid] / start.loc[valid]) - np.log(benchmark_end.loc[valid] / benchmark_start.loc[valid])
    return output


def industry_relative_metrics(annual: pd.DataFrame, benchmarks: pd.DataFrame, period_name: str) -> pd.DataFrame:
    """Compute one period's area-industry relative measures at the required grain."""
    start_year, end_year = PERIODS[period_name]
    merged = annual.merge(benchmarks, on=["year", "industry_code"], how="left")
    start = merged.loc[merged["year"].eq(start_year)].set_index(["area_code", "industry_code"])
    end = merged.loc[merged["year"].eq(end_year)].set_index(["area_code", "industry_code"])
    both = start.join(end, how="inner", lsuffix="_start", rsuffix="_end")
    output = both.reset_index()[["area_code", "industry_code", "area_name_start", "area_type_start", "industry_name_start"]].rename(columns={
        "area_name_start": "area_name", "area_type_start": "area_type", "industry_name_start": "industry_name",
    })
    for metric, (area_column, benchmark_column, _) in MEASURES.items():
        output[metric] = _log_relative(
            both[f"{area_column}_start"], both[f"{area_column}_end"],
            both[f"{benchmark_column}_start"], both[f"{benchmark_column}_end"],
        ).to_numpy()
    # A flow ratio is not log-transformed. Use the mean annual rate over each period,
    # then compare the area with the matching Seoul industry benchmark.
    rate_data = merged.loc[merged["year"].between(start_year, end_year)].copy()
    area_rate = rate_data.groupby(["area_code", "industry_code"], as_index=False)["net_entry_rate"].mean().rename(columns={"net_entry_rate": "area_period_net_entry_rate"})
    seoul_rate = rate_data.groupby("industry_code", as_index=False)["seoul_net_entry_rate"].mean().rename(columns={"seoul_net_entry_rate": "seoul_period_net_entry_rate"})
    output = output.merge(area_rate, on=["area_code", "industry_code"], how="left").merge(seoul_rate, on="industry_code", how="left")
    output["net_entry_rel"] = output["area_period_net_entry_rate"] - output["seoul_period_net_entry_rate"]
    output["period"] = period_name
    return output


def aggregate_to_area(industry_metrics: pd.DataFrame, annual: pd.DataFrame, period_name: str) -> pd.DataFrame:
    """Weight industry relative performance using start-year sales or store shares."""
    start_year, _ = PERIODS[period_name]
    base = annual.loc[annual["year"].eq(start_year), ["area_code", "industry_code", "sales_amount", "store_count"]].rename(
        columns={"sales_amount": "start_sales", "store_count": "start_stores"}
    )
    # Coverage must be measured against *all* observed start-year food
    # industries, including industries that cannot produce an end-point log
    # metric.  Using only the joined metric rows would overstate coverage.
    baseline_totals = base.groupby("area_code", as_index=False).agg(
        baseline_sales_weight=("start_sales", lambda x: x.where(x > 0).sum()),
        baseline_store_weight=("start_stores", lambda x: x.where(x > 0).sum()),
    )
    metrics = industry_metrics.merge(base, on=["area_code", "industry_code"], how="left")
    rows: list[dict[str, object]] = []
    for area_code, group in metrics.groupby("area_code", dropna=False):
        row: dict[str, object] = {
            "area_code": area_code,
            "area_name": group["area_name"].dropna().iloc[0] if group["area_name"].notna().any() else None,
            "area_type": group["area_type"].dropna().iloc[0] if group["area_type"].notna().any() else None,
            "period": period_name,
            "valid_industry_count": int(group[["sales_rel", "transactions_rel", "sales_per_store_rel", "store_rel", "net_entry_rel"]].notna().any(axis=1).sum()),
        }
        totals = baseline_totals.loc[baseline_totals["area_code"].eq(area_code)].iloc[0]
        sales_total = totals["baseline_sales_weight"]
        store_total = totals["baseline_store_weight"]
        sales_valid_union = group.loc[group[["sales_rel", "transactions_rel", "sales_per_store_rel"]].notna().all(axis=1) & group["start_sales"].gt(0), "start_sales"].sum()
        store_valid_union = group.loc[group[["store_rel", "net_entry_rel"]].notna().all(axis=1) & group["start_stores"].gt(0), "start_stores"].sum()
        row["sales_weight_coverage"] = sales_valid_union / sales_total if sales_total > 0 else np.nan
        row["store_weight_coverage"] = store_valid_union / store_total if store_total > 0 else np.nan
        for metric in METRIC_WEIGHTS:
            is_sales_metric = metric in {"sales_rel", "transactions_rel", "sales_per_store_rel"}
            weight_column = "start_sales" if is_sales_metric else "start_stores"
            valid = group[metric].notna() & group[weight_column].gt(0)
            denominator = group.loc[valid, weight_column].sum()
            row[metric] = (group.loc[valid, metric] * group.loc[valid, weight_column]).sum() / denominator if denominator > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_relative_area_metrics(annual: pd.DataFrame, benchmarks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return both industry-level provenance and weighted area-level measures."""
    industry_parts = [industry_relative_metrics(annual, benchmarks, period) for period in PERIODS]
    industry = pd.concat(industry_parts, ignore_index=True)
    areas = pd.concat([aggregate_to_area(industry.loc[industry["period"].eq(period)], annual, period) for period in PERIODS], ignore_index=True)
    return industry, areas
