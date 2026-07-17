"""Run the falsification checks requested for ``증산역 4번``.

The module deliberately separates evidence that can be established from the
project files from hypotheses that need a separate external source.  It does
not manufacture zeroes for an industry whose source row is absent.

Run after the core ranking pipeline:

    python -m src.run_analysis
    python -m src.validation.run_validation
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Iterable
import os

# The normal user profile cache is not writable in the sandbox.  Keep the
# matplotlib cache local to the reproducible project instead.
_MPL_CACHE = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile

plt.rcParams["axes.unicode_minus"] = False

from ..build_panel import remove_duplicate_keys
from ..calculate_benchmarks import annual_benchmarks
from ..calculate_scores import build_eligibility, calculate_ranking
from ..config import (
    ANALYSIS_QUARTERS,
    FIGURE_DIR,
    FOOD_CODES,
    MATCHED_CONTROL_COUNT,
    METRIC_WEIGHTS,
    PERIODS,
    RAW_DIR,
    REPORT_DIR,
    TABLE_DIR,
    OUTPUT_DIR,
    TARGET_AREA_CODE,
    TARGET_AREA_NAME,
    VALIDATION_ALL_QUARTERS,
)
from ..export_results import markdown_table, save_csv, write_text
from ..load_data import discover_inputs, load_area_data, read_area_reference


METRICS = list(METRIC_WEIGHTS)
SALES_METRICS = {"sales_rel", "transactions_rel", "sales_per_store_rel"}
TARGET_CODE = str(TARGET_AREA_CODE)
SPATIAL_FILE = RAW_DIR / "서울시 상권분석서비스(영역-상권).shp"
EPSILON = 1e-12
JEUNGSAN_OUTPUT_DIR = OUTPUT_DIR / "jeungsan4"
JEUNGSAN_TABLE_DIR = JEUNGSAN_OUTPUT_DIR / "tables"
JEUNGSAN_FIGURE_DIR = JEUNGSAN_OUTPUT_DIR / "figures"
JEUNGSAN_REPORT_DIR = REPORT_DIR / "jeungsan4"
for directory in (JEUNGSAN_OUTPUT_DIR, JEUNGSAN_TABLE_DIR, JEUNGSAN_FIGURE_DIR, JEUNGSAN_REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".0", "", regex=False).str.strip()


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if pd.notna(numerator) and pd.notna(denominator) and denominator != 0 else np.nan


def _pct_change(start: float, end: float) -> float:
    return end / start - 1 if pd.notna(start) and start != 0 and pd.notna(end) else np.nan


def _log_change(start: float, end: float) -> float:
    return float(np.log(end / start)) if pd.notna(start) and pd.notna(end) and start > 0 and end > 0 else np.nan


def _existing(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    for column in ("area_code", "industry_code", "quarter"):
        if column in data:
            data[column] = _code(data[column])
    return data


def _md_number(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "검증 불가"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{digits}f}"
    return str(value)


def _target_name(data: pd.DataFrame) -> str:
    names = data.loc[data["area_code"].eq(TARGET_CODE), "area_name"].dropna().unique()
    return str(names[0]) if len(names) else TARGET_AREA_NAME


def _metric_log_relative(start: pd.Series, end: pd.Series, b_start: pd.Series, b_end: pd.Series) -> pd.Series:
    valid = (start > 0) & (end > 0) & (b_start > 0) & (b_end > 0)
    result = pd.Series(np.nan, index=start.index, dtype=float)
    result.loc[valid] = np.log(end.loc[valid] / start.loc[valid]) - np.log(b_end.loc[valid] / b_start.loc[valid])
    return result


def _period_industry_metrics(annual: pd.DataFrame, start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute five relative metrics for an arbitrary complete-year interval."""
    benchmark = annual_benchmarks(annual)
    merged = annual.merge(benchmark, on=["year", "industry_code"], how="left")
    start = merged.loc[merged["year"].eq(start_year)].set_index(["area_code", "industry_code"])
    end = merged.loc[merged["year"].eq(end_year)].set_index(["area_code", "industry_code"])
    both = start.join(end, how="inner", lsuffix="_start", rsuffix="_end")
    columns = ["area_name_start", "area_type_start", "industry_name_start"]
    output = both.reset_index()[["area_code", "industry_code", *columns]].rename(columns={
        "area_name_start": "area_name", "area_type_start": "area_type", "industry_name_start": "industry_name",
    })
    pairs = {
        "sales_rel": ("sales_amount", "seoul_sales_amount"),
        "transactions_rel": ("sales_transactions", "seoul_sales_transactions"),
        "sales_per_store_rel": ("sales_per_store", "seoul_sales_per_store"),
        "store_rel": ("year_end_store_count", "seoul_year_end_store_count"),
    }
    for metric, (area_column, benchmark_column) in pairs.items():
        output[metric] = _metric_log_relative(
            both[f"{area_column}_start"], both[f"{area_column}_end"],
            both[f"{benchmark_column}_start"], both[f"{benchmark_column}_end"],
        ).to_numpy()
    rate = merged.loc[merged["year"].between(start_year, end_year)].copy()
    area_rate = rate.groupby(["area_code", "industry_code"], as_index=False)["net_entry_rate"].mean().rename(columns={"net_entry_rate": "area_rate"})
    seoul_rate = rate.groupby("industry_code", as_index=False)["seoul_net_entry_rate"].mean().rename(columns={"seoul_net_entry_rate": "seoul_rate"})
    output = output.merge(area_rate, on=["area_code", "industry_code"], how="left").merge(seoul_rate, on="industry_code", how="left")
    output["net_entry_rel"] = output["area_rate"] - output["seoul_rate"]
    base = annual.loc[annual["year"].eq(start_year), ["area_code", "industry_code", "area_name", "area_type", "industry_name", "sales_amount", "store_count"]].copy()
    base = base.rename(columns={"sales_amount": "start_sales", "store_count": "start_stores"})
    return output, base


def _aggregate_period(
    metrics: pd.DataFrame,
    base: pd.DataFrame,
    label: str,
    fill_missing_relative_with_zero: bool = False,
) -> pd.DataFrame:
    """Start-year weighted aggregation with an explicit missingness policy."""
    cols = ["area_code", "industry_code", "area_name", "area_type", "industry_name", "start_sales", "start_stores"]
    joined = base[cols].merge(metrics[["area_code", "industry_code", *METRICS]], on=["area_code", "industry_code"], how="left")
    if fill_missing_relative_with_zero:
        joined[METRICS] = joined[METRICS].fillna(0.0)
    output: list[dict[str, object]] = []
    for area_code, group in joined.groupby("area_code", dropna=False):
        row: dict[str, object] = {
            "area_code": str(area_code),
            "area_name": group["area_name"].dropna().iloc[0] if group["area_name"].notna().any() else pd.NA,
            "area_type": group["area_type"].dropna().iloc[0] if group["area_type"].notna().any() else pd.NA,
            "period": label,
        }
        row["valid_industry_count"] = int(group[METRICS].notna().any(axis=1).sum())
        sales_total = group.loc[group["start_sales"].gt(0), "start_sales"].sum()
        store_total = group.loc[group["start_stores"].gt(0), "start_stores"].sum()
        sales_valid = group.loc[group[["sales_rel", "transactions_rel", "sales_per_store_rel"]].notna().all(axis=1) & group["start_sales"].gt(0), "start_sales"].sum()
        store_valid = group.loc[group[["store_rel", "net_entry_rel"]].notna().all(axis=1) & group["start_stores"].gt(0), "start_stores"].sum()
        row["sales_weight_coverage"] = _safe_ratio(sales_valid, sales_total)
        row["store_weight_coverage"] = _safe_ratio(store_valid, store_total)
        for metric in METRICS:
            weight = "start_sales" if metric in SALES_METRICS else "start_stores"
            valid = group[metric].notna() & group[weight].gt(0)
            denominator = group.loc[valid, weight].sum()
            row[metric] = _safe_ratio((group.loc[valid, metric] * group.loc[valid, weight]).sum(), denominator)
        output.append(row)
    return pd.DataFrame(output)


def _scenario_area_metrics(annual: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Build the three-period score input for a stated industry-missingness rule."""
    period_rows: list[pd.DataFrame] = []
    common_keys: pd.DataFrame | None = None
    if mode == "common_all_periods":
        history = annual.loc[annual["year"].between(2021, 2025)].copy()
        valid = history[["sales_amount", "sales_transactions", "store_count", "year_end_store_count"]].gt(0).all(axis=1)
        history = history.loc[history["complete_year"] & valid]
        common_keys = history.groupby(["area_code", "industry_code"], as_index=False)["year"].nunique().query("year == 5")[["area_code", "industry_code"]]
    for label, (start, end) in PERIODS.items():
        metrics, base = _period_industry_metrics(annual, start, end)
        if common_keys is not None:
            metrics = metrics.merge(common_keys, on=["area_code", "industry_code"], how="inner")
            base = base.merge(common_keys, on=["area_code", "industry_code"], how="inner")
        period_rows.append(_aggregate_period(metrics, base, label, fill_missing_relative_with_zero=mode == "seoul_average"))
    return pd.concat(period_rows, ignore_index=True)


def _score_scenario(annual: pd.DataFrame, panel: pd.DataFrame, area_metrics: pd.DataFrame, coverage_80: bool = False) -> pd.DataFrame:
    eligibility = build_eligibility(annual, panel, area_metrics)
    if coverage_80:
        eligibility = eligibility.copy()
        eligibility["base_eligible"] = eligibility["base_eligible"] & eligibility["sales_weight_coverage"].ge(0.80)
    return calculate_ranking(area_metrics, eligibility)


def _scenario_target_summary(label: str, ranking: pd.DataFrame, baseline_rank: float, rule: str) -> dict[str, object]:
    row = ranking.loc[ranking["area_code"].eq(TARGET_CODE)]
    if row.empty:
        return {"scenario": label, "rule": rule, "eligible_areas": len(ranking), "target_included": False,
                "target_rank": np.nan, "rank_change_vs_baseline": np.nan, "target_score": np.nan,
                "top20": False}
    current = row.iloc[0]
    return {"scenario": label, "rule": rule, "eligible_areas": len(ranking), "target_included": True,
            "target_rank": int(current["overall_rank"]), "rank_change_vs_baseline": int(current["overall_rank"] - baseline_rank),
            "target_score": current["CoreDeclineScore"], "top20": bool(current["overall_rank"] <= 20)}


def _presence_matrix(raw_sales: pd.DataFrame, raw_stores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a full target quarter×industry state matrix and citywide endpoint frequency."""
    sales, _ = remove_duplicate_keys(raw_sales, ["sales_amount", "sales_transactions"])
    stores, _ = remove_duplicate_keys(raw_stores, ["store_count", "open_count", "close_count"])
    sales = sales.loc[sales["area_code"].eq(TARGET_CODE), ["quarter", "industry_code", "industry_name", "sales_amount", "sales_transactions"]].copy()
    stores = stores.loc[stores["area_code"].eq(TARGET_CODE), ["quarter", "industry_code", "industry_name", "store_count", "open_count", "close_count"]].copy()
    grid = pd.MultiIndex.from_product([VALIDATION_ALL_QUARTERS, list(FOOD_CODES)], names=["quarter", "industry_code"]).to_frame(index=False)
    matrix = grid.merge(sales, on=["quarter", "industry_code"], how="left", suffixes=("", "_sales"))
    matrix = matrix.merge(stores, on=["quarter", "industry_code"], how="left", suffixes=("_sales", "_store"))
    matrix["industry_name"] = matrix["industry_name_sales"].combine_first(matrix["industry_name_store"])
    matrix["sales_row_present"] = matrix["sales_amount"].notna()
    matrix["store_row_present"] = matrix["store_count"].notna()
    matrix["sales_zero"] = matrix["sales_row_present"] & matrix["sales_amount"].eq(0)
    matrix["store_zero"] = matrix["store_row_present"] & matrix["store_count"].eq(0)
    matrix["state"] = np.select(
        [matrix["sales_row_present"] & matrix["store_row_present"] & ~(matrix["sales_zero"] | matrix["store_zero"]),
         matrix["sales_row_present"] & matrix["store_row_present"], matrix["sales_row_present"], matrix["store_row_present"]],
        ["both_present_positive", "both_present_with_zero", "sales_only", "store_only"], default="both_rows_missing",
    )
    matrix = matrix.drop(columns=["industry_name_sales", "industry_name_store"])

    raw_sales = raw_sales.copy()
    raw_stores = raw_stores.copy()
    raw_sales["year"] = raw_sales["quarter"].str[:4]
    raw_stores["year"] = raw_stores["quarter"].str[:4]
    start = raw_sales.loc[raw_sales["year"].eq("2021")].groupby(["area_code", "industry_code"], as_index=False)["quarter"].nunique().rename(columns={"quarter": "sales_quarters_2021"})
    end = raw_sales.loc[raw_sales["year"].eq("2025")].groupby(["area_code", "industry_code"], as_index=False)["quarter"].nunique().rename(columns={"quarter": "sales_quarters_2025"})
    city = start.merge(end, on=["area_code", "industry_code"], how="left").fillna({"sales_quarters_2025": 0})
    city["observed_2021_full"] = city["sales_quarters_2021"].eq(4)
    city["absent_2025"] = city["sales_quarters_2025"].eq(0)
    frequency = city.groupby("industry_code", as_index=False).agg(
        area_industry_with_2021_sales=("area_code", "size"),
        full_2021_observed=("observed_2021_full", "sum"),
        absent_sales_2025=("absent_2025", "sum"),
    )
    frequency["absent_2025_share"] = frequency["absent_sales_2025"] / frequency["area_industry_with_2021_sales"]
    return matrix, frequency


def _quarterly_series(panel: pd.DataFrame) -> pd.DataFrame:
    target = panel.loc[panel["area_code"].eq(TARGET_CODE) & panel["quarter"].isin(VALIDATION_ALL_QUARTERS)].copy()
    grouped = target.groupby("quarter", as_index=False).agg(
        industry_rows=("industry_code", "nunique"), sales_amount=("sales_amount", "sum"),
        sales_transactions=("sales_transactions", "sum"), store_count=("store_count", "sum"),
        open_count=("open_count", "sum"), close_count=("close_count", "sum"),
    )
    result = pd.DataFrame({"quarter": VALIDATION_ALL_QUARTERS}).merge(grouped, on="quarter", how="left")
    result["year"] = result["quarter"].str[:4].astype(int)
    result["quarter_no"] = result["quarter"].str[-1].astype(int)
    result["sales_per_store"] = result.apply(lambda x: _safe_ratio(x.sales_amount, x.store_count), axis=1)
    result["transactions_per_store"] = result.apply(lambda x: _safe_ratio(x.sales_transactions, x.store_count), axis=1)
    result["sales_per_transaction"] = result.apply(lambda x: _safe_ratio(x.sales_amount, x.sales_transactions), axis=1)
    result["net_entry_count"] = result["open_count"] - result["close_count"]
    result["net_entry_rate"] = result.apply(lambda x: _safe_ratio(x.net_entry_count, x.store_count), axis=1)
    for metric in ("sales_amount", "sales_transactions", "store_count", "sales_per_store", "transactions_per_store", "sales_per_transaction"):
        result[f"{metric}_yoy"] = result[metric].pct_change(4, fill_method=None)
    result["sales_amount_4q_ma"] = result["sales_amount"].rolling(4, min_periods=4).mean()
    return result


def _seasonal_outliers(series: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ["sales_amount", "sales_transactions", "store_count", "sales_per_store", "transactions_per_store", "sales_per_transaction"]
    for quarter_no in range(1, 5):
        peer = series.loc[series["year"].between(2022, 2025) & series["quarter_no"].eq(quarter_no)]
        observed = series.loc[series["year"].eq(2021) & series["quarter_no"].eq(quarter_no)]
        if observed.empty:
            continue
        for metric in metrics:
            value = observed.iloc[0][metric]
            values = peer[metric].dropna()
            median = values.median()
            mad = (values - median).abs().median()
            z = 0.67448975 * (value - median) / mad if pd.notna(value) and mad and np.isfinite(mad) else np.nan
            rows.append({"quarter": observed.iloc[0]["quarter"], "metric": metric, "value_2021": value,
                         "peer_2022_2025_median": median, "peer_2022_2025_mad": mad, "robust_z": z,
                         "outlier_abs_z_ge_3_5": bool(abs(z) >= 3.5) if pd.notna(z) else False})
    return pd.DataFrame(rows)


def _single_period_rank(annual: pd.DataFrame, panel: pd.DataFrame, start: int, end: int, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, base = _period_industry_metrics(annual, start, end)
    aggregate = _aggregate_period(metrics, base, label)
    eligibility = build_eligibility(annual, panel, pd.concat([
        aggregate.assign(period="long"), aggregate.assign(period="medium"), aggregate.assign(period="recent")
    ], ignore_index=True))
    # The copied periods above only construct the candidate gate.  This score is
    # intentionally one-period, so each of the five components is standardized once.
    candidate = eligibility.loc[eligibility["base_eligible"], ["area_code", "area_name", "area_type"]].merge(aggregate, on=["area_code", "area_name", "area_type"], how="left")
    candidate = candidate.loc[candidate[METRICS].notna().all(axis=1)].copy()
    for metric in METRICS:
        value = -candidate[metric].astype(float)
        low, high = value.quantile(.01), value.quantile(.99)
        value = value.clip(low, high)
        mad = (value - value.median()).abs().median()
        candidate[f"{metric}_score"] = 0.67448975 * (value - value.median()) / mad if mad else (value - value.mean()) / value.std(ddof=0)
    candidate["period_decline_score"] = sum(METRIC_WEIGHTS[metric] * candidate[f"{metric}_score"] for metric in METRICS)
    candidate["period_rank"] = candidate["period_decline_score"].rank(method="min", ascending=False).astype(int)
    return candidate.sort_values("period_rank"), aggregate


def _annual_total(annual: pd.DataFrame, codes: Iterable[str], years: Iterable[int], label: str) -> pd.DataFrame:
    subset = annual.loc[annual["area_code"].eq(TARGET_CODE) & annual["industry_code"].isin(list(codes)) & annual["year"].isin(list(years))].copy()
    aggregate = subset.groupby("year", as_index=False).agg(
        industry_count=("industry_code", "nunique"), sales_amount=("sales_amount", "sum"),
        sales_transactions=("sales_transactions", "sum"), store_count=("store_count", "sum"),
        year_end_store_count=("year_end_store_count", "sum"), open_count=("open_count", "sum"), close_count=("close_count", "sum"),
    )
    aggregate["sales_per_store"] = aggregate.apply(lambda r: _safe_ratio(r.sales_amount, r.store_count), axis=1)
    aggregate["transactions_per_store"] = aggregate.apply(lambda r: _safe_ratio(r.sales_transactions, r.store_count), axis=1)
    aggregate["sales_per_transaction"] = aggregate.apply(lambda r: _safe_ratio(r.sales_amount, r.sales_transactions), axis=1)
    aggregate["net_entry_count"] = aggregate["open_count"] - aggregate["close_count"]
    aggregate["comparison"] = label
    return aggregate


def _absolute_changes(annual: pd.DataFrame) -> pd.DataFrame:
    target = annual.loc[annual["area_code"].eq(TARGET_CODE) & annual["year"].isin([2021, 2025])].copy()
    start_codes = set(target.loc[target["year"].eq(2021), "industry_code"])
    end_codes = set(target.loc[target["year"].eq(2025), "industry_code"])
    comparisons = [
        ("all_observed_endpoint", start_codes | end_codes, "endpoint industry composition differs; not a pure same-industry comparison"),
        ("common_observed_2021_2025", start_codes & end_codes, "only industries with complete annual observations at both endpoints"),
        ("korean_food_only", {"CS100001"}, "한식 only"),
        ("snack_food_only", {"CS100008"}, "분식 only"),
        ("chicken_only", {"CS100007"}, "치킨 only"),
    ]
    rows: list[dict[str, object]] = []
    for label, codes, note in comparisons:
        totals = _annual_total(annual, codes, [2021, 2025], label).set_index("year")
        for metric in ["sales_amount", "sales_transactions", "store_count", "year_end_store_count", "sales_per_store", "transactions_per_store", "sales_per_transaction", "open_count", "close_count", "net_entry_count"]:
            start = totals.at[2021, metric] if 2021 in totals.index else np.nan
            end = totals.at[2025, metric] if 2025 in totals.index else np.nan
            rows.append({"comparison": label, "industry_codes": ",".join(sorted(codes)), "metric": metric,
                         "start_2021": start, "end_2025": end, "absolute_change": end - start if pd.notna(start) and pd.notna(end) else np.nan,
                         "percent_change": _pct_change(start, end), "comparability_note": note})
    return pd.DataFrame(rows)


def _match_controls(annual: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = annual.loc[annual["year"].eq(2021) & annual["complete_year"]].copy()
    total = start.groupby(["area_code", "area_name", "area_type"], as_index=False).agg(
        total_sales=("sales_amount", "sum"), total_transactions=("sales_transactions", "sum"), total_stores=("store_count", "sum"),
    )
    korean = start.loc[start["industry_code"].eq("CS100001")].groupby("area_code", as_index=False)["sales_amount"].sum().rename(columns={"sales_amount": "korean_sales"})
    total = total.merge(korean, on="area_code", how="left").fillna({"korean_sales": 0})
    total["korean_sales_share"] = total["korean_sales"] / total["total_sales"]
    total["sales_per_store"] = total["total_sales"] / total["total_stores"]
    target = total.loc[total["area_code"].eq(TARGET_CODE)].copy()
    if target.empty:
        raise ValueError("Target commercial area is absent from 2021 annual panel.")
    target_type = target.iloc[0]["area_type"]
    features = ["total_stores", "total_sales", "korean_sales_share", "total_transactions", "sales_per_store"]
    candidates = total.loc[total["area_type"].eq(target_type) & ~total["area_code"].eq(TARGET_CODE)].dropna(subset=features).copy()
    pooled = pd.concat([target, candidates], ignore_index=True)
    numeric_features = pooled[features].astype(float)
    std = numeric_features.std(ddof=0).replace(0, 1.0)
    target_vector = target.iloc[0][features].astype(float)
    candidates["matching_distance"] = np.sqrt(
        ((candidates[features].astype(float) - target_vector) / std).pow(2).sum(axis=1).astype(float)
    )
    selected = candidates.nsmallest(MATCHED_CONTROL_COUNT, "matching_distance").copy()
    selected["selection"] = "nearest neighbour; same area type; standardized 2021 features"
    result = pd.concat([target.assign(matching_distance=0.0, selection="target"), selected], ignore_index=True)
    if not reference.empty:
        result = result.merge(reference[[c for c in ["area_code", "district", "administrative_dong"] if c in reference]], on="area_code", how="left")
    balances: list[dict[str, object]] = []
    for feature in features:
        target_value = target.iloc[0][feature]
        control_mean = selected[feature].mean()
        pooled_sd = np.sqrt((target[feature].var(ddof=0) + selected[feature].var(ddof=0)) / 2)
        balances.append({"feature": feature, "target_2021": target_value, "controls_mean_2021": control_mean,
                         "difference": target_value - control_mean, "standardized_mean_difference": _safe_ratio(target_value - control_mean, pooled_sd),
                         "control_min": selected[feature].min(), "control_max": selected[feature].max()})
    return result, pd.DataFrame(balances)


def _decomposition_row(label: str, data: pd.DataFrame, codes: set[str]) -> dict[str, object]:
    subset = data.loc[data["industry_code"].isin(codes) & data["year"].isin([2021, 2025])].groupby("year", as_index=False).agg(
        sales_amount=("sales_amount", "sum"), sales_transactions=("sales_transactions", "sum"), store_count=("store_count", "sum"),
    ).set_index("year")
    start = subset.loc[2021] if 2021 in subset.index else pd.Series(dtype=float)
    end = subset.loc[2025] if 2025 in subset.index else pd.Series(dtype=float)
    def get(row: pd.Series, column: str) -> float:
        return float(row.get(column, np.nan))
    stores_start, stores_end = get(start, "store_count"), get(end, "store_count")
    tx_start, tx_end = get(start, "sales_transactions"), get(end, "sales_transactions")
    sales_start, sales_end = get(start, "sales_amount"), get(end, "sales_amount")
    tx_store_start, tx_store_end = _safe_ratio(tx_start, stores_start), _safe_ratio(tx_end, stores_end)
    sales_tx_start, sales_tx_end = _safe_ratio(sales_start, tx_start), _safe_ratio(sales_end, tx_end)
    return {
        "scope": label, "industry_codes": ",".join(sorted(codes)), "sales_2021": sales_start, "sales_2025": sales_end,
        "stores_2021": stores_start, "stores_2025": stores_end, "transactions_2021": tx_start, "transactions_2025": tx_end,
        "log_sales_change": _log_change(sales_start, sales_end), "log_store_count_contribution": _log_change(stores_start, stores_end),
        "log_transactions_per_store_contribution": _log_change(tx_store_start, tx_store_end),
        "log_sales_per_transaction_contribution": _log_change(sales_tx_start, sales_tx_end),
    }


def _sales_decomposition(annual: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    target = annual.loc[annual["area_code"].eq(TARGET_CODE)]
    start_codes = set(target.loc[target["year"].eq(2021), "industry_code"])
    end_codes = set(target.loc[target["year"].eq(2025), "industry_code"])
    common = start_codes & end_codes
    benchmark = annual_benchmarks(annual).rename(columns={
        "seoul_sales_amount": "sales_amount", "seoul_sales_transactions": "sales_transactions", "seoul_avg_store_count": "store_count",
    })[["year", "industry_code", "sales_amount", "sales_transactions", "store_count"]]
    control_data = annual.loc[annual["area_code"].isin(controls.loc[controls["area_code"].ne(TARGET_CODE), "area_code"])]
    rows = [
        _decomposition_row("target_common_industries", target, common),
        _decomposition_row("seoul_same_industries", benchmark, common),
        _decomposition_row("matched_controls_same_industries", control_data, common),
    ]
    for code in ["CS100001", "CS100008", "CS100007"]:
        rows.append(_decomposition_row(f"target_{FOOD_CODES[code]}", target, {code}))
    result = pd.DataFrame(rows)
    result["decomposition_residual"] = result["log_sales_change"] - result[[
        "log_store_count_contribution", "log_transactions_per_store_contribution", "log_sales_per_transaction_contribution"
    ]].sum(axis=1, min_count=3)
    return result


@dataclass
class PolygonRecord:
    area_code: str
    area_name: str
    area_type: str
    district: str
    dong: str
    reported_centroid_x: float
    reported_centroid_y: float
    reported_area: float
    parts: list[np.ndarray]

    @property
    def centroid(self) -> tuple[float, float]:
        return self.reported_centroid_x, self.reported_centroid_y

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        points = np.vstack(self.parts)
        return float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())


def _rings(shape: shapefile.Shape) -> list[np.ndarray]:
    points = np.asarray(shape.points, dtype=float)
    starts = list(shape.parts) + [len(points)]
    return [points[starts[i]:starts[i + 1]] for i in range(len(starts) - 1) if len(points[starts[i]:starts[i + 1]]) >= 3]


def _load_polygons() -> dict[str, PolygonRecord]:
    if not SPATIAL_FILE.exists():
        raise FileNotFoundError(f"Polygon file not found: {SPATIAL_FILE.name}")
    reader = shapefile.Reader(str(SPATIAL_FILE), encoding="utf-8")
    result: dict[str, PolygonRecord] = {}
    for record, shape in zip(reader.iterRecords(), reader.iterShapes()):
        row = record.as_dict()
        code = str(row["TRDAR_CD"]).strip()
        result[code] = PolygonRecord(
            code, str(row["TRDAR_CD_N"]), str(row["TRDAR_SE_1"]), str(row["SIGNGU_CD_"]), str(row["ADSTRD_CD_"]),
            float(row["XCNTS_VALU"]), float(row["YDNTS_VALU"]), float(row["RELM_AR"]), _rings(shape),
        )
    return result


def _ring_area_centroid(ring: np.ndarray) -> tuple[float, float, float]:
    if not np.array_equal(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    x, y = ring[:, 0], ring[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    signed_area = cross.sum() / 2
    if abs(signed_area) < EPSILON:
        return 0.0, float(x[:-1].mean()), float(y[:-1].mean())
    return signed_area, float(((x[:-1] + x[1:]) * cross).sum() / (6 * signed_area)), float(((y[:-1] + y[1:]) * cross).sum() / (6 * signed_area))


def _polygon_properties(poly: PolygonRecord) -> tuple[float, float, float, float]:
    components = [_ring_area_centroid(ring) for ring in poly.parts]
    # Shapefile rings are oriented; use signed areas so inner rings subtract.
    signed = sum(item[0] for item in components)
    area = abs(signed)
    if abs(signed) < EPSILON:
        x, y = poly.centroid
    else:
        x = sum(item[0] * item[1] for item in components) / signed
        y = sum(item[0] * item[2] for item in components) / signed
    perimeter = sum(np.linalg.norm(np.diff(np.vstack([ring, ring[0]]) if not np.array_equal(ring[0], ring[-1]) else ring, axis=0), axis=1).sum() for ring in poly.parts)
    return area, perimeter, x, y


def _bbox_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return hypot(dx, dy)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    length = float(np.dot(vector, vector))
    if length == 0:
        return float(np.linalg.norm(point - start))
    t = min(1.0, max(0.0, float(np.dot(point - start, vector) / length)))
    return float(np.linalg.norm(point - (start + t * vector)))


def _boundary_distance(a: PolygonRecord, b: PolygonRecord) -> float:
    """Conservative vertex-to-edge distance; enough for the 10 m adjacency flag."""
    if _bbox_distance(a.bbox, b.bbox) > 10:
        return _bbox_distance(a.bbox, b.bbox)
    best = float("inf")
    for first, second in ((a, b), (b, a)):
        for ring in first.parts:
            for point in ring:
                for other in second.parts:
                    closed = np.vstack([other, other[0]]) if not np.array_equal(other[0], other[-1]) else other
                    for i in range(len(closed) - 1):
                        best = min(best, _point_segment_distance(point, closed[i], closed[i + 1]))
    return best


def _neighbor_table(polygons: dict[str, PolygonRecord], annual: pd.DataFrame) -> pd.DataFrame:
    target = polygons[TARGET_CODE]
    target_bbox = target.bbox
    annual_codes = set(annual["area_code"])
    rows: list[dict[str, object]] = []
    for code, poly in polygons.items():
        if code == TARGET_CODE or code not in annual_codes:
            continue
        centroid_distance = hypot(poly.centroid[0] - target.centroid[0], poly.centroid[1] - target.centroid[1])
        bbox_distance = _bbox_distance(poly.bbox, target_bbox)
        boundary_distance = _boundary_distance(target, poly) if bbox_distance <= 10 else bbox_distance
        name_candidate = any(word.lower() in poly.area_name.lower() for word in ["증산", "수색", "북가좌", "dmc"])
        same_dong = poly.dong == target.dong
        adjacent = boundary_distance <= 10
        within_1km = centroid_distance <= 1000
        if not (adjacent or within_1km or same_dong or name_candidate):
            continue
        reasons = "; ".join(name for name, yes in [
            ("boundary_within_10m", adjacent), ("centroid_within_1km", within_1km), ("same_administrative_dong", same_dong), ("name_candidate", name_candidate)
        ] if yes)
        rows.append({"area_code": code, "area_name": poly.area_name, "area_type": poly.area_type, "district": poly.district,
                     "administrative_dong": poly.dong, "centroid_distance_m": centroid_distance,
                     "boundary_distance_m": boundary_distance, "adjacent_boundary_10m": adjacent,
                     "within_centroid_1km": within_1km, "same_administrative_dong": same_dong,
                     "name_candidate": name_candidate, "selection_reason": reasons})
    return pd.DataFrame(rows).sort_values(["centroid_distance_m", "area_code"]).reset_index(drop=True)


def _boundary_outputs(polygons: dict[str, PolygonRecord], neighbors: pd.DataFrame) -> pd.DataFrame:
    target = polygons[TARGET_CODE]
    area, perimeter, centroid_x, centroid_y = _polygon_properties(target)
    matching = [code for code, item in polygons.items() if item.area_code == TARGET_CODE or item.area_name == target.area_name]
    frame = pd.DataFrame([{
        "target_area_code": TARGET_CODE, "target_area_name": target.area_name, "polygon_feature_count_for_code": sum(item.area_code == TARGET_CODE for item in polygons.values()),
        "matching_code_or_name_feature_count": len(matching), "polygon_area_m2_calculated": area, "dbf_reported_area_m2": target.reported_area,
        "area_difference_m2": area - target.reported_area, "perimeter_m": perimeter,
        "polygon_centroid_x": centroid_x, "polygon_centroid_y": centroid_y,
        "dbf_centroid_x": target.reported_centroid_x, "dbf_centroid_y": target.reported_centroid_y,
        "centroid_difference_m": hypot(centroid_x - target.reported_centroid_x, centroid_y - target.reported_centroid_y),
        "coordinate_reference": "Korea_2000_Korea_Central_Belt (metre units; source .prj)",
        "spatial_snapshot_count": 1, "boundary_change_verifiable": False,
        "store_point_data_available": False, "nearby_selected_area_count": len(neighbors),
    }])
    return frame


def _font() -> str | None:
    candidates = [Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/NanumGothic.ttf")]
    for candidate in candidates:
        if candidate.exists():
            return font_manager.FontProperties(fname=str(candidate)).get_name()
    return None


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_boundary(polygons: dict[str, PolygonRecord], neighbors: pd.DataFrame) -> None:
    target = polygons[TARGET_CODE]
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    fig, ax = plt.subplots(figsize=(8, 7))
    selected = neighbors.loc[neighbors["within_centroid_1km"] | neighbors["same_administrative_dong"]]
    for code in selected["area_code"]:
        poly = polygons[code]
        for ring in poly.parts:
            ax.plot(ring[:, 0], ring[:, 1], color="#8b8b8b", linewidth=.8)
    for ring in target.parts:
        ax.fill(ring[:, 0], ring[:, 1], facecolor="#d73027", edgecolor="#7f0000", alpha=.45, linewidth=1.6)
    ax.scatter(*target.centroid, color="#7f0000", s=22, zorder=3)
    ax.annotate(target.area_name, target.centroid, xytext=(5, 5), textcoords="offset points", fontsize=10, weight="bold")
    ax.set_title("증산역 4번 및 선택된 공간 비교권역\n(원본 상권 폴리곤, 좌표 단위 m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    _savefig(JEUNGSAN_FIGURE_DIR / "jeungsan4_boundary_map.png")


def _plot_quarterly(series: pd.DataFrame) -> None:
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    x = np.arange(len(series))
    labels = series["quarter"].str[:4] + "Q" + series["quarter"].str[-1]
    items = [("sales_amount", "총매출액"), ("sales_transactions", "거래건수"), ("store_count", "운영점포 수"), ("sales_per_store", "점포당 매출")]
    for axis, (metric, title) in zip(axes.flat, items):
        axis.plot(x, series[metric], marker="o", linewidth=1.6)
        axis.set_title(title)
        axis.grid(alpha=.25)
        axis.ticklabel_format(style="plain", axis="y")
    axes[-1, 0].set_xticks(x[::2], labels[::2], rotation=45, ha="right")
    axes[-1, 1].set_xticks(x[::2], labels[::2], rotation=45, ha="right")
    fig.suptitle("증산역 4번 분기별 추이 (2021Q1–2026Q1)", y=1.02)
    _savefig(JEUNGSAN_FIGURE_DIR / "jeungsan4_quarterly_trends.png")


def _plot_decomposition(data: pd.DataFrame) -> None:
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    shown = data.loc[data["scope"].isin(["target_common_industries", "seoul_same_industries", "matched_controls_same_industries"])].copy()
    cols = ["log_store_count_contribution", "log_transactions_per_store_contribution", "log_sales_per_transaction_contribution"]
    labels = ["점포 수", "점포당 거래", "건당 매출"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(shown))
    for col, label, color in zip(cols, labels, ["#d73027", "#4575b4", "#74add1"]):
        value = shown[col].fillna(0).to_numpy()
        ax.bar(shown["scope"], value, bottom=bottom, label=label, color=color)
        bottom += value
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_ylabel("로그 변화 기여도 (2021→2025)")
    ax.set_title("매출 변화 분해: 매출 = 점포 수 × 점포당 거래 × 건당 매출")
    ax.legend()
    ax.tick_params(axis="x", rotation=15)
    _savefig(JEUNGSAN_FIGURE_DIR / "jeungsan4_sales_decomposition.png")


def _neighbor_annual(annual: pd.DataFrame, neighbors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes = [TARGET_CODE, *neighbors["area_code"].tolist()]
    data = annual.loc[annual["area_code"].isin(codes) & annual["year"].isin([2021, 2022, 2023, 2024, 2025])].copy()
    output = data.groupby(["area_code", "area_name", "year"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"), sales_transactions=("sales_transactions", "sum"), store_count=("store_count", "sum"),
    )
    output["sales_per_store"] = output["sales_amount"] / output["store_count"]
    pivot = output.pivot(index="area_code", columns="year", values="sales_amount")
    output = output.merge((pivot[2025] / pivot[2021] - 1).rename("sales_change_2021_2025").reset_index(), on="area_code", how="left")
    aggregate = data.assign(group=np.where(data["area_code"].eq(TARGET_CODE), "target", "selected_neighbors")).groupby(["group", "year"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"), sales_transactions=("sales_transactions", "sum"), store_count=("store_count", "sum"),
    )
    aggregate["sales_per_store"] = aggregate["sales_amount"] / aggregate["store_count"]
    aggregate["sales_index_2021_100"] = aggregate.groupby("group")["sales_amount"].transform(lambda x: x / x.iloc[0] * 100)
    return output, aggregate


def _plot_neighbors(aggregate: pd.DataFrame) -> None:
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    fig, ax = plt.subplots(figsize=(9, 5))
    for group, values in aggregate.groupby("group"):
        label = "증산역 4번" if group == "target" else "선택된 인접·비교 권역 합계"
        ax.plot(values["year"], values["sales_index_2021_100"], marker="o", label=label)
    ax.axhline(100, color="grey", linewidth=.8, linestyle="--")
    ax.set_xticks([2021, 2022, 2023, 2024, 2025])
    ax.set_ylabel("2021=100 총매출 지수")
    ax.set_title("증산역 4번과 인접·후보 권역의 매출 추이")
    ax.legend()
    ax.grid(alpha=.2)
    _savefig(JEUNGSAN_FIGURE_DIR / "jeungsan4_vs_neighbors.png")


def _plot_controls(annual: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    codes = controls["area_code"].tolist()
    data = annual.loc[annual["area_code"].isin(codes) & annual["year"].isin([2021, 2022, 2023, 2024, 2025])].copy()
    aggregate = data.assign(group=np.where(data["area_code"].eq(TARGET_CODE), "target", "matched_controls")).groupby(["group", "year"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"), sales_transactions=("sales_transactions", "sum"), store_count=("store_count", "sum"),
    )
    aggregate["sales_index_2021_100"] = aggregate.groupby("group")["sales_amount"].transform(lambda x: x / x.iloc[0] * 100)
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    fig, ax = plt.subplots(figsize=(9, 5))
    for group, values in aggregate.groupby("group"):
        ax.plot(values["year"], values["sales_index_2021_100"], marker="o", label="증산역 4번" if group == "target" else "매칭 통제군(합계)")
    ax.axhline(100, color="grey", linewidth=.8, linestyle="--")
    ax.set_xticks([2021, 2022, 2023, 2024, 2025])
    ax.set_ylabel("2021=100 총매출 지수")
    ax.set_title("증산역 4번과 2021년 유사 통제군")
    ax.legend()
    ax.grid(alpha=.2)
    _savefig(JEUNGSAN_FIGURE_DIR / "jeungsan4_vs_matched_controls.png")
    return aggregate


def _external_events() -> pd.DataFrame:
    rows = [
        {"event_category": "재개발·재건축", "event_date": pd.NA, "event": pd.NA, "source": "프로젝트 제공 파일", "verification_status": "not_verifiable", "note": "사업·인허가·공사·이주 일정 데이터가 제공되지 않음"},
        {"event_category": "교통공사·역세권 변화", "event_date": pd.NA, "event": pd.NA, "source": "프로젝트 제공 파일", "verification_status": "not_verifiable", "note": "공사·개통·출입구 변화 일정 데이터가 제공되지 않음"},
        {"event_category": "상권정책·시설 변화", "event_date": pd.NA, "event": pd.NA, "source": "프로젝트 제공 파일", "verification_status": "not_verifiable", "note": "정책·시설 사건의 위치와 시점 자료가 제공되지 않음"},
    ]
    return pd.DataFrame(rows)


def _plot_event_placeholder() -> None:
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.axis("off")
    ax.text(.5, .58, "프로젝트 제공 자료에 외부 사건의 위치·발생시점 데이터가 없습니다.", ha="center", va="center", fontsize=13)
    ax.text(.5, .30, "따라서 재개발·교통·정책 충격의 시간적 선후관계는 본 분석에서 검증하지 않았습니다.", ha="center", va="center", fontsize=10)
    _savefig(JEUNGSAN_FIGURE_DIR / "jeungsan4_event_timeline.png")


def _write_reports(
    reproduction: pd.DataFrame, matrix: pd.DataFrame, city_frequency: pd.DataFrame, sensitivity: pd.DataFrame,
    quarterly: pd.DataFrame, outliers: pd.DataFrame, period_sensitivity: pd.DataFrame, absolute: pd.DataFrame,
    decomposition: pd.DataFrame, boundary: pd.DataFrame, neighbors: pd.DataFrame, neighbor_aggregate: pd.DataFrame,
    controls: pd.DataFrame, balance: pd.DataFrame, control_trend: pd.DataFrame, events: pd.DataFrame, evidence: pd.DataFrame,
) -> None:
    target = reproduction.iloc[0]
    missing_states = matrix.groupby("state").size().reset_index(name="cells")
    endpoint = matrix.loc[matrix["quarter"].isin(["20214", "20254"]), ["quarter", "industry_code", "industry_name", "state"]]
    rank_sens = sensitivity[["scenario", "target_included", "target_rank", "rank_change_vs_baseline", "eligible_areas", "top20"]]
    target_period = period_sensitivity.loc[period_sensitivity["target_included"]]
    common_sales = absolute.loc[(absolute["comparison"].eq("common_observed_2021_2025")) & (absolute["metric"].eq("sales_amount"))].iloc[0]
    common_tx = absolute.loc[(absolute["comparison"].eq("common_observed_2021_2025")) & (absolute["metric"].eq("sales_transactions"))].iloc[0]
    decomp_target = decomposition.loc[decomposition["scope"].eq("target_common_industries")].iloc[0]
    neighbor_target = neighbor_aggregate.loc[neighbor_aggregate["group"].eq("target")].set_index("year")
    neighbor_other = neighbor_aggregate.loc[neighbor_aggregate["group"].eq("selected_neighbors")].set_index("year")
    neighbor_gap = (neighbor_target.at[2025, "sales_index_2021_100"] - neighbor_other.at[2025, "sales_index_2021_100"])
    control_final = control_trend.loc[control_trend["year"].eq(2025)].set_index("group")["sales_index_2021_100"]
    control_gap = control_final.get("target", np.nan) - control_final.get("matched_controls", np.nan)

    write_text(JEUNGSAN_REPORT_DIR / "00_reproduction_check.md", f"""# 00. 기존 1위 결과 재현 점검

## 결과

현재 재현 입력·코드로 `{target['area_name']}`(코드 `{TARGET_CODE}`)은 전체 **{int(target['overall_rank'])}위 / {int(target['eligible_area_count'])}개**입니다. CoreDeclineScore는 **{target['CoreDeclineScore']:.6f}**이며, 장기·중기·최근 점수는 각각 {_md_number(target['long_score'])}, {_md_number(target['medium_score'])}, {_md_number(target['recent_score'])}입니다.

`jeungsan4_reproduction.csv`에는 최종 점수, 모든 기간별 원지표와 표준화 구성점수를 보관했습니다. 이 재현은 원인·인과관계를 입증하지 않으며, 아래 가설 검증의 기준선입니다.
""")
    write_text(JEUNGSAN_REPORT_DIR / "01_missingness_validation.md", f"""# 01. 결측·업종 이탈 가설 검증

대상 상권의 2021Q1–2026Q1, 10개 외식업종 전체 격자를 원시 매출·점포 파일에서 다시 만들었습니다. **행이 없다는 사실을 0으로 바꾸지 않았습니다.** `both_rows_missing`은 양쪽 원시 파일에 행이 없음을 뜻하고, `store_only`/`sales_only`는 한쪽 파일만 존재함을 뜻합니다.

## 대상 격자 상태

{markdown_table(missing_states, ['state', 'cells'])}

## 2021Q4·2025Q4 업종 상태

{markdown_table(endpoint, ['quarter', 'industry_code', 'industry_name', 'state'], 30)}

도시 전체에서도 2021년에 매출이 있던 상권-업종이 2025년에 사라지는 빈도는 업종별로 아래와 같습니다. 따라서 이런 패턴은 대상 하나에만 있는 현상인지 분리해 볼 수 있습니다.

{markdown_table(city_frequency, ['industry_code', 'area_industry_with_2021_sales', 'absent_sales_2025', 'absent_2025_share'], 20)}

## 순위 민감도

{markdown_table(rank_sens, ['scenario', 'target_included', 'target_rank', 'rank_change_vs_baseline', 'eligible_areas', 'top20'], 10)}

- A1은 2021–2025 모든 해에 양 끝점 지표가 가능한 공통 업종만 사용합니다.
- A2는 기존 방식(끝점이 없는 업종은 상대지표 계산에서 제외하고 남은 업종 가중치를 재정규화)입니다.
- A3는 끝점 상대지표가 없는 시작 업종을 서울 평균과 같음(상대지표 0)으로 둡니다. 이는 실제 매출을 보정한 것이 아니라 순위 민감도용 반대 가정입니다.
- A4는 기존 방식에서 장기 매출가중 커버리지가 80% 이상인 상권만 비교합니다.

결측 처리 하나만으로 원 결론을 확정하거나 무효화하지 않으며, A4에서 대상이 제외되면 그 자체가 데이터 신뢰도 제한입니다.
""")
    write_text(JEUNGSAN_REPORT_DIR / "02_boundary_validation.md", f"""# 02. 경계·공간 단위 검증

대상 코드 `{TARGET_CODE}`는 원본 Shapefile에서 **1개 피처**로 확인되었습니다. DBF 속성의 상권명·유형·자치구·행정동은 폴리곤 피처와 동일한 코드에서 읽었습니다. 좌표계는 `.prj`에 기록된 Korea_2000_Korea_Central_Belt이며 단위는 m입니다.

{markdown_table(boundary, list(boundary.columns), 5)}

현재 프로젝트에는 한 시점의 폴리곤만 있으므로 **경계 변경 여부는 검증 불가**입니다. 점포 좌표 파일도 없어 점포가 경계 밖으로 이동했는지는 판단하지 않았습니다. 지도는 `outputs/jeungsan4/figures/jeungsan4_boundary_map.png`에 저장했습니다.
""")
    write_text(JEUNGSAN_REPORT_DIR / "03_base_year_validation.md", f"""# 03. 2021년 기준연도·시간축 검증

분기 시계열에는 총매출·거래·점포 수·점포당 매출·점포당 거래·건당 매출·개폐업·순진입률과 전년동기 대비, 4분기 이동평균을 보관했습니다. 2021년 각 분기는 2022–2025년 같은 분기의 median/MAD robust Z로 이상치를 점검했습니다.

## 2021 분기 계절기준 이상치

{markdown_table(outliers.loc[outliers['outlier_abs_z_ge_3_5']], ['quarter', 'metric', 'value_2021', 'peer_2022_2025_median', 'robust_z'], 30)}

## 시작연도별 단일기간 상대쇠퇴 순위

{markdown_table(target_period, ['scenario', 'period', 'target_rank', 'eligible_areas', 'target_score', 'top20'], 10)}

`recent_8q_endpoint`는 2024Q1–2025Q4의 두 완결연도 끝점 비교이며, 더 긴 사전기간이 없으므로 독립적인 장기 사전추세 placebo는 만들 수 없습니다. 분기표 `jeungsan4_quarterly_series.csv`와 그림을 함께 해석해야 합니다.
""")
    write_text(JEUNGSAN_REPORT_DIR / "04_absolute_decline_validation.md", f"""# 04. 절대 변화 검증

상대성과와 별개로, 2021→2025 공통 관측 업종 기준 총매출은 **{_md_number(common_sales['start_2021'], 0)} → {_md_number(common_sales['end_2025'], 0)} ({_md_number(common_sales['percent_change'] * 100, 1)}%)**, 거래건수는 **{_md_number(common_tx['start_2021'], 0)} → {_md_number(common_tx['end_2025'], 0)} ({_md_number(common_tx['percent_change'] * 100, 1)}%)**입니다.

`all_observed_endpoint` 행은 각 연도에 관측된 전체 업종을 더한 것으로 업종 구성이 달라져 직접 비교의 신뢰도가 낮습니다. `common_observed_2021_2025`와 한식·분식·치킨 단독 행이 같은 업종 비교입니다. CPI 또는 물가 자료는 제공되지 않아 실질매출은 계산하지 않았습니다.
""")
    write_text(JEUNGSAN_REPORT_DIR / "05_sales_decomposition.md", f"""# 05. 매출 변화 분해

공통 관측 업종에 대해 `매출 = 평균 점포 수 × 점포당 거래건수 × 건당 매출`의 로그 항등식으로 분해했습니다. 대상의 로그 매출 변화는 {_md_number(decomp_target['log_sales_change'])}이며, 점포 수·점포당 거래·건당 매출 기여는 각각 {_md_number(decomp_target['log_store_count_contribution'])}, {_md_number(decomp_target['log_transactions_per_store_contribution'])}, {_md_number(decomp_target['log_sales_per_transaction_contribution'])}입니다. 세 기여 합과 전체 로그변화의 차이는 반올림·집계의 잔차로 별도 열에 보관했습니다.

이는 가격·수요를 완전하게 식별하는 인과 분해가 아니라, 관측된 매출 감소가 점포 수·점포당 거래·건당 매출 중 어디와 함께 나타나는지를 기술하는 회계적 분해입니다.
""")
    write_text(JEUNGSAN_REPORT_DIR / "06_spatial_displacement_validation.md", f"""# 06. 인접 상권·공간 재배치 가설

이웃은 (a) 폴리곤 경계 거리 10m 이하, (b) 중심점 거리 1km 이하, (c) 같은 행정동, (d) 증산·수색·북가좌·DMC 명칭 후보 중 하나로 선정했습니다. 이 기준은 표에 모두 기록해 사후적으로 바꾸지 않도록 했습니다.

선택된 이웃은 {len(neighbors)}개입니다. 2025년 대상과 이웃 합계의 2021=100 매출지수 격차는 **{_md_number(neighbor_gap, 1)}포인트**입니다. 이는 공간 재배치의 단서일 수 있으나, 소비자 이동이나 점포 이전의 추적자료가 없으므로 **재배치의 인과 증거는 아닙니다**.

{markdown_table(neighbors, ['area_code', 'area_name', 'area_type', 'centroid_distance_m', 'boundary_distance_m', 'selection_reason'], 25)}
""")
    write_text(JEUNGSAN_REPORT_DIR / "07_external_shock_validation.md", f"""# 07. 외부 충격 가설

프로젝트 제공 자료에는 재개발·교통공사·정책·시설 변화의 위치와 발생일이 없습니다. 따라서 매출 변화와 외부 사건의 선후관계를 검증할 수 없습니다. 사건을 추정하거나 외부 검색으로 메우지 않았습니다.

{markdown_table(events, ['event_category', 'verification_status', 'note'], 10)}
""")
    write_text(JEUNGSAN_REPORT_DIR / "08_matched_control_validation.md", f"""# 08. 유사 비교군 검증

통제군은 2021년 같은 상권유형 안에서 총점포수·총매출·한식 매출비중·거래건수·점포당 매출을 표준화한 유클리드 거리로 가장 가까운 {MATCHED_CONTROL_COUNT}개를 택했습니다. 이는 성향점수매칭(PSM)이 아니라, 제공 자료만으로 만든 투명한 nearest-neighbour 비교입니다.

## 균형

{markdown_table(balance, ['feature', 'target_2021', 'controls_mean_2021', 'difference', 'standardized_mean_difference'], 10)}

2025년 2021=100 매출지수에서 대상−통제군 격차는 **{_md_number(control_gap, 1)}포인트**입니다. 단, 통제군에도 관측되지 않은 입지·임대료·소비자 특성 차이가 있어 인과효과로 해석할 수 없습니다.
""")

    verdict = evidence.loc[evidence["condition"].eq("final_classification"), "result"].iloc[0]
    write_text(JEUNGSAN_REPORT_DIR / "09_final_verdict.md", f"""# 09. 최종 판정

## 판정: {verdict}

{markdown_table(evidence, ['condition', 'result', 'evidence', 'interpretation'], 20)}

분류 규칙은 사전에 명시했습니다. `구조적 쇠퇴`는 결측·경계·기준연도·절대감소의 첫 4 조건을 모두 통과하고, 8개 조건 중 6개 이상이 지지하며, 데이터 신뢰도가 중간 이상일 때만 사용합니다. 그 외에는 확인 가능한 증거의 강도에 맞춰 `상대적 약화`, `공간 재배치 가능성`, `외부충격 검증 불가`, `데이터/경계 이상`, `판정 유보` 중 하나를 사용합니다.
""")
    write_text(JEUNGSAN_REPORT_DIR / "README.md", "# 증산역 4번 검증 묶음\n\n"
        "이 문서는 기존 1위 결론을 정당화하기 위한 문서가 아니라, 원시 관측·경계·기준연도·비교집단의 다른 합리적 정의에서 결론이 유지되는지 검증한 결과입니다.\n\n"
        "## 결론\n\n" + f"최종 분류는 **{verdict}**입니다. 상세 판정표는 `09_final_verdict.md`에 있습니다.\n\n"
        "## 읽는 순서\n\n1. `00_reproduction_check.md` — 기존 결과 재현\n2. `01_missingness_validation.md` — 업종 행 누락과 4개 민감도\n3. `02_boundary_validation.md` — 폴리곤·경계 단위\n4. `03_base_year_validation.md` — 2021 기준연도·분기추세\n5. `04_absolute_decline_validation.md` — 절대 변화\n6. `05_sales_decomposition.md` — 매출 항등식 분해\n7. `06_spatial_displacement_validation.md` — 이웃·재배치\n8. `07_external_shock_validation.md` — 외부충격 자료 한계\n9. `08_matched_control_validation.md` — 유사 비교군\n\n"
        "모든 수치는 `outputs/jeungsan4/tables/`와 `src/validation/run_validation.py`에서 다시 계산할 수 있습니다. 결측을 0으로 대체하지 않았으며, 제공되지 않은 보조자료는 `검증 불가`로 남겼습니다.\n")
    write_text(JEUNGSAN_OUTPUT_DIR / "fieldwork_checklist.md", """# 증산역 4번 현장확인 체크리스트

- [ ] 상권 경계 안·밖의 실제 영업 점포를 업종별로 전수 확인한다.
- [ ] 2021년과 현재의 한식·중식·치킨·분식·호프 점포 변화를 주소 단위로 추적한다.
- [ ] 빈 점포, 업종 전환, 이전(인접 상권 포함)을 사진·좌표·확인일과 함께 기록한다.
- [ ] 증산역 3·4번 출구, 증산종합시장·골목시장 간 보행 흐름을 평일/주말·점심/저녁에 관찰한다.
- [ ] 임대료·권리금·공실 및 재개발·공사·교통 변화의 정확한 발생일과 위치를 행정자료로 확인한다.
- [ ] 매출 하락이 고객수 감소인지 객단가 변화인지 점포 인터뷰와 POS 자료로 교차검증한다.
- [ ] 프로젝트의 현행 폴리곤이 실제 영업구역과 맞는지 현장에서 표시한다.
""")


def _evidence_matrix(
    sensitivity: pd.DataFrame, boundary: pd.DataFrame, period_sensitivity: pd.DataFrame, absolute: pd.DataFrame,
    decomposition: pd.DataFrame, neighbor_aggregate: pd.DataFrame, events: pd.DataFrame, control_trend: pd.DataFrame,
) -> pd.DataFrame:
    a = sensitivity.set_index("scenario")
    a1 = a.loc["A1_common_industries_all_2021_2025"]
    a3 = a.loc["A3_missing_industry_equals_seoul_average"]
    a4 = a.loc["A4_sales_coverage_at_least_80pct"]
    missing_result = "supports" if bool(a1["target_included"]) and bool(a3["target_included"]) and min(a1["target_rank"], a3["target_rank"]) <= 20 else "mixed"
    if not bool(a4["target_included"]):
        missing_result = "mixed_low_coverage"
    # The DBF X/Y fields are a supplied representative coordinate, whereas the
    # calculated value is an areal geometry centroid.  Their difference alone
    # is not evidence of a boundary error.  Code uniqueness and the nearly
    # identical reported/calculated polygon area are the reproducible checks.
    area_tolerance = max(1.0, float(boundary.iloc[0]["dbf_reported_area_m2"]) * 0.01)
    boundary_result = "supports" if (
        boundary.iloc[0]["polygon_feature_count_for_code"] == 1
        and abs(float(boundary.iloc[0]["area_difference_m2"])) <= area_tolerance
    ) else "data_boundary_anomaly"
    start_rows = period_sensitivity.loc[period_sensitivity["scenario"].str.startswith("start_")]
    base_result = "supports" if (start_rows["target_rank"] <= 20).sum() >= 3 else "mixed"
    sales = absolute.loc[(absolute["comparison"].eq("common_observed_2021_2025")) & (absolute["metric"].eq("sales_amount")), "percent_change"].iloc[0]
    tx = absolute.loc[(absolute["comparison"].eq("common_observed_2021_2025")) & (absolute["metric"].eq("sales_transactions")), "percent_change"].iloc[0]
    absolute_result = "supports" if sales < 0 and tx < 0 else "does_not_support"
    target_decomp = decomposition.loc[decomposition["scope"].eq("target_common_industries")].iloc[0]
    decomp_result = "supports" if target_decomp["log_transactions_per_store_contribution"] < 0 else "mixed"
    piv = neighbor_aggregate.pivot(index="year", columns="group", values="sales_index_2021_100")
    reallocation_result = "possible" if piv.at[2025, "selected_neighbors"] > piv.at[2025, "target"] else "not_supported"
    external_result = "not_verifiable" if events["verification_status"].eq("not_verifiable").all() else "mixed"
    control = control_trend.loc[control_trend["year"].eq(2025)].set_index("group")["sales_index_2021_100"]
    control_result = "supports" if control["target"] < control["matched_controls"] else "does_not_support"
    rows = [
        {"condition": "1_missingness_robustness", "result": missing_result, "evidence": f"A1 rank={_md_number(a1['target_rank'],0)}, A3 rank={_md_number(a3['target_rank'],0)}, A4 included={a4['target_included']}", "interpretation": "Missingness alternatives do not provide an unqualified robustness pass when the 80% coverage screen excludes the target."},
        {"condition": "2_boundary_consistency", "result": boundary_result, "evidence": f"features={boundary.iloc[0]['polygon_feature_count_for_code']}, area difference={boundary.iloc[0]['area_difference_m2']:.3f}m², representative-coordinate gap={boundary.iloc[0]['centroid_difference_m']:.1f}m", "interpretation": "One polygon feature and matching reported area support the current spatial unit. The DBF representative coordinate need not equal a geometric centroid; historical boundary change is unverified."},
        {"condition": "3_base_year_robustness", "result": base_result, "evidence": f"top20 in {(start_rows['target_rank'] <= 20).sum()} of {len(start_rows)} start-year endpoint ranks", "interpretation": "Changing the start year tests whether 2021 alone creates the ranking."},
        {"condition": "4_absolute_common_industry_decline", "result": absolute_result, "evidence": f"common-industry sales {sales:.1%}, transactions {tx:.1%}", "interpretation": "Absolute change is evaluated only on common observed industries."},
        {"condition": "5_operating_demand_decomposition", "result": decomp_result, "evidence": f"log tx/store contribution={target_decomp['log_transactions_per_store_contribution']:.3f}", "interpretation": "A negative point indicates declining transactions per operating store, not only fewer stores."},
        {"condition": "6_spatial_reallocation", "result": reallocation_result, "evidence": f"2025 index target={piv.at[2025, 'target']:.1f}, neighbor={piv.at[2025, 'selected_neighbors']:.1f}", "interpretation": "A nearby relative increase is descriptive evidence only; origin-destination or store-move data are absent."},
        {"condition": "7_external_shock", "result": external_result, "evidence": "No dated, geocoded project external-event data", "interpretation": "This hypothesis is not filled with conjecture."},
        {"condition": "8_matched_control_comparison", "result": control_result, "evidence": f"2025 index target={control['target']:.1f}, controls={control['matched_controls']:.1f}", "interpretation": "Nearest-neighbour controls are descriptive and not causal PSM."},
    ]
    supported = sum(row["result"] == "supports" for row in rows)
    first_four = all(rows[i]["result"] == "supports" for i in range(4))
    data_reliability_medium = missing_result == "supports" and boundary_result == "supports"
    if first_four and supported >= 6 and data_reliability_medium:
        verdict = "구조적 쇠퇴"
    elif boundary_result == "data_boundary_anomaly":
        verdict = "데이터/경계 이상"
    elif reallocation_result == "possible" and absolute_result != "supports":
        verdict = "공간 재배치 가능성"
    elif absolute_result == "supports" and control_result == "supports":
        verdict = "상대적 약화"
    elif external_result == "not_verifiable" and supported < 4:
        verdict = "판정 유보(외부충격 검증 불가 포함)"
    else:
        verdict = "판정 유보"
    rows.append({"condition": "final_classification", "result": verdict, "evidence": f"supports={supported}/8; first_four_all={first_four}; data_reliability_medium={data_reliability_medium}", "interpretation": "Predeclared decision rule applied without assuming missing external facts."})
    return pd.DataFrame(rows)


def run() -> dict[str, object]:
    """Generate all validation tables, figures, and reports."""
    ranking = _existing(TABLE_DIR / "commercial_area_decline_ranking.csv")
    annual = _existing(Path("data/processed/food_commercial_area_annual_panel.csv"))
    panel = _existing(Path("data/interim/food_commercial_area_quarter_panel.csv"))
    area_metrics = _existing(Path("data/interim/area_relative_metrics.csv"))
    reference = read_area_reference()
    if not reference.empty:
        reference["area_code"] = _code(reference["area_code"])
    target = ranking.loc[ranking["area_code"].eq(TARGET_CODE)].copy()
    if len(target) != 1:
        raise ValueError(f"Expected exactly one target ranking row, found {len(target)}")
    target["eligible_area_count"] = len(ranking)
    reproduction_columns = ["area_code", "area_name", "district", "area_type", "eligible_area_count", "overall_rank", "CoreDeclineScore", "long_score", "medium_score", "recent_score",
                            "sales_weight_coverage", "store_weight_coverage", "valid_industry_count", "quarter_observation_rate", "sensitivity_mean_rank", "sensitivity_rank_std", "top20_appearance_rate",
                            *[c for c in ranking.columns if c.endswith("_rel") or c.endswith("_rel_score")]]
    reproduction = target[[c for c in reproduction_columns if c in target]].copy()
    save_csv(reproduction, JEUNGSAN_TABLE_DIR / "jeungsan4_reproduction.csv")

    inputs = discover_inputs()
    raw_sales = load_area_data(inputs, "sales_area")
    raw_stores = load_area_data(inputs, "stores_area")
    for data in (raw_sales, raw_stores):
        data["area_code"] = _code(data["area_code"])
        data["industry_code"] = _code(data["industry_code"])
        data["quarter"] = _code(data["quarter"])
    matrix, city_frequency = _presence_matrix(raw_sales, raw_stores)
    save_csv(matrix, JEUNGSAN_TABLE_DIR / "jeungsan4_industry_presence_matrix.csv")
    save_csv(city_frequency, JEUNGSAN_TABLE_DIR / "jeungsan4_citywide_endpoint_missingness_frequency.csv")

    # A2 is loaded from the core pipeline to guarantee literal baseline identity.
    baseline_summary = _scenario_target_summary("A2_baseline_drop_and_reweight", ranking, float(target.iloc[0]["overall_rank"]), "Existing endpoint-complete industries; remaining start-year weights renormalized")
    a1_ranking = _score_scenario(annual, panel, _scenario_area_metrics(annual, "common_all_periods"))
    a3_ranking = _score_scenario(annual, panel, _scenario_area_metrics(annual, "seoul_average"))
    baseline_eligibility = build_eligibility(annual, panel, area_metrics)
    a4_ranking = _score_scenario(annual, panel, area_metrics, coverage_80=True)
    sensitivity = pd.DataFrame([
        _scenario_target_summary("A1_common_industries_all_2021_2025", a1_ranking, float(target.iloc[0]["overall_rank"]), "Use only industries valid in every full year 2021–2025"),
        baseline_summary,
        _scenario_target_summary("A3_missing_industry_equals_seoul_average", a3_ranking, float(target.iloc[0]["overall_rank"]), "Start-year industries without endpoint metric receive relative value zero"),
        _scenario_target_summary("A4_sales_coverage_at_least_80pct", a4_ranking, float(target.iloc[0]["overall_rank"]), "Baseline calculation, comparison set restricted to long sales coverage >=80%"),
    ])
    sensitivity["target_baseline_sales_coverage"] = target.iloc[0]["sales_weight_coverage"]
    sensitivity["baseline_eligible_count"] = int(baseline_eligibility["base_eligible"].sum())
    save_csv(sensitivity, JEUNGSAN_TABLE_DIR / "jeungsan4_missingness_sensitivity.csv")

    quarterly = _quarterly_series(panel)
    outliers = _seasonal_outliers(quarterly)
    save_csv(quarterly, JEUNGSAN_TABLE_DIR / "jeungsan4_quarterly_series.csv")
    save_csv(outliers, JEUNGSAN_TABLE_DIR / "jeungsan4_2021_seasonal_outliers.csv")
    period_rows: list[dict[str, object]] = []
    for start in [2021, 2022, 2023, 2024]:
        ranked, _ = _single_period_rank(annual, panel, start, 2025, f"start_{start}_to_2025")
        row = ranked.loc[ranked["area_code"].eq(TARGET_CODE)]
        period_rows.append({"scenario": f"start_{start}_to_2025", "period": f"{start}–2025", "eligible_areas": len(ranked),
                            "target_included": not row.empty, "target_rank": row.iloc[0]["period_rank"] if not row.empty else np.nan,
                            "target_score": row.iloc[0]["period_decline_score"] if not row.empty else np.nan,
                            "top20": bool(not row.empty and row.iloc[0]["period_rank"] <= 20)})
    recent_row = period_rows[-1].copy()
    recent_row["scenario"] = "recent_8q_endpoint"
    recent_row["period"] = "2024Q1–2025Q4 (8 complete quarters)"
    period_rows.append(recent_row)
    period_sensitivity = pd.DataFrame(period_rows)
    save_csv(period_sensitivity, JEUNGSAN_TABLE_DIR / "jeungsan4_period_sensitivity.csv")

    absolute = _absolute_changes(annual)
    save_csv(absolute, JEUNGSAN_TABLE_DIR / "jeungsan4_absolute_change.csv")
    controls, balance = _match_controls(annual, reference)
    save_csv(controls, JEUNGSAN_TABLE_DIR / "jeungsan4_matched_controls.csv")
    save_csv(balance, JEUNGSAN_TABLE_DIR / "jeungsan4_matching_balance.csv")
    decomposition = _sales_decomposition(annual, controls)
    save_csv(decomposition, JEUNGSAN_TABLE_DIR / "jeungsan4_sales_decomposition.csv")

    polygons = _load_polygons()
    if TARGET_CODE not in polygons:
        raise ValueError("Target code is absent from supplied polygon shapefile.")
    neighbors = _neighbor_table(polygons, annual)
    boundary = _boundary_outputs(polygons, neighbors)
    save_csv(neighbors, JEUNGSAN_TABLE_DIR / "jeungsan4_neighbor_comparison.csv")
    save_csv(boundary, JEUNGSAN_TABLE_DIR / "jeungsan4_boundary_check.csv")
    neighbor_annual, neighbor_aggregate = _neighbor_annual(annual, neighbors)
    save_csv(neighbor_annual, JEUNGSAN_TABLE_DIR / "jeungsan4_neighbor_annual_metrics.csv")
    save_csv(neighbor_aggregate, JEUNGSAN_TABLE_DIR / "jeungsan_area_aggregate.csv")

    events = _external_events()
    save_csv(events, JEUNGSAN_TABLE_DIR / "jeungsan4_external_events.csv")
    control_trend = _plot_controls(annual, controls)
    save_csv(control_trend, JEUNGSAN_TABLE_DIR / "jeungsan4_matched_control_trend.csv")
    _plot_boundary(polygons, neighbors)
    _plot_quarterly(quarterly)
    _plot_decomposition(decomposition)
    _plot_neighbors(neighbor_aggregate)
    _plot_event_placeholder()

    evidence = _evidence_matrix(sensitivity, boundary, period_sensitivity, absolute, decomposition, neighbor_aggregate, events, control_trend)
    save_csv(evidence, JEUNGSAN_TABLE_DIR / "jeungsan4_evidence_matrix.csv")
    _write_reports(reproduction, matrix, city_frequency, sensitivity, quarterly, outliers, period_sensitivity, absolute, decomposition, boundary, neighbors, neighbor_aggregate, controls, balance, control_trend, events, evidence)
    return {
        "target": _target_name(ranking), "baseline_rank": int(target.iloc[0]["overall_rank"]),
        "presence_cells": len(matrix), "neighbors": len(neighbors),
        "final_classification": evidence.iloc[-1]["result"],
    }


if __name__ == "__main__":
    print(run())
