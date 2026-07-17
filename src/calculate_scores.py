"""Eligibility, robust scoring, and optional early-warning calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    METRIC_WEIGHTS, MIN_OBSERVATION_RATE, MIN_VALID_INDUSTRIES, MIN_WEIGHT_COVERAGE,
    PERIOD_WEIGHTS, PERIODS, WINSOR_LIMITS,
)


METRICS = list(METRIC_WEIGHTS)


def build_eligibility(annual: pd.DataFrame, quarter_panel: pd.DataFrame, area_metrics: pd.DataFrame) -> pd.DataFrame:
    """Apply the size, coverage, observation, and industry-count gates."""
    food_store_year = annual.loc[annual["complete_year"]].groupby(["area_code", "year"], as_index=False)["store_count"].sum()
    start_store = food_store_year.loc[food_store_year["year"].eq(2021), ["area_code", "store_count"]].rename(columns={"store_count": "start_food_stores"})
    average_store = food_store_year.groupby("area_code", as_index=False)["store_count"].mean().rename(columns={"store_count": "analysis_avg_food_stores"})
    area_observation = quarter_panel.loc[quarter_panel["quarter"].astype(str).str[:4].astype(int).between(2021, 2025)].groupby("area_code", as_index=False)["quarter"].nunique()
    area_observation["quarter_observation_rate"] = area_observation["quarter"] / 20
    area_observation = area_observation.drop(columns="quarter")
    long = area_metrics.loc[area_metrics["period"].eq("long"), ["area_code", "area_name", "area_type", "valid_industry_count", "sales_weight_coverage", "store_weight_coverage"]]
    eligible = long.merge(start_store, on="area_code", how="left").merge(average_store, on="area_code", how="left").merge(area_observation, on="area_code", how="left")
    metric_complete = area_metrics.pivot(index="area_code", columns="period", values=METRICS).notna().all(axis=1).rename("all_metric_periods_available").reset_index()
    eligible = eligible.merge(metric_complete, on="area_code", how="left")
    criteria = {
        "start_store_20": eligible["start_food_stores"].ge(20),
        "average_store_20": eligible["analysis_avg_food_stores"].ge(20),
        "industry_3": eligible["valid_industry_count"].ge(MIN_VALID_INDUSTRIES),
        "observation_80": eligible["quarter_observation_rate"].ge(MIN_OBSERVATION_RATE),
        "sales_coverage_70": eligible["sales_weight_coverage"].ge(MIN_WEIGHT_COVERAGE),
        "store_coverage_70": eligible["store_weight_coverage"].ge(MIN_WEIGHT_COVERAGE),
        "metrics_available": eligible["all_metric_periods_available"].fillna(False),
    }
    for name, series in criteria.items():
        eligible[name] = series.fillna(False)
    eligible["base_eligible"] = pd.concat(criteria, axis=1).all(axis=1)
    eligible["exclusion_reason"] = eligible.apply(
        lambda row: "; ".join(name for name in criteria if not bool(row[name])) or "포함", axis=1
    )
    return eligible


def _winsorize(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 5:
        return series.astype(float)
    low, high = valid.quantile(WINSOR_LIMITS[0]), valid.quantile(WINSOR_LIMITS[1])
    return series.clip(low, high)


def standardize_decline(series: pd.Series, method: str = "robust") -> pd.Series:
    """Return a standardized decline component: larger means relatively worse."""
    value = -_winsorize(series.astype(float))
    if method == "standard":
        scale = value.std(ddof=0)
        return (value - value.mean()) / scale if scale and np.isfinite(scale) else pd.Series(0.0, index=value.index)
    median = value.median()
    mad = (value - median).abs().median()
    if mad and np.isfinite(mad):
        return 0.67448975 * (value - median) / mad
    scale = value.std(ddof=0)
    return (value - value.mean()) / scale if scale and np.isfinite(scale) else pd.Series(0.0, index=value.index)


def calculate_ranking(
    area_metrics: pd.DataFrame,
    eligibility: pd.DataFrame,
    metric_weights: dict[str, float] = METRIC_WEIGHTS,
    min_start_stores: int = 20,
    z_method: str = "robust",
) -> pd.DataFrame:
    """Calculate CoreDeclineScore for one sensitivity scenario.

    Scores are standardized within that scenario's eligible comparison set.
    """
    candidate = eligibility.loc[eligibility["base_eligible"] & eligibility["start_food_stores"].ge(min_start_stores)].copy()
    candidate = candidate[["area_code", "area_name", "area_type", "start_food_stores", "analysis_avg_food_stores", "quarter_observation_rate", "valid_industry_count", "sales_weight_coverage", "store_weight_coverage"]]
    active_metrics = [metric for metric, weight in metric_weights.items() if weight > 0]
    output = candidate.copy()
    for period in PERIODS:
        current = candidate[["area_code"]].merge(area_metrics.loc[area_metrics["period"].eq(period), ["area_code", *METRICS]], on="area_code", how="left")
        period_score = pd.Series(0.0, index=current.index)
        complete = current[active_metrics].notna().all(axis=1)
        for metric in active_metrics:
            standard = standardize_decline(current.loc[complete, metric], z_method)
            output.loc[output["area_code"].isin(current.loc[complete, "area_code"]), f"{period}_{metric}_score"] = standard.to_numpy()
            period_score.loc[complete] += metric_weights[metric] * standard
        current[f"{period}_score"] = period_score.where(complete, np.nan)
        current = current[["area_code", f"{period}_score", *METRICS]].rename(
            columns={metric: f"{period}_{metric}" for metric in METRICS}
        )
        output = output.merge(current, on="area_code", how="left")
    output["CoreDeclineScore"] = sum(PERIOD_WEIGHTS[period] * output[f"{period}_score"] for period in PERIODS)
    output["overall_rank"] = output["CoreDeclineScore"].rank(method="min", ascending=False).astype("Int64")
    output["area_type_rank"] = output.groupby("area_type")["CoreDeclineScore"].rank(method="min", ascending=False).astype("Int64")
    return output.sort_values(["overall_rank", "area_code"]).reset_index(drop=True)


def calculate_early_warning(
    quarter_panel: pd.DataFrame,
    quarterly_benchmark: pd.DataFrame,
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the non-annualized 2025Q1→2026Q1 auxiliary score when available."""
    if not {"20251", "20261"}.issubset(set(quarter_panel["quarter"].astype(str))):
        return ranking.assign(EarlyWarningScore=np.nan, EarlyWarningAdjustedScore=np.nan, early_warning_available=False)
    merged = quarter_panel.merge(quarterly_benchmark, on=["quarter", "industry_code"], how="left")
    start = merged.loc[merged["quarter"].eq("20251")].set_index(["area_code", "industry_code"])
    end = merged.loc[merged["quarter"].eq("20261")].set_index(["area_code", "industry_code"])
    both = start.join(end, how="inner", lsuffix="_start", rsuffix="_end")
    industry = both.reset_index()[["area_code", "industry_code"]].copy()
    pairs = {
        "sales_rel": ("sales_amount", "seoul_sales_amount"),
        "transactions_rel": ("sales_transactions", "seoul_sales_transactions"),
        "sales_per_store_rel": ("sales_per_store", "seoul_sales_per_store"),
        "store_rel": ("store_count", "seoul_store_count"),
    }
    for name, (area_col, seoul_col) in pairs.items():
        valid = (both[f"{area_col}_start"] > 0) & (both[f"{area_col}_end"] > 0) & (both[f"{seoul_col}_start"] > 0) & (both[f"{seoul_col}_end"] > 0)
        value = pd.Series(np.nan, index=both.index)
        value.loc[valid] = np.log(both.loc[valid, f"{area_col}_end"] / both.loc[valid, f"{area_col}_start"]) - np.log(both.loc[valid, f"{seoul_col}_end"] / both.loc[valid, f"{seoul_col}_start"])
        industry[name] = value.to_numpy()
    industry["net_entry_rel"] = (both["net_entry_rate_end"] - both["seoul_net_entry_rate_end"]).to_numpy()
    base = start.reset_index()[["area_code", "industry_code", "sales_amount", "store_count"]].rename(columns={"sales_amount": "start_sales", "store_count": "start_stores"})
    industry = industry.merge(base, on=["area_code", "industry_code"], how="left")
    aggregated: list[dict[str, object]] = []
    for code, group in industry.groupby("area_code"):
        row: dict[str, object] = {"area_code": code}
        for metric in METRICS:
            measure = "start_sales" if metric in {"sales_rel", "transactions_rel", "sales_per_store_rel"} else "start_stores"
            valid = group[metric].notna() & group[measure].gt(0)
            denominator = group.loc[valid, measure].sum()
            row[metric] = (group.loc[valid, metric] * group.loc[valid, measure]).sum() / denominator if denominator > 0 else np.nan
        aggregated.append(row)
    early = pd.DataFrame(aggregated)
    eligible_codes = ranking["area_code"]
    early = eligible_codes.to_frame().merge(early, on="area_code", how="left")
    components = pd.Series(0.0, index=early.index)
    complete = early[METRICS].notna().all(axis=1)
    for metric, weight in METRIC_WEIGHTS.items():
        components.loc[complete] += weight * standardize_decline(early.loc[complete, metric], "robust")
    early["EarlyWarningScore"] = components.where(complete, np.nan)
    result = ranking.merge(early[["area_code", "EarlyWarningScore"]], on="area_code", how="left")
    result["EarlyWarningAdjustedScore"] = 0.90 * result["CoreDeclineScore"] + 0.10 * result["EarlyWarningScore"]
    result["early_warning_available"] = result["EarlyWarningScore"].notna()
    result["early_warning_rank"] = result["EarlyWarningAdjustedScore"].rank(method="min", ascending=False).astype("Int64")
    return result
