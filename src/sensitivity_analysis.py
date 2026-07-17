"""Sensitivity scenarios and rank-stability calculations."""
from __future__ import annotations

import pandas as pd

from .calculate_scores import calculate_ranking
from .config import SIZE_SENSITIVITY, WEIGHT_SCENARIOS


def run_sensitivity(area_metrics: pd.DataFrame, eligibility: pd.DataFrame, exit_area_metrics: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the requested weight, scale, and standardization scenarios."""
    outputs: list[pd.DataFrame] = []
    for weight_name, weights in WEIGHT_SCENARIOS.items():
        for minimum in SIZE_SENSITIVITY:
            for z_method in ("robust", "standard"):
                ranking = calculate_ranking(area_metrics, eligibility, weights, minimum, z_method)
                scenario = f"{weight_name}|점포{minimum}+|{z_method}Z"
                outputs.append(ranking[["area_code", "overall_rank", "CoreDeclineScore"]].assign(
                    scenario=scenario, weight_scenario=weight_name, min_start_stores=minimum, z_method=z_method
                ))
    if exit_area_metrics is not None and not exit_area_metrics.empty:
        baseline = calculate_ranking(area_metrics, eligibility, WEIGHT_SCENARIOS["기본"], 20, "robust")
        exit_wide = exit_area_metrics.pivot(index="area_code", columns="period", values="confirmed_exit_store_weight").fillna(0.0)
        adjusted = baseline.copy()
        for period in ("long", "medium", "recent"):
            weights = adjusted["area_code"].map(exit_wide[period] if period in exit_wide else pd.Series(dtype=float)).fillna(0.0)
            # Confirmed exits are a zero-inflated, non-negative signal. A
            # general Z-score would penalize zero-exit areas and magnify rare
            # exits. Map store-weighted exit share to a bounded 0–3 signal.
            exit_component = (3.0 * weights).clip(lower=0.0, upper=3.0)
            adjusted[f"{period}_exit_adjusted_score"] = 0.80 * adjusted[f"{period}_score"] + 0.20 * exit_component
        adjusted["CoreDeclineScore"] = 0.50 * adjusted["long_exit_adjusted_score"] + 0.25 * adjusted["medium_exit_adjusted_score"] + 0.25 * adjusted["recent_exit_adjusted_score"]
        adjusted["overall_rank"] = adjusted["CoreDeclineScore"].rank(method="min", ascending=False).astype("Int64")
        outputs.append(adjusted[["area_code", "overall_rank", "CoreDeclineScore"]].assign(
            scenario="확정업종소멸반영|점포20+|robustZ", weight_scenario="확정업종소멸반영", min_start_stores=20, z_method="robust"
        ))
    all_rankings = pd.concat(outputs, ignore_index=True)
    stability = all_rankings.groupby("area_code", as_index=False).agg(
        sensitivity_scenarios=("scenario", "nunique"),
        sensitivity_mean_rank=("overall_rank", "mean"),
        sensitivity_rank_std=("overall_rank", "std"),
        top20_appearances=("overall_rank", lambda x: int((x <= 20).sum())),
    )
    stability["top20_appearance_rate"] = stability["top20_appearances"] / stability["sensitivity_scenarios"]
    stability["strong_decline_candidate"] = (
        stability["top20_appearance_rate"].ge(0.70)
        & stability["sensitivity_mean_rank"].le(stability["sensitivity_mean_rank"].quantile(0.01))
        & stability["sensitivity_rank_std"].le(stability["sensitivity_mean_rank"].clip(lower=3) * 0.50)
    )
    return all_rankings, stability


def metric_correlations(area_metrics: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations between the five component metrics by period."""
    rows: list[pd.DataFrame] = []
    codes = set(eligibility.loc[eligibility["base_eligible"], "area_code"])
    for period, group in area_metrics.loc[area_metrics["area_code"].isin(codes)].groupby("period"):
        corr = group[["sales_rel", "transactions_rel", "sales_per_store_rel", "store_rel", "net_entry_rel"]].corr(method="spearman")
        corr.index.name = "metric_a"
        tidy = corr.reset_index().melt(id_vars="metric_a", var_name="metric_b", value_name="spearman_rho")
        tidy["period"] = period
        tidy["high_correlation"] = tidy["spearman_rho"].abs().ge(0.85)
        rows.append(tidy)
    return pd.concat(rows, ignore_index=True)
