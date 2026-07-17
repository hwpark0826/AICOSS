"""Conservative confirmed-exit signals derived from the store source only."""
from __future__ import annotations

import pandas as pd

from .config import ANALYSIS_YEARS, PERIODS
from .load_data import quarter_to_year


def build_confirmed_exit_metrics(stores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return industry exit flags and store-weighted area exit intensity.

    A missing sales row is never treated as a closure. An industry is a
    confirmed exit only when the store dataset itself shows four start-year
    observations with positive stores, a zero-store final Q4, and a recorded
    closure during the period.
    """
    data = stores.copy()
    data["year"] = quarter_to_year(data["quarter"])
    data["quarter_no"] = pd.to_numeric(data["quarter"].astype(str).str[-1], errors="coerce")
    data = data.loc[data["year"].isin(ANALYSIS_YEARS)].copy()
    key = ["year", "area_code", "industry_code"]
    annual = data.groupby(key, as_index=False).agg(
        start_avg_store_count=("store_count", "mean"),
        observed_quarters=("quarter", "nunique"),
        annual_close_count=("close_count", "sum"),
    )
    q4 = data.loc[data["quarter_no"].eq(4), ["year", "area_code", "industry_code", "store_count"]].rename(columns={"store_count": "year_end_store_count"})
    annual = annual.merge(q4, on=key, how="left")
    rows: list[pd.DataFrame] = []
    for period, (start_year, end_year) in PERIODS.items():
        start = annual.loc[annual["year"].eq(start_year), ["area_code", "industry_code", "start_avg_store_count", "observed_quarters"]].rename(columns={"start_avg_store_count": "period_start_store_weight", "observed_quarters": "start_observed_quarters"})
        end = annual.loc[annual["year"].eq(end_year), ["area_code", "industry_code", "year_end_store_count"]]
        closures = annual.loc[annual["year"].between(start_year, end_year)].groupby(["area_code", "industry_code"], as_index=False)["annual_close_count"].sum()
        joined = start.merge(end, on=["area_code", "industry_code"], how="left").merge(closures, on=["area_code", "industry_code"], how="left")
        joined["period"] = period
        joined["confirmed_structural_exit"] = (
            joined["start_observed_quarters"].eq(4)
            & joined["period_start_store_weight"].gt(0)
            & joined["year_end_store_count"].eq(0)
            & joined["annual_close_count"].gt(0)
        )
        rows.append(joined)
    industry = pd.concat(rows, ignore_index=True)
    area_rows: list[dict[str, object]] = []
    for (area_code, period), group in industry.groupby(["area_code", "period"]):
        denominator = group.loc[group["period_start_store_weight"].gt(0), "period_start_store_weight"].sum()
        numerator = group.loc[group["confirmed_structural_exit"], "period_start_store_weight"].sum()
        area_rows.append({"area_code": area_code, "period": period, "confirmed_exit_industry_count": int(group["confirmed_structural_exit"].sum()), "confirmed_exit_store_weight": numerator / denominator if denominator > 0 else 0.0})
    return industry, pd.DataFrame(area_rows)
