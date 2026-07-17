"""Audit measurement bases and interpretation boundaries for the Leeum analysis.

This is deliberately separate from the hypothesis test.  It verifies that a
claim never silently moves between a relative ranking metric, a complete-year
absolute total, an index, or an incomplete endpoint comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import OUTPUT_DIR, REPORT_DIR
from ..export_results import markdown_table, save_csv, write_text


TARGET_CODE = "3110091"
LEEUM_DIR = OUTPUT_DIR / "leeum"
REPORT_DIR = REPORT_DIR / "leeum"


def _read(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "area_code" in data:
        data["area_code"] = data["area_code"].astype(str).str.replace(".0", "", regex=False)
    return data


def _pct(start: float, end: float) -> float:
    return end / start - 1 if pd.notna(start) and pd.notna(end) and start != 0 else np.nan


def _industry_baselines(annual: pd.DataFrame) -> pd.DataFrame:
    target = annual.loc[annual["area_code"].eq(TARGET_CODE)].copy()
    values = target.pivot(index=["industry_code", "industry_name"], columns="year", values="sales_amount")
    complete = target.pivot(index=["industry_code", "industry_name"], columns="year", values="complete_year")
    values.columns = [f"sales_{year}" for year in values.columns]
    complete.columns = [f"complete_{year}" for year in complete.columns]
    result = values.join(complete).reset_index()
    for start, end, name in [(2021, 2025, "change_2021_2025"), (2022, 2025, "change_2022_2025")]:
        result[name] = result.apply(lambda row: _pct(row.get(f"sales_{start}"), row.get(f"sales_{end}")), axis=1)
        result[f"{name}_usable"] = result.get(f"complete_{start}", False).fillna(False) & result.get(f"complete_{end}", False).fillna(False)
    result["interpretation"] = np.where(
        result["change_2021_2025_usable"],
        "2021→2025 is a valid industry-specific endpoint comparison, but it answers a different question from the 2022-base charts.",
        "2021 endpoint is incomplete; do not calculate or narrate a 2021→2025 change for this industry.",
    )
    return result


def _reconciliation(annual: pd.DataFrame, annual_total: pd.DataFrame, quarterly: pd.DataFrame, decomposition: pd.DataFrame) -> pd.DataFrame:
    target = annual.loc[annual["area_code"].eq(TARGET_CODE) & annual["year"].isin([2022, 2023, 2024, 2025])]
    rows: list[dict[str, object]] = []
    for year, group in target.groupby("year"):
        published = annual_total.loc[annual_total["year"].eq(year)].iloc[0]
        summed_sales = group["sales_amount"].sum()
        summed_tx = group["sales_transactions"].sum()
        rows.extend([
            {"check": f"annual sales identity {year}", "expected": summed_sales, "observed": published["sales_amount"], "difference": published["sales_amount"] - summed_sales, "status": "pass" if np.isclose(published["sales_amount"], summed_sales) else "fail"},
            {"check": f"annual transactions identity {year}", "expected": summed_tx, "observed": published["sales_transactions"], "difference": published["sales_transactions"] - summed_tx, "status": "pass" if np.isclose(published["sales_transactions"], summed_tx) else "fail"},
            {"check": f"annual sales/store identity {year}", "expected": published["sales_amount"] / published["operating_stores"], "observed": published["sales_per_store"], "difference": published["sales_per_store"] - published["sales_amount"] / published["operating_stores"], "status": "pass" if np.isclose(published["sales_per_store"], published["sales_amount"] / published["operating_stores"]) else "fail"},
        ])
    target_decomp = decomposition.loc[decomposition["scope"].eq("leeum")].iloc[0]
    rows.append({"check": "sales decomposition identity 2022→2025", "expected": 0.0, "observed": target_decomp["decomposition_residual"], "difference": target_decomp["decomposition_residual"], "status": "pass" if np.isclose(target_decomp["decomposition_residual"], 0.0) else "fail"})
    incomplete = quarterly.loc[~quarterly["total_series_comparable"], "quarter"].astype(str).tolist()
    rows.append({"check": "incomplete total-series quarters excluded", "expected": "20211, 20212, 20213", "observed": ", ".join(incomplete), "difference": "", "status": "pass" if incomplete == ["20211", "20212", "20213"] else "review"})
    return pd.DataFrame(rows)


def run() -> dict[str, object]:
    annual = _read(Path("data/processed/food_commercial_area_annual_panel.csv"))
    quarterly = _read(LEEUM_DIR / "quarterly_metrics.csv")
    annual_total = _read(LEEUM_DIR / "annual_metrics_2022_2025.csv")
    decomposition = _read(LEEUM_DIR / "sales_decomposition.csv")
    reproduction = _read(LEEUM_DIR / "leeum_reproduction.csv")

    registry = pd.DataFrame([
        {"artifact": "commercial_area_decline_ranking.csv", "metric_type": "relative composite ranking", "time_or_base": "long 2021→2025; medium 2023→2025; recent 2024→2025", "aggregation_or_denominator": "industry-level log change minus Seoul same-industry log change; weighted then robust-standardized", "allowed_interpretation": "relative underperformance signal within the eligible sample", "not_allowed": "absolute sales decline or causal diagnosis"},
        {"artifact": "annual_metrics_2022_2025.csv", "metric_type": "absolute complete-year totals", "time_or_base": "2022→2025", "aggregation_or_denominator": "sum of four observed target industries; sales/transactions annual sums, stores quarterly mean", "allowed_interpretation": "absolute change after the 2022 observed peak", "not_allowed": "2021 total comparison or causal attribution"},
        {"artifact": "quarterly_metrics.csv", "metric_type": "absolute quarterly series", "time_or_base": "2021Q1→2026Q1", "aggregation_or_denominator": "target-industry aggregate; total_series_comparable marks complete coverage", "allowed_interpretation": "only quarters with all four target industries observed are comparable", "not_allowed": "using 2021Q1–Q3 aggregate levels or YoY values as full-total evidence"},
        {"artifact": "industry_deep_dive.csv / industry_trends.png", "metric_type": "industry index", "time_or_base": "2022=100; chart uses 2022→2025", "aggregation_or_denominator": "each industry's own 2022 sales", "allowed_interpretation": "within-industry change from 2022", "not_allowed": "comparison to a 2021-based percentage without naming the different baseline"},
        {"artifact": "industry_contribution.csv / industry_contribution.png", "metric_type": "absolute loss contribution", "time_or_base": "2022→2025", "aggregation_or_denominator": "industry sales_2025 minus sales_2022; share among observed losses", "allowed_interpretation": "which industry contributes more won loss amount", "not_allowed": "proof of industry reorganization or relative percentage decline"},
        {"artifact": "sales_decomposition.csv", "metric_type": "accounting decomposition", "time_or_base": "2022→2025", "aggregation_or_denominator": "sales = mean stores × transactions per store × sales per transaction; log changes", "allowed_interpretation": "mechanical components associated with the sales change", "not_allowed": "customer origin, price cause, or causal mechanism"},
        {"artifact": "neighbor_aggregate.csv", "metric_type": "independent-neighbour equal-weight index", "time_or_base": "2022=100", "aggregation_or_denominator": "mean of each retained independent area's own 2022=100 index; the overlapping tourism-zone candidate is excluded", "allowed_interpretation": "descriptive divergence from the retained independent-neighbour average", "not_allowed": "proven spatial displacement or a common-shock counterfactual"},
        {"artifact": "matched_control_trend.csv", "metric_type": "imbalanced comparison aggregate index", "time_or_base": "2022=100", "aggregation_or_denominator": "sum over 7 nearest-feature areas", "allowed_interpretation": "supplementary descriptive comparison only", "not_allowed": "causal treatment effect or counterfactual"},
    ])
    baselines = _industry_baselines(annual)
    quarterly_audit = quarterly[["quarter", "observed_industries", "expected_industries", "total_series_comparable", "comparability_note"]].copy()
    reconciliation = _reconciliation(annual, annual_total, quarterly, decomposition)
    claims = pd.DataFrame([
        {"claim_or_chart": "rank 5 / CoreDeclineScore 2.231", "status": "valid with scope", "preferred_wording": "Eligible-sample relative-underperformance rank, not an absolute-decline rank."},
        {"claim_or_chart": "2021 total vs 2025 total", "status": "invalid", "preferred_wording": "Do not compare: 2021Q1–Q3 omit one target industry."},
        {"claim_or_chart": "2021→2025 industry percentages", "status": "valid only per complete industry", "preferred_wording": "Label the start/end years explicitly; never compare directly with a 2022=100 chart."},
        {"claim_or_chart": "2022→2025 total sales, transactions, stores", "status": "valid descriptive total", "preferred_wording": "Complete-year absolute trend after the observed 2022 peak; no causal claim."},
        {"claim_or_chart": "sales decomposition", "status": "valid accounting identity", "preferred_wording": "Transaction intensity is the negative component; transaction value rises and partly offsets it."},
        {"claim_or_chart": "industry reorganization", "status": "not supported", "preferred_wording": "All four observed industries decline; store/business continuity data are absent."},
        {"claim_or_chart": "neighbor/control indices", "status": "descriptive only", "preferred_wording": "Indices are 2022=100 aggregates; control balance is poor (max |SMD| 2.69)."},
        {"claim_or_chart": "2022Q4 event break", "status": "not identified", "preferred_wording": "2022Q4 is an event marker; the descriptive mean-break candidate is 2024Q3 and is not causal."},
    ])

    save_csv(registry, LEEUM_DIR / "measurement_registry.csv")
    save_csv(baselines, LEEUM_DIR / "industry_baseline_comparison.csv")
    save_csv(quarterly_audit, LEEUM_DIR / "quarterly_comparability_audit.csv")
    save_csv(reconciliation, LEEUM_DIR / "numerical_reconciliation_audit.csv")
    save_csv(claims, LEEUM_DIR / "claim_review.csv")

    rank = reproduction.iloc[0]
    report = f"""# 10. 측정 기준·해석 경계 재검토

## 결론

기존 순위 **{int(rank['overall_rank'])}위**와 2022→2025 완결연도 절대 추이는 수치적으로 재현된다. 다만 이 둘은 같은 지표가 아니다. 순위는 서울 동일 업종 대비 **상대성과 복합점수**이고, 연도 총량은 리움의 **절대 집계값**이다. 이 문서는 두 결과를 서로의 증거로 바꾸어 해석하지 않는다.

## 즉시 정정한 항목

- 2021Q1–Q3은 관측 업종이 3개뿐이어서 총량·점포수·전년동기 비교·변화점 탐색에서 제외했다.
- 2022→2025 매출분해에서 건당 매출은 하락이 아니라 상승(+0.132 로그 기여)이며, 점포당 거래 하락(-0.721)이 매출 감소의 주된 회계 항목이다.
- 2022=100 업종지수와 2021→2025 업종 변동률은 기준연도가 다르다. 같은 문장에서 직접 대비하지 않는다.
- 모든 관측 업종의 2022→2025 매출이 감소했으므로, 현 자료는 업종 재편을 주된 설명으로 지지하지 않는다.

## 지표별 사용 규칙

{markdown_table(registry, ['artifact', 'metric_type', 'time_or_base', 'allowed_interpretation', 'not_allowed'], 20)}

## 업종 기준연도 대조

{markdown_table(baselines, ['industry_name', 'sales_2021', 'sales_2022', 'sales_2025', 'change_2021_2025', 'change_2022_2025', 'change_2021_2025_usable'], 10)}

`change_2021_2025`와 `change_2022_2025`는 서로 다른 질문의 답이다. 전자는 코로나 시기를 포함한 2021년 대비 위치, 후자는 2022년 관측 고점 이후의 변화다. 분석의 총량·분해·인접 비교 그림은 후자만 사용한다.

## 수치 대사 결과

{markdown_table(reconciliation, ['check', 'expected', 'observed', 'difference', 'status'], 20)}

## 표현 허용·금지 목록

{markdown_table(claims, ['claim_or_chart', 'status', 'preferred_wording'], 20)}

## 재현 경로

1. `python -m src.leeum.run_validation`
2. `python -m src.leeum.run_measurement_audit`

두 명령은 원자료를 0으로 보정하지 않는다. 2021 불완전 집계는 별도 표기로 보존하되, 비교 가능한 총량·전년동기·변화점 계산에서는 제외한다.
"""
    write_text(REPORT_DIR / "10_measurement_interpretation_audit.md", report)
    return {"rank": int(rank["overall_rank"]), "reconciliation_passes": int(reconciliation["status"].eq("pass").sum()), "claims_reviewed": len(claims)}


if __name__ == "__main__":
    print(run())
