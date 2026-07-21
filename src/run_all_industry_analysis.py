"""Run a separate all-observed-industry relative-decline ranking.

This never overwrites the existing food-service ranking.  It uses every
industry for which both sales and store observations are available.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_panel import build_annual_panel, build_quarter_panel
from .calculate_benchmarks import annual_benchmarks
from .calculate_relative_metrics import build_relative_area_metrics
from .calculate_scores import build_eligibility, calculate_ranking
from .config import OUTPUT_DIR, REPORT_DIR, TABLE_DIR
from .export_results import markdown_table, save_csv, write_text
from .load_data import discover_inputs, load_area_data, read_area_reference
from .sensitivity_analysis import metric_correlations, run_sensitivity
from .structural_exit import build_confirmed_exit_metrics
from .visualize import create_figures


OUT_DIR = OUTPUT_DIR / "all_industry"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"
REPORT_DIR_ALL = REPORT_DIR / "all_industry"
for directory in (OUT_DIR, TABLES_DIR, FIGURES_DIR, REPORT_DIR_ALL):
    directory.mkdir(parents=True, exist_ok=True)


def _report(ranking: pd.DataFrame, eligibility: pd.DataFrame, industry_count: int, comparison: pd.DataFrame) -> str:
    top = ranking.head(20).copy()
    top["강건후보"] = top["strong_decline_candidate"].map({True: "예", False: "아니오"})
    leeum = comparison.loc[comparison["area_code"].eq("3110091")]
    return f"""# 서울 전체 관측 업종 상대 악화 순위

## 목적과 범위

이 결과는 기존 외식업 순위를 대체하지 않는다. 매출과 점포가 함께 관측되는 **{industry_count}개 업종**을 대상으로, 각 업종에서 서울 동일 업종 대비 상대성과를 계산한 별도 시나리오다. 따라서 이 순위는 원자료에 포함된 관측 업종의 상대적 악화 신호이며, 서울의 모든 경제활동 업종 또는 원인·인과관계의 순위가 아니다.

## 계산 원칙

- 기간: 2021Q1–2025Q4, 연간 비교는 4개 분기 완결 업종만 사용
- 지표: 매출액·거래건수·점포당 매출·연말 점포 수의 서울 동일 업종 대비 로그변화, 순진입률 격차
- 업종 통합: 매출 관련 지표는 시작연도 매출 비중, 점포 관련 지표는 시작연도 점포 비중으로 가중평균
- 점수: 기간별 robust Z-score(1%·99% 윈저라이징), 지표 가중치 25%·20%·15%·20%·20%, 장기·중기·최근 비중 50%·25%·25%
- 적격성: 시작·평균 점포 20개 이상, 유효 업종 3개 이상, 관측률 80% 이상, 매출·점포 가중치 커버리지 70% 이상, 세 기간 지표 산출 가능

## 결과

- 적격 상권 수: **{len(ranking)}개**
- 적격성에서 제외된 상권 수: **{int((~eligibility['base_eligible']).sum())}개**

### 상위 20개

{markdown_table(top, ['overall_rank', 'area_name', 'district', 'area_type', 'CoreDeclineScore', 'long_score', 'medium_score', 'recent_score', 'top20_appearance_rate', '강건후보'], 20)}

### 기존 외식업 순위와의 비교: 리움미술관

{markdown_table(leeum, ['area_name', 'food_overall_rank', 'all_industry_rank', 'food_CoreDeclineScore', 'all_industry_CoreDeclineScore'], 5)}

## 해석 경계

서로 다른 업종은 소비 목적·가격·점포 규모가 다르다. 전체업종 점수가 높다는 것은 관측 업종들을 가중 통합했을 때 서울 동일 업종 대비 상대적 악화가 크다는 뜻이며, 외식업 결과나 특정 업종의 절대 매출 감소와 동일하지 않다.
"""


def run() -> dict[str, object]:
    inputs = discover_inputs()
    sales = load_area_data(inputs, "sales_area", industry_codes=None)
    stores = load_area_data(inputs, "stores_area", industry_codes=None)
    quarter_panel, duplicate_audit = build_quarter_panel(sales, stores)
    annual = build_annual_panel(quarter_panel)
    benchmarks = annual_benchmarks(annual)
    industry_metrics, area_metrics = build_relative_area_metrics(annual, benchmarks)
    eligibility = build_eligibility(annual, quarter_panel, area_metrics)
    ranking = calculate_ranking(area_metrics, eligibility)
    exit_industry, exit_area = build_confirmed_exit_metrics(stores)
    sensitivity, stability = run_sensitivity(area_metrics, eligibility, exit_area)
    correlations = metric_correlations(area_metrics, eligibility)

    reference = read_area_reference()
    if not reference.empty:
        reference["area_code"] = reference["area_code"].astype(str).str.strip()
        ranking = ranking.merge(reference, on="area_code", how="left", suffixes=("", "_reference"))
        ranking["area_name"] = ranking["area_name"].fillna(ranking.get("area_name_reference"))
        ranking["area_type"] = ranking["area_type"].fillna(ranking.get("area_type_reference"))
    else:
        ranking["district"] = pd.NA
        ranking["administrative_dong"] = pd.NA
    ranking = ranking.merge(stability, on="area_code", how="left").sort_values("overall_rank").reset_index(drop=True)

    food_path = TABLE_DIR / "commercial_area_decline_ranking.csv"
    food = pd.read_csv(food_path, dtype={"area_code": str})[["area_code", "overall_rank", "CoreDeclineScore"]].rename(columns={"overall_rank": "food_overall_rank", "CoreDeclineScore": "food_CoreDeclineScore"})
    comparison = food.merge(ranking[["area_code", "area_name", "overall_rank", "CoreDeclineScore"]], on="area_code", how="outer").rename(columns={"overall_rank": "all_industry_rank", "CoreDeclineScore": "all_industry_CoreDeclineScore"})

    save_csv(ranking, TABLES_DIR / "all_industry_relative_decline_ranking.csv")
    save_csv(ranking.head(20), TABLES_DIR / "top20_all_industry_relative_decline.csv")
    save_csv(eligibility, TABLES_DIR / "all_industry_eligibility_and_exclusions.csv")
    save_csv(sensitivity, TABLES_DIR / "all_industry_ranking_sensitivity.csv")
    save_csv(correlations, TABLES_DIR / "all_industry_metric_correlations.csv")
    save_csv(comparison, TABLES_DIR / "food_vs_all_industry_rank_comparison.csv")
    save_csv(benchmarks, OUT_DIR / "seoul_all_industry_benchmarks.csv")
    save_csv(area_metrics, OUT_DIR / "all_industry_relative_area_metrics.csv")
    save_csv(industry_metrics, OUT_DIR / "all_industry_relative_industry_metrics.csv")
    save_csv(duplicate_audit, OUT_DIR / "all_industry_duplicate_audit.csv")
    save_csv(exit_industry, OUT_DIR / "all_industry_confirmed_structural_exits_by_industry.csv")
    create_figures(ranking, stability, correlations, FIGURES_DIR)
    write_text(REPORT_DIR_ALL / "all_industry_methodology_and_results.md", _report(ranking, eligibility, annual["industry_code"].nunique(), comparison))
    summary = {
        "observed_industries": int(annual["industry_code"].nunique()),
        "joined_areas": int(quarter_panel["area_code"].nunique()),
        "eligible_areas": int(len(ranking)),
        "leeum": comparison.loc[comparison["area_code"].eq("3110091")].to_dict(orient="records"),
        "top10": ranking.head(10)[["overall_rank", "area_name", "CoreDeclineScore"]].to_dict(orient="records"),
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
