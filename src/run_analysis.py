"""Command-line entry point for the Seoul food commercial-area analysis."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

from .build_panel import build_annual_panel, build_early_warning_panel, build_quarter_panel
from .calculate_benchmarks import annual_benchmarks, quarterly_benchmarks
from .calculate_relative_metrics import build_relative_area_metrics
from .calculate_scores import build_eligibility, calculate_early_warning, calculate_ranking
from .config import INTERIM_DIR, PROCESSED_DIR, REPORT_DIR, TABLE_DIR, FIGURE_DIR
from .export_results import (
    format_inventory, format_methodology, format_quality, markdown_table, save_csv, write_text,
)
from .load_data import InputFile, discover_inputs, load_area_data, read_area_reference
from .report_addenda import confirmed_exit_methodology_note
from .sensitivity_analysis import metric_correlations, run_sensitivity
from .structural_exit import build_confirmed_exit_metrics
from .validate_data import validate_sources
from .visualize import create_figures


def _count_rows(path: Path, encoding: str) -> int:
    with path.open("r", encoding=encoding, newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def _inventory(inputs: list[InputFile], sales: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for item in inputs:
        subset = sales.loc[sales["source_file"].eq(item.path.name)] if item.role == "sales_area" else stores.loc[stores["source_file"].eq(item.path.name)] if item.role == "stores_area" else pd.DataFrame()
        required_missing = "-"
        if item.role == "sales_area":
            required = {"기준_년분기_코드", "상권_코드", "서비스_업종_코드", "당월_매출_금액", "당월_매출_건수"}
            required_missing = ", ".join(sorted(required - set(item.columns))) or "없음"
        if item.role == "stores_area":
            required = {"기준_년분기_코드", "상권_코드", "서비스_업종_코드", "개업_점포_수", "폐업_점포_수"}
            has_store_count = "점포_수" in item.columns or "전체_점포_수" in item.columns
            required_missing = ", ".join(sorted(required - set(item.columns)) + ([] if has_store_count else ["점포_수/전체_점포_수"])) or "없음"
        records.append({
            "file": item.path.name, "role": item.role, "encoding": item.encoding,
            "raw_rows": _count_rows(item.path, item.encoding),
            "period": f"{subset['quarter'].min()}~{subset['quarter'].max()}" if not subset.empty else "-",
            "food_rows": len(subset), "food_areas": subset["area_code"].nunique() if not subset.empty else 0,
            "food_industries": subset["industry_code"].nunique() if not subset.empty else 0,
            "duplicate_keys": int(subset.duplicated(["quarter", "area_code", "industry_code"], keep=False).sum()) if not subset.empty else 0,
            "missing_required": required_missing,
        })
    return pd.DataFrame(records)


def _report_ranking(ranking: pd.DataFrame, stability: pd.DataFrame, eligibility: pd.DataFrame) -> str:
    combined = ranking.copy() if "strong_decline_candidate" in ranking.columns else ranking.merge(stability, on="area_code", how="left")
    top = combined.sort_values("overall_rank").head(20).copy()
    top["강건후보"] = top["strong_decline_candidate"].map({True: "예", False: "아니오"})
    excluded = eligibility.loc[~eligibility["base_eligible"]]
    candidate = combined.loc[combined["strong_decline_candidate"].fillna(False)].sort_values("overall_rank").head(5)
    return "# 서울 외식업 상권 쇠퇴 신호 순위\n\n## 해석 원칙\n\n이 결과는 원인이나 인과관계를 의미하지 않습니다. 본 데이터와 정의된 복합지표 기준에서 서울 동일 세부업종 대비 상대적 악화 신호가 강하게 나타난 상권의 순위입니다.\n\n## 분석 결과\n\n- 주 분석 기간: 2021Q1~2025Q4\n- 주 순위 적격 상권 수: " + str(len(ranking)) + "\n- 적격성 기준으로 제외된 상권 수: " + str(len(excluded)) + "\n- 2026Q1 자료 기반 조기경보 보조점수: " + ("계산됨" if ranking["early_warning_available"].any() else "계산 불가") + "\n\n## 쇠퇴 신호 상위 20개 상권\n\n" + markdown_table(top, ["overall_rank", "area_name", "district", "area_type", "CoreDeclineScore", "long_score", "medium_score", "recent_score", "top20_appearance_rate", "강건후보"], 20) + "\n\n## 현장조사 우선 후보\n\n" + markdown_table(candidate, ["overall_rank", "area_name", "district", "CoreDeclineScore", "sensitivity_mean_rank", "sensitivity_rank_std", "top20_appearance_rate"], 5) + "\n\n## 유의사항\n\n- 공식 서울시 단위 업종 자료가 없어 상권 단위 자료의 업종별 합계로 기준선을 만들었습니다. 따라서 공식 집계 범위와 다를 수 있습니다.\n- 0 또는 결측인 시작·종료값의 로그성장률은 보정하지 않고 해당 지표를 결측 처리했습니다.\n- 상권코드 명칭·유형 변경, 결합 누락, 급변값은 품질 이슈 표를 함께 확인해야 합니다.\n"


def run() -> dict[str, object]:
    """Execute every stage and write all non-workbook deliverables."""
    inputs = discover_inputs()
    sales = load_area_data(inputs, "sales_area")
    stores = load_area_data(inputs, "stores_area")
    inventory = _inventory(inputs, sales, stores)
    write_text(REPORT_DIR / "data_inventory.md", format_inventory(inventory))

    quality_issues, quality_summary = validate_sources(sales, stores)
    quarter_panel, duplicate_audit = build_quarter_panel(sales, stores)
    if not duplicate_audit.empty:
        quality_issues = pd.concat([quality_issues, duplicate_audit.assign(issue_type="duplicate_resolution", detail=lambda x: x["duplicate_resolution"])], ignore_index=True, sort=False)
    save_csv(quality_issues, TABLE_DIR / "data_quality_issues.csv")
    write_text(REPORT_DIR / "data_quality_report.md", format_quality(quality_summary, quality_issues))

    annual = build_annual_panel(quarter_panel)
    save_csv(quarter_panel, INTERIM_DIR / "food_commercial_area_quarter_panel.csv")
    save_csv(annual, PROCESSED_DIR / "food_commercial_area_annual_panel.csv")
    parquet_path = PROCESSED_DIR / "food_commercial_area_annual_panel.parquet"
    try:
        annual.to_parquet(parquet_path, index=False)
        parquet_note = "연간 패널은 CSV와 Parquet로 모두 저장했습니다."
    except Exception as exc:  # Engine availability is environment-dependent; never create a mislabeled substitute.
        parquet_note = f"Parquet 엔진을 사용할 수 없어 CSV만 저장했습니다. 사유: {type(exc).__name__}."
    benchmarks = annual_benchmarks(annual)
    industry_metrics, area_metrics = build_relative_area_metrics(annual, benchmarks)
    eligibility = build_eligibility(annual, quarter_panel, area_metrics)
    ranking = calculate_ranking(area_metrics, eligibility)
    early = build_early_warning_panel(quarter_panel)
    ranking = calculate_early_warning(early, quarterly_benchmarks(early), ranking)
    confirmed_exit_industry, confirmed_exit_area = build_confirmed_exit_metrics(stores)
    sensitivity, stability = run_sensitivity(area_metrics, eligibility, confirmed_exit_area)
    correlations = metric_correlations(area_metrics, eligibility)

    reference = read_area_reference()
    if not reference.empty:
        reference["area_code"] = reference["area_code"].astype(str).str.strip()
        ranking = ranking.merge(reference, on="area_code", how="left", suffixes=("", "_reference"))
        if "area_name_reference" in ranking:
            ranking["area_name"] = ranking["area_name"].fillna(ranking["area_name_reference"])
        if "area_type_reference" in ranking:
            ranking["area_type"] = ranking["area_type"].fillna(ranking["area_type_reference"])
    else:
        ranking["district"] = pd.NA
        ranking["administrative_dong"] = pd.NA
    ranking = ranking.merge(stability, on="area_code", how="left")
    ranking = ranking.sort_values("overall_rank").reset_index(drop=True)
    top20 = ranking.head(20)

    save_csv(benchmarks, INTERIM_DIR / "seoul_food_industry_benchmarks.csv")
    save_csv(industry_metrics, INTERIM_DIR / "industry_relative_metrics.csv")
    save_csv(area_metrics, INTERIM_DIR / "area_relative_metrics.csv")
    save_csv(eligibility, TABLE_DIR / "eligibility_and_exclusions.csv")
    save_csv(ranking, TABLE_DIR / "commercial_area_decline_ranking.csv")
    save_csv(top20, TABLE_DIR / "top20_declining_areas.csv")
    save_csv(sensitivity, TABLE_DIR / "ranking_sensitivity.csv")
    save_csv(correlations, TABLE_DIR / "metric_correlations.csv")
    save_csv(confirmed_exit_industry, TABLE_DIR / "confirmed_structural_exits_by_industry.csv")
    save_csv(confirmed_exit_area, TABLE_DIR / "confirmed_structural_exit_weights.csv")
    create_figures(ranking, stability, correlations, FIGURE_DIR)
    write_text(REPORT_DIR / "methodology.md", format_methodology(parquet_note) + confirmed_exit_methodology_note())
    write_text(REPORT_DIR / "decline_ranking_report.md", _report_ranking(ranking, stability, eligibility))
    summary = {
        "input_files": len(inputs), "sales_rows_food": len(sales), "store_rows_food": len(stores),
        "quarter_panel_rows": len(quarter_panel), "annual_panel_rows": len(annual),
        "eligible_areas": len(ranking), "top10": ranking.head(10)[["overall_rank", "area_name", "CoreDeclineScore"]].to_dict(orient="records"),
        "parquet_written": parquet_path.exists(), "early_warning": bool(ranking["early_warning_available"].any()),
    }
    (REPORT_DIR / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    try:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    except Exception as error:
        print(f"ANALYSIS_FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise
