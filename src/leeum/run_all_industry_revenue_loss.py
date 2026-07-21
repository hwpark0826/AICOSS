"""Calculate Leeum's observed all-industry revenue decline without imputation."""
from __future__ import annotations

import pandas as pd

from ..build_panel import remove_duplicate_keys
from ..config import ANALYSIS_QUARTERS, OUTPUT_DIR, REPORT_DIR
from ..export_results import markdown_table, save_csv, write_text
from ..load_data import discover_inputs, load_area_data


TARGET_CODE = "3110091"
REFERENCE_YEAR = 2022
CURRENT_YEAR = 2025
OUT_DIR = OUTPUT_DIR / "leeum" / "all_industry_revenue_loss"
REPORT_PATH = REPORT_DIR / "leeum" / "18_all_industry_revenue_loss.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _annual_summary(data):
    return data.groupby("year", as_index=False).agg(
        observed_sales=("sales_amount", "sum"),
        observed_transactions=("sales_transactions", "sum"),
        observed_industries=("industry_code", "nunique"),
        observed_industry_quarter_rows=("industry_code", "size"),
    )


def run() -> dict[str, object]:
    sales = load_area_data(discover_inputs(), "sales_area", industry_codes=None)
    sales, duplicate_audit = remove_duplicate_keys(sales, ["sales_amount", "sales_transactions"])
    if not duplicate_audit.empty:
        save_csv(duplicate_audit, OUT_DIR / "sales_duplicate_resolution.csv")
    leeum = sales.loc[(sales["area_code"].eq(TARGET_CODE)) & sales["quarter"].isin(ANALYSIS_QUARTERS)].copy()
    leeum["year"] = leeum["quarter"].str[:4].astype(int)
    annual_observed = _annual_summary(leeum)

    # A like-for-like check: retain only industries with all four quarters in both comparison years.
    coverage = leeum.groupby(["year", "industry_code"])["quarter"].nunique().unstack(fill_value=0)
    balanced_codes = coverage.columns[
        coverage.loc[REFERENCE_YEAR].eq(4) & coverage.loc[CURRENT_YEAR].eq(4)
    ].tolist()
    annual_balanced = _annual_summary(leeum.loc[leeum["industry_code"].isin(balanced_codes)]).rename(columns={
        "observed_sales": "balanced_sales",
        "observed_transactions": "balanced_transactions",
        "observed_industries": "balanced_industries",
        "observed_industry_quarter_rows": "balanced_industry_quarter_rows",
    })
    annual = annual_observed.merge(annual_balanced, on="year", how="left")
    reference = annual.loc[annual["year"].eq(REFERENCE_YEAR)].iloc[0]
    current = annual.loc[annual["year"].eq(CURRENT_YEAR)].iloc[0]
    loss = []
    for basis, sales_column, transaction_column, industry_column in (
        ("원자료 관측 합계", "observed_sales", "observed_transactions", "observed_industries"),
        ("동일 업종·완전분기 비교", "balanced_sales", "balanced_transactions", "balanced_industries"),
    ):
        loss.append({
            "basis": basis,
            "reference_year": REFERENCE_YEAR,
            "current_year": CURRENT_YEAR,
            "reference_sales": int(reference[sales_column]),
            "current_sales": int(current[sales_column]),
            "sales_decrease_amount": int(reference[sales_column] - current[sales_column]),
            "sales_change_percent": (current[sales_column] / reference[sales_column] - 1) * 100,
            "reference_transactions": int(reference[transaction_column]),
            "current_transactions": int(current[transaction_column]),
            "transaction_change_percent": (current[transaction_column] / reference[transaction_column] - 1) * 100,
            "industries_in_current_year": int(current[industry_column]),
        })
    loss_summary = pd.DataFrame(loss)
    balanced_industries = leeum.loc[leeum["industry_code"].isin(balanced_codes), ["industry_code", "industry_name"]].drop_duplicates().sort_values("industry_code")
    save_csv(annual, OUT_DIR / "leeum_all_industry_annual_sales.csv")
    save_csv(loss_summary, OUT_DIR / "leeum_all_industry_revenue_loss_summary.csv")
    save_csv(balanced_industries, OUT_DIR / "balanced_industries_2022_2025.csv")

    observed = loss_summary.iloc[0]
    comparable = loss_summary.iloc[1]
    write_text(REPORT_PATH, f"""# 리움미술관 상권: 전체 업종 매출 감소액

## 결론

2022년을 비교 기준으로 두면, 리움미술관 상권에서 원자료로 관측되는 전체 업종 매출은 2025년에 **{observed['sales_decrease_amount']:,.0f}원 감소**했다. 이는 **{observed['sales_change_percent']:.1f}%** 감소이며, 거래건수는 **{observed['transaction_change_percent']:.1f}%** 줄었다.

이 값은 ‘머뭇이 없어서 발생한 피해액’이나 인과적 손실이 아니다. **2022년 매출 수준을 기준으로 한 현재 관측 매출 격차**다. 서비스의 회복효과는 향후 대조군을 둔 실증에서 별도로 추정해야 한다.

## 연도별 원자료 관측 합계

{markdown_table(annual, ['year', 'observed_sales', 'observed_transactions', 'observed_industries', 'observed_industry_quarter_rows', 'balanced_sales', 'balanced_transactions', 'balanced_industries'], 10)}

## 두 비교 기준

{markdown_table(loss_summary, ['basis', 'reference_sales', 'current_sales', 'sales_decrease_amount', 'sales_change_percent', 'reference_transactions', 'current_transactions', 'transaction_change_percent', 'industries_in_current_year'], 10)}

### 1. 원자료 관측 합계: 발표용 기본 수치

- 2022년: **{observed['reference_sales']:,.0f}원**
- 2025년: **{observed['current_sales']:,.0f}원**
- 차이: **{observed['sales_decrease_amount']:,.0f}원 ({observed['sales_change_percent']:.1f}%)**

### 2. 동일 업종·완전분기 비교: 보수적 검증 수치

업종 행의 유무가 합계 차이를 만들지 않도록, 2022년과 2025년에 모두 4개 분기가 관측된 업종만 남겼다. 해당 업종의 매출 차이는 **{comparable['sales_decrease_amount']:,.0f}원 ({comparable['sales_change_percent']:.1f}%)**이다. 포함 업종은 `balanced_industries_2022_2025.csv`에 저장했다.

## 해석 경계

- 리움 상권에서 2021–2025년에 관측된 업종은 7개이고, 2025년에 관측된 업종은 6개다. 따라서 ‘전체 업종’은 서울시 서비스업 전체가 아니라 이 상권에서 원자료로 확인되는 업종 합계다.
- 화장품 등 비외식업은 소수 점포·고액 거래로 크게 변동할 수 있다. 그래서 원자료 합계와 동일 업종·완전분기 비교를 함께 제시했다.
- 이 수치는 서비스 도입 목표의 **기준선**으로 쓸 수 있지만, 회복 가능액은 머뭇의 실제 두 번째 방문 전환율, 참여 점포의 객단가, 지속기간을 측정한 뒤 산정해야 한다.
""")
    return {"observed_loss": loss_summary.iloc[0].to_dict(), "comparable_loss": loss_summary.iloc[1].to_dict()}


if __name__ == "__main__":
    print(run())
