"""Describe food-sales links with general-apparel and cosmetics sales.

This deliberately keeps absent industry-quarter observations missing.  A missing
row can mean no report, no qualifying business, or a suppression/coverage issue;
it is not silently converted to zero sales.
"""
from __future__ import annotations

import os
from pathlib import Path

_MPL_CACHE = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..build_panel import remove_duplicate_keys
from ..config import ANALYSIS_QUARTERS, OUTPUT_DIR, REPORT_DIR
from ..export_results import markdown_table, save_csv, write_text
from ..load_data import discover_inputs, load_area_data
from .run_food_nonfood_spillover import (
    TARGET_CODE,
    _cluster_ols,
    _font,
    _load_population,
)


TARGETS = {"CS300011": "일반의류", "CS300022": "화장품"}
OUT_DIR = OUTPUT_DIR / "leeum" / "food_apparel_cosmetics_relationship"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "leeum" / "17_food_apparel_cosmetics_relationship.md"
for directory in (OUT_DIR, FIGURE_DIR, REPORT_PATH.parent):
    directory.mkdir(parents=True, exist_ok=True)


def _quarter_grid(area_names: pd.DataFrame) -> pd.DataFrame:
    quarters = pd.DataFrame({"quarter": ANALYSIS_QUARTERS})
    grid = area_names.assign(_key=1).merge(quarters.assign(_key=1), on="_key").drop(columns="_key")
    grid["quarter_index"] = grid["quarter"].str[:4].astype(int) * 4 + grid["quarter"].str[-1].astype(int)
    return grid.sort_values(["area_code", "quarter_index"]).reset_index(drop=True)


def _yoy_changes(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.sort_values(["area_code", "quarter_index"]).copy()
    for column in columns:
        prior = frame.groupby("area_code")[column].shift(4)
        frame[f"{column}_yoy_log_change"] = np.where(
            (frame[column] > 0) & (prior > 0), np.log(frame[column] / prior), np.nan,
        )
    frame["food_sales_yoy_log_change_lag1"] = frame.groupby("area_code")["food_sales_yoy_log_change"].shift(1)
    frame["quarter_label"] = frame["quarter"].str[:4] + "Q" + frame["quarter"].str[-1]
    return frame


def _build_panels() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    sales = load_area_data(discover_inputs(), "sales_area", industry_codes=None)
    sales, duplicate_audit = remove_duplicate_keys(sales, ["sales_amount", "sales_transactions"])
    if not duplicate_audit.empty:
        save_csv(duplicate_audit, OUT_DIR / "sales_duplicate_resolution.csv")
    sales = sales.loc[sales["quarter"].isin(ANALYSIS_QUARTERS)].copy()
    food = sales.loc[sales["industry_code"].str.startswith("CS100")].groupby(
        ["area_code", "quarter"], as_index=False
    ).agg(food_sales=("sales_amount", "sum"), food_transactions=("sales_transactions", "sum"))
    population = _load_population()
    panels: dict[str, pd.DataFrame] = {}
    availability: list[dict[str, object]] = []
    for code, name in TARGETS.items():
        target = sales.loc[sales["industry_code"].eq(code)].groupby(
            ["area_code", "area_name", "quarter"], as_index=False
        ).agg(target_sales=("sales_amount", "sum"), target_transactions=("sales_transactions", "sum"))
        names = target.groupby("area_code", as_index=False).agg(area_name=("area_name", "first"))
        panel = _quarter_grid(names).merge(food, on=["area_code", "quarter"], how="left")
        panel = panel.merge(target.drop(columns="area_name"), on=["area_code", "quarter"], how="left")
        panel = panel.merge(population, on=["area_code", "quarter"], how="left")
        panel = _yoy_changes(panel, ["food_sales", "target_sales", "floating_population"])
        panel["target_code"] = code
        panel["target_industry"] = name
        panels[code] = panel
        leeum = panel.loc[panel["area_code"].eq(TARGET_CODE)]
        availability.append({
            "target_code": code,
            "target_industry": name,
            "seoul_areas_with_observed_sales": int(target["area_code"].nunique()),
            "leeum_observed_quarters_2022_2025": int(leeum.loc[leeum["quarter"].str[:4].astype(int).between(2022, 2025), "target_sales"].notna().sum()),
            "leeum_total_quarters_2022_2025": 16,
            "leeum_valid_yoy_quarters": int(leeum["target_sales_yoy_log_change"].notna().sum()),
            "leeum_valid_lagged_pair_quarters": int(leeum.dropna(subset=["target_sales_yoy_log_change", "food_sales_yoy_log_change_lag1"]).shape[0]),
        })
    return panels, pd.DataFrame(availability)


def _leeum_index(leeum: pd.DataFrame, column: str) -> pd.Series:
    """Index to first observed quarter in 2022; do not bridge missing values."""
    base = leeum.loc[leeum["quarter"].str.startswith("2022") & leeum[column].gt(0), column]
    return leeum[column] / base.iloc[0] * 100 if not base.empty else pd.Series(np.nan, index=leeum.index)


def _plot(leeum_panels: dict[str, pd.DataFrame], residuals: dict[str, pd.DataFrame], estimates: pd.DataFrame) -> None:
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row, (code, name) in enumerate(TARGETS.items()):
        leeum = leeum_panels[code].loc[leeum_panels[code]["area_code"].eq(TARGET_CODE)].copy()
        visible = leeum.loc[leeum["quarter"].str[:4].astype(int).between(2022, 2025)].copy()
        axes[row, 0].plot(visible["quarter_label"], _leeum_index(visible, "food_sales"), marker="o", label="외식업 매출")
        axes[row, 0].plot(visible["quarter_label"], _leeum_index(visible, "target_sales"), marker="o", label=f"{name} 매출")
        axes[row, 0].axhline(100, color="grey", linewidth=.8)
        axes[row, 0].tick_params(axis="x", rotation=45)
        axes[row, 0].set_ylabel("지수 (각 시리즈의 첫 2022년 관측치=100)")
        axes[row, 0].set_title(f"리움미술관: 외식업·{name} 관측 매출")
        axes[row, 0].legend()
        axes[row, 0].grid(alpha=.2)

        model = f"{code}_lag1_with_population"
        estimate = estimates.loc[(estimates["model"].eq(model)) & (estimates["predictor"].eq("food_sales_yoy_log_change_lag1"))].iloc[0]
        residual = residuals[code]
        sample = residual.sample(min(6000, len(residual)), random_state=42)
        x = sample["food_sales_yoy_log_change_lag1_residual"]
        y = sample["outcome_residual"]
        axes[row, 1].scatter(x * 100, y * 100, s=5, alpha=.12, color="#4575b4")
        low, high = np.quantile(x, [.01, .99])
        line = np.linspace(low, high, 100)
        axes[row, 1].plot(line * 100, float(estimate["coefficient"]) * line * 100, color="#d73027", label=f"계수={estimate['coefficient']:.3f}")
        axes[row, 1].axhline(0, color="grey", linewidth=.8)
        axes[row, 1].axvline(0, color="grey", linewidth=.8)
        axes[row, 1].set_xlabel("직전 분기 외식업 매출 YoY 잔차 (%p)")
        axes[row, 1].set_ylabel(f"{name} 매출 YoY 잔차 (%p)")
        axes[row, 1].set_title(f"서울 상권 패널: 외식업→{name} 1분기 선행 연관성")
        axes[row, 1].legend()
        axes[row, 1].grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "food_apparel_cosmetics_relationship.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run() -> dict[str, object]:
    panels, availability = _build_panels()
    estimates: list[pd.DataFrame] = []
    residuals: dict[str, pd.DataFrame] = {}
    for code, panel in panels.items():
        outcome = "target_sales_yoy_log_change"
        for suffix, predictors in (
            ("same_quarter", ["food_sales_yoy_log_change"]),
            ("lag1", ["food_sales_yoy_log_change_lag1"]),
            ("lag1_with_population", ["food_sales_yoy_log_change_lag1", "floating_population_yoy_log_change"]),
        ):
            result, residual = _cluster_ols(panel, outcome, predictors, f"{code}_{suffix}")
            result["target_code"] = code
            result["target_industry"] = TARGETS[code]
            estimates.append(result)
            if suffix == "lag1_with_population":
                residuals[code] = residual
    estimate_table = pd.concat(estimates, ignore_index=True)
    leeum_panels = {code: panel.loc[panel["area_code"].eq(TARGET_CODE)].copy() for code, panel in panels.items()}
    leeum_profiles = pd.concat(leeum_panels.values(), ignore_index=True)
    save_csv(availability, OUT_DIR / "data_availability.csv")
    save_csv(estimate_table, OUT_DIR / "food_apparel_cosmetics_panel_estimates.csv")
    save_csv(leeum_profiles, OUT_DIR / "leeum_food_apparel_cosmetics_quarterly_profile.csv")
    for code, panel in panels.items():
        save_csv(panel, OUT_DIR / f"seoul_area_quarter_{code}_panel.csv")
        save_csv(residuals[code], OUT_DIR / f"{code}_lag1_with_population_residuals.csv")
    _plot(leeum_panels, residuals, estimate_table)

    key = estimate_table.loc[
        estimate_table["model"].str.endswith("lag1_with_population")
        & estimate_table["predictor"].eq("food_sales_yoy_log_change_lag1")
    ].copy()
    display_key = key.copy()
    for column in ("coefficient", "cluster_se_area", "ci95_low", "ci95_high"):
        display_key[column] = display_key[column].round(3)
    rows: list[str] = []
    for _, item in key.iterrows():
        relation = "0을 포함한다" if item["ci95_low"] <= 0 <= item["ci95_high"] else "0을 포함하지 않는다"
        rows.append(
            f"- **{item['target_industry']}**: 계수 {item['coefficient']:.3f}, 95% 신뢰구간 "
            f"[{item['ci95_low']:.3f}, {item['ci95_high']:.3f}] ({relation})."
        )
    write_text(REPORT_PATH, f"""# 외식업 매출과 일반의류·화장품 매출의 관계

## 질문과 처리 원칙

리움미술관에서 실제로 관측되는 대표 비외식업인 일반의류(`CS300011`)와 화장품(`CS300022`)을 각각 분리해, 외식업 매출 변화와의 관계를 살폈다. 업종이 사라져 행이 없는 분기는 **0으로 채우거나 보간하지 않았다**. 따라서 아래 결과는 관측된 업종×상권×분기에 한정된다.

## 서울 상권 패널 결과

- 표본: 2021Q1–2025Q4, 각 업종이 실제 관측된 서울 상권×분기
- 결과변수: 일반의류 또는 화장품의 전년동기 대비 로그 매출 변화
- 설명변수: 외식업 10개 업종 합계의 전년동기 대비 로그 매출 변화(같은 분기 및 1분기 선행)
- 보정: 상권 고정효과, 분기 고정효과, 유동인구 YoY 변화; 표준오차는 상권 단위 군집화

{markdown_table(display_key, ['target_industry', 'coefficient', 'cluster_se_area', 'ci95_low', 'ci95_high', 'observations', 'areas'], 3)}

{'\n'.join(rows)}

계수가 양수여도 이것은 공통 수요·입지·임대료·관광수요·업종 재편을 완전히 제거하지 못한 **관찰상 연관성**이다. 외식업을 늘리면 의류·화장품 매출이 늘어난다는 인과효과는 아니다.

## 리움미술관의 관측 범위와 대조

{markdown_table(availability, ['target_industry', 'leeum_observed_quarters_2022_2025', 'leeum_total_quarters_2022_2025', 'leeum_valid_yoy_quarters', 'leeum_valid_lagged_pair_quarters'], 20)}

리움의 분기별 실제 관측값은 `leeum_food_apparel_cosmetics_quarterly_profile.csv`, 그래프는 `figures/food_apparel_cosmetics_relationship.png`에 저장했다. 지수 그래프에서 선이 끊긴 부분은 결측을 뜻하며 추정값이 아니다. 일반의류와 화장품은 모두 소수 점포·고액 거래의 영향을 받을 수 있어, 외식업과의 분기 상관만으로 상권 전체 파급을 판단하기에는 표본이 작고 변동성이 크다.

## 판단

일반의류·화장품을 기준으로도, 외식업 활성화가 리움의 비외식업 회복을 보장한다는 증거는 제공되지 않는다. 정책 필요성을 주장하려면 매장 단위 개폐업·임대료·방문객 목적·지원 전후의 비교군을 추가해, 외식업 변화 뒤에 두 업종의 매출·거래가 일관되게 변하는지를 별도로 검증해야 한다.
""")
    return {"availability": availability.to_dict(orient="records"), "lag_estimates": key.to_dict(orient="records")}


if __name__ == "__main__":
    print(run())
