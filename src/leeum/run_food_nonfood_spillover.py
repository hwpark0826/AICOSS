"""Test whether food-sales changes precede non-food sales changes descriptively.

The model uses all Seoul commercial areas, not Leeum alone.  It estimates
within-area, quarter-adjusted associations; it does not identify a causal
effect of food sales on other industries.
"""
from __future__ import annotations

import os
from pathlib import Path

_MPL_CACHE = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..build_panel import remove_duplicate_keys
from ..config import ANALYSIS_QUARTERS, OUTPUT_DIR, RAW_DIR, REPORT_DIR
from ..export_results import markdown_table, save_csv, write_text
from ..load_data import detect_encoding, discover_inputs, load_area_data


TARGET_CODE = "3110091"
OUT_DIR = OUTPUT_DIR / "leeum" / "food_nonfood_spillover"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "leeum" / "16_food_nonfood_spillover.md"
for directory in (OUT_DIR, FIGURE_DIR, REPORT_PATH.parent):
    directory.mkdir(parents=True, exist_ok=True)


def _font() -> str | None:
    for path in (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/NanumGothic.ttf")):
        if path.exists():
            return font_manager.FontProperties(fname=str(path)).get_name()
    return None


def _load_population() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    required = {"기준_년분기_코드", "상권_코드", "총_유동인구_수"}
    for path in sorted(RAW_DIR.glob("*.csv")):
        encoding = detect_encoding(path)
        header = pd.read_csv(path, encoding=encoding, nrows=0).columns.tolist()
        if not required.issubset(header):
            continue
        piece = pd.read_csv(path, encoding=encoding, usecols=["기준_년분기_코드", "상권_코드", "총_유동인구_수"], low_memory=False)
        piece["상권_코드"] = piece["상권_코드"].astype(str).str.replace(".0", "", regex=False).str.strip()
        piece["기준_년분기_코드"] = piece["기준_년분기_코드"].astype(str).str.replace(".0", "", regex=False).str.strip()
        parts.append(piece.loc[piece["기준_년분기_코드"].isin(ANALYSIS_QUARTERS)])
    data = pd.concat(parts, ignore_index=True).drop_duplicates(["기준_년분기_코드", "상권_코드"], keep="first")
    return data.rename(columns={"기준_년분기_코드": "quarter", "상권_코드": "area_code", "총_유동인구_수": "floating_population"})


def _build_panel() -> pd.DataFrame:
    inputs = discover_inputs()
    sales = load_area_data(inputs, "sales_area", industry_codes=None)
    sales, duplicate_audit = remove_duplicate_keys(sales, ["sales_amount", "sales_transactions"])
    if not duplicate_audit.empty:
        save_csv(duplicate_audit, OUT_DIR / "sales_duplicate_resolution.csv")
    sales = sales.loc[sales["quarter"].isin(ANALYSIS_QUARTERS)].copy()
    sales["sector"] = np.where(sales["industry_code"].str.startswith("CS100"), "food", "nonfood")
    grouped = sales.groupby(["area_code", "area_name", "quarter", "sector"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"), transactions=("sales_transactions", "sum"),
    )
    wide = grouped.pivot(index=["area_code", "area_name", "quarter"], columns="sector", values=["sales_amount", "transactions"]).reset_index()
    wide.columns = ["_".join(str(part) for part in column if part) if isinstance(column, tuple) else column for column in wide.columns]
    wide = wide.rename(columns={
        "sales_amount_food": "food_sales", "sales_amount_nonfood": "nonfood_sales",
        "transactions_food": "food_transactions", "transactions_nonfood": "nonfood_transactions",
    })
    panel = wide.merge(_load_population(), on=["area_code", "quarter"], how="left")
    panel["quarter_index"] = panel["quarter"].str[:4].astype(int) * 4 + panel["quarter"].str[-1].astype(int)
    panel = panel.sort_values(["area_code", "quarter_index"]).reset_index(drop=True)
    for column in ("food_sales", "nonfood_sales", "floating_population"):
        prior = panel.groupby("area_code")[column].shift(4)
        panel[f"{column}_yoy_log_change"] = np.where((panel[column] > 0) & (prior > 0), np.log(panel[column] / prior), np.nan)
    panel["food_sales_yoy_log_change_lag1"] = panel.groupby("area_code")["food_sales_yoy_log_change"].shift(1)
    panel["quarter_label"] = panel["quarter"].str[:4] + "Q" + panel["quarter"].str[-1]
    return panel


def _twfe_residual(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Residualize columns against area and quarter fixed effects by iteration."""
    work = frame[columns].astype(float).copy()
    area, quarter = frame["area_code"], frame["quarter"]
    for _ in range(30):
        before = work.to_numpy(copy=True)
        work = work - work.groupby(area).transform("mean")
        work = work - work.groupby(quarter).transform("mean")
        if np.max(np.abs(work.to_numpy() - before)) < 1e-10:
            break
    return work


def _cluster_ols(frame: pd.DataFrame, outcome: str, predictors: list[str], model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    use = frame.dropna(subset=[outcome, *predictors]).copy()
    residual = _twfe_residual(use, [outcome, *predictors])
    y = residual[outcome].to_numpy()
    x = residual[predictors].to_numpy()
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    errors = y - x @ beta
    meat = np.zeros((len(predictors), len(predictors)))
    for _, indices in use.groupby("area_code").groups.items():
        position = use.index.get_indexer(indices)
        x_group = x[position]
        u_group = errors[position]
        score = x_group.T @ u_group
        meat += np.outer(score, score)
    groups, nobs, k = use["area_code"].nunique(), len(use), len(predictors)
    correction = (groups / max(groups - 1, 1)) * ((nobs - 1) / max(nobs - k, 1))
    covariance = correction * xtx_inv @ meat @ xtx_inv
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0, None))
    result = pd.DataFrame({
        "model": model, "outcome": outcome, "predictor": predictors, "coefficient": beta,
        "cluster_se_area": standard_error, "ci95_low": beta - 1.96 * standard_error,
        "ci95_high": beta + 1.96 * standard_error, "observations": nobs, "areas": groups,
        "fixed_effects": "area + quarter",
    })
    residual_plot = use[["area_code", "quarter", "quarter_label"]].copy()
    residual_plot["outcome_residual"] = residual[outcome].to_numpy()
    for predictor in predictors:
        residual_plot[f"{predictor}_residual"] = residual[predictor].to_numpy()
    return result, residual_plot


def _annual_summary(leeum: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete quarters so Leeum's actual trajectory is readable."""
    annual = leeum.assign(year=leeum["quarter"].str[:4].astype(int)).groupby("year", as_index=False).agg(
        food_sales=("food_sales", "sum"),
        nonfood_sales=("nonfood_sales", "sum"),
        food_transactions=("food_transactions", "sum"),
        nonfood_transactions=("nonfood_transactions", "sum"),
        floating_population=("floating_population", "sum"),
        observed_quarters=("quarter", "nunique"),
    )
    baseline = annual.loc[annual["year"].eq(2022)].iloc[0]
    for column in ("food_sales", "nonfood_sales", "food_transactions", "nonfood_transactions", "floating_population"):
        annual[f"{column}_index_2022_100"] = annual[column] / baseline[column] * 100
    return annual


def _plot(leeum: pd.DataFrame, annual: pd.DataFrame, lag_residual: pd.DataFrame, beta: float) -> None:
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    shown = leeum.dropna(subset=["food_sales_yoy_log_change", "nonfood_sales_yoy_log_change"])
    axes[0].plot(shown["quarter_label"], shown["food_sales_yoy_log_change"] * 100, marker="o", label="외식업 매출 YoY")
    axes[0].plot(shown["quarter_label"], shown["nonfood_sales_yoy_log_change"] * 100, marker="o", label="비외식업 매출 YoY")
    axes[0].axhline(0, color="grey", linewidth=.8)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("전년동기 대비 로그변화 (%)")
    axes[0].set_title("리움미술관: 외식업·비외식업 매출 변화")
    axes[0].legend()
    axes[0].grid(alpha=.2)
    for column, label, color in (
        ("food_sales_index_2022_100", "외식업 매출", "#d73027"),
        ("nonfood_sales_index_2022_100", "비외식업 매출", "#4575b4"),
        ("floating_population_index_2022_100", "유동인구", "#1a9850"),
    ):
        axes[1].plot(annual["year"], annual[column], marker="o", label=label, color=color)
    axes[1].axhline(100, color="grey", linewidth=.8)
    axes[1].set_xticks(annual["year"])
    axes[1].set_ylabel("연간 합계 지수 (2022=100)")
    axes[1].set_title("리움미술관: 실제 연간 경로")
    axes[1].legend()
    axes[1].grid(alpha=.2)
    sample = lag_residual.sample(min(len(lag_residual), 6000), random_state=42)
    x = sample["food_sales_yoy_log_change_lag1_residual"]
    y = sample["outcome_residual"]
    axes[2].scatter(x * 100, y * 100, s=5, alpha=.12, color="#4575b4")
    limits = np.quantile(x, [.01, .99])
    line = np.linspace(*limits, 100)
    axes[2].plot(line * 100, beta * line * 100, color="#d73027", label="고정효과 잔차 회귀선")
    axes[2].axhline(0, color="grey", linewidth=.8)
    axes[2].axvline(0, color="grey", linewidth=.8)
    axes[2].set_xlabel("직전 분기 외식업 매출 YoY 잔차 (%p)")
    axes[2].set_ylabel("비외식업 매출 YoY 잔차 (%p)")
    axes[2].set_title("서울 상권 패널: 1분기 선행 연관성")
    axes[2].legend()
    axes[2].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "food_nonfood_lag_association.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run() -> dict[str, object]:
    panel = _build_panel()
    outcome = "nonfood_sales_yoy_log_change"
    specs = [
        ("same_quarter", ["food_sales_yoy_log_change"]),
        ("lag1", ["food_sales_yoy_log_change_lag1"]),
        ("lag1_with_population", ["food_sales_yoy_log_change_lag1", "floating_population_yoy_log_change"]),
    ]
    results, residuals = [], {}
    for name, predictors in specs:
        result, residual = _cluster_ols(panel, outcome, predictors, name)
        results.append(result)
        residuals[name] = residual
    estimates = pd.concat(results, ignore_index=True)
    lag_beta = float(estimates.loc[(estimates["model"].eq("lag1_with_population")) & (estimates["predictor"].eq("food_sales_yoy_log_change_lag1")), "coefficient"].iloc[0])
    leeum = panel.loc[panel["area_code"].eq(TARGET_CODE)].copy()
    leeum["lag1_association_component"] = lag_beta * leeum["food_sales_yoy_log_change_lag1"]
    annual = _annual_summary(leeum)
    save_csv(panel, OUT_DIR / "seoul_area_quarter_food_nonfood_panel.csv")
    save_csv(estimates, OUT_DIR / "food_nonfood_panel_estimates.csv")
    save_csv(leeum, OUT_DIR / "leeum_food_nonfood_quarterly_profile.csv")
    save_csv(annual, OUT_DIR / "leeum_food_nonfood_annual_summary.csv")
    save_csv(residuals["lag1_with_population"], OUT_DIR / "lag1_with_population_residuals.csv")
    _plot(leeum, annual, residuals["lag1_with_population"], lag_beta)
    lag_row = estimates.loc[(estimates["model"].eq("lag1_with_population")) & (estimates["predictor"].eq("food_sales_yoy_log_change_lag1"))].iloc[0]
    same_row = estimates.loc[(estimates["model"].eq("same_quarter")) & (estimates["predictor"].eq("food_sales_yoy_log_change"))].iloc[0]
    inference = "0을 포함한다" if lag_row["ci95_low"] <= 0 <= lag_row["ci95_high"] else "0을 포함하지 않는다"
    annual_2022 = annual.loc[annual["year"].eq(2022)].iloc[0]
    annual_2025 = annual.loc[annual["year"].eq(2025)].iloc[0]
    food_change = (annual_2025["food_sales"] / annual_2022["food_sales"] - 1) * 100
    nonfood_change = (annual_2025["nonfood_sales"] / annual_2022["nonfood_sales"] - 1) * 100
    population_change = (annual_2025["floating_population"] / annual_2022["floating_population"] - 1) * 100
    write_text(REPORT_PATH, f"""# 외식업 매출 변화와 비외식업 매출 변화의 연관성

## 질문

리움미술관 외식업을 활성화할 필요성을 검토하기 위해, 서울 상권 패널에서 외식업 매출 변화가 비외식업 매출 변화와 동반되거나 선행하는지를 분석했다. 비외식업 결과변수에는 외식업 매출을 포함하지 않아 부분-전체의 기계적 관계를 피했다.

## 설계

- 표본: 2021Q1–2025Q4 서울 상권×분기, 매출이 관측된 63개 업종
- 외식업: 업종코드 `CS100*` 10개 업종 합계; 비외식업: 나머지 53개 업종 합계
- 변화율: 전년동기 대비 로그 매출 변화
- 모형: 상권 고정효과와 분기 고정효과를 제거한 회귀, 표준오차는 상권 단위 군집화
- 핵심 설명변수: 직전 분기 외식업 매출 YoY 변화; 보조 사양에는 유동인구 YoY 변화 추가

## 추정 결과

{markdown_table(estimates, ['model', 'predictor', 'coefficient', 'cluster_se_area', 'ci95_low', 'ci95_high', 'observations', 'areas'], 3)}

유동인구를 보정한 1분기 선행 사양에서 외식업 변화 계수는 **{lag_row['coefficient']:.3f}**이고 95% 신뢰구간은 **[{lag_row['ci95_low']:.3f}, {lag_row['ci95_high']:.3f}]**로 {inference}. 계수가 양수라면 외식업 매출 변화가 낮은 상권·분기일수록 다음 분기 비외식업 매출 변화도 낮게 관측되는 방향의 연관성을 뜻한다. 다만 크기는 작다. 근사적으로 외식업 YoY 변화가 10%p 낮을 때 다음 분기 비외식업 YoY 변화는 평균 **0.4%p** 낮게 연관된다. 같은 분기 계수는 **{same_row['coefficient']:.3f}**이다.

## 리움미술관에 대한 대조 검증

리움의 실제 연간 합계는 `leeum_food_nonfood_annual_summary.csv`에서 별도로 확인했다. 2022→2025년에 외식업 매출은 **{food_change:.1f}%** 변했지만, 비외식업 매출은 **{nonfood_change:.1f}%**, 유동인구는 **{population_change:.1f}%** 변했다. 즉 이 데이터에서는 외식업의 감소와 비외식업의 동반 감소가 리움에서 관측되지 않는다. 따라서 서울 전체의 약한 양(+)의 선행 연관성을 리움에 기계적으로 적용해 “외식업 감소가 비외식업 감소를 만들었다”고 말할 수 없다.

비외식업은 53개 세부 업종의 합계이고 리움에서는 거래 건수가 상대적으로 작아 특정 업종의 진입·퇴출에 민감할 수 있다. 따라서 이 결과는 비외식업이 안정적이라는 증명도 아니다. 다만 **현재 집계 수준에서는 외식업 하락을 근거로 비외식업까지 함께 살려야 한다는 논거는 지지되지 않는다.**

서울 패널 분석도 관광수요·임대료·상권 접근성·공사 등 공통 원인과 역인과를 완전히 제거하지 못한다. 외식업을 살리면 비외식업 매출이 반드시 증가한다는 인과효과나 정책효과를 증명하지 않는다.

## 정책 판단 경계

이 결과만으로 외식업 지원을 정당화하지 않는다. 현재로서는 외식업을 **상권 전체 회복의 수단**으로 지원해야 한다는 근거가 약하다. 정책 필요성은 (1) 외식업 자체의 보존 가치 또는 공공목적, (2) 점포·방문객·임대료 등 현장 자료에서 확인된 경로, (3) 지원 전후 비교가 가능한 소규모 실험에서 비외식업 파급이 확인될 때 강화된다.
""")
    return {"observations": int(lag_row["observations"]), "areas": int(lag_row["areas"]), "lag1_food_coefficient": float(lag_row["coefficient"]), "lag1_ci95": [float(lag_row["ci95_low"]), float(lag_row["ci95_high"])]}


if __name__ == "__main__":
    print(run())
