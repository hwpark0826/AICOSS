"""Create clearly separated 2021-base absolute and relative industry charts."""
from __future__ import annotations

from pathlib import Path
import os

_MPL_CACHE = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..calculate_benchmarks import annual_benchmarks
from ..config import OUTPUT_DIR, REPORT_DIR
from ..export_results import markdown_table, save_csv, write_text


TARGET_CODE = "3110091"
LEEUM_DIR = OUTPUT_DIR / "leeum"
FIGURE_DIR = LEEUM_DIR / "figures"
LEEUM_REPORT_DIR = REPORT_DIR / "leeum"
SELECTED_CODES = ("CS100003", "CS100004", "CS100010")
YEARS = (2021, 2022, 2023, 2024, 2025)


def _font() -> str | None:
    for path in (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/NanumGothic.ttf")):
        if path.exists():
            return font_manager.FontProperties(fname=str(path)).get_name()
    return None


def _configure_plot() -> None:
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _read_annual() -> pd.DataFrame:
    data = pd.read_csv("data/processed/food_commercial_area_annual_panel.csv")
    data["area_code"] = data["area_code"].astype(str).str.replace(".0", "", regex=False)
    return data


def _pct(start: float, end: float) -> float:
    return end / start - 1 if pd.notna(start) and pd.notna(end) and start != 0 else np.nan


def _build(annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = annual.loc[
        annual["area_code"].eq(TARGET_CODE) & annual["industry_code"].isin(SELECTED_CODES) & annual["year"].isin(YEARS)
    ].copy()
    completeness = target.groupby("industry_code")["complete_year"].all()
    valid_codes = completeness.loc[completeness].index.tolist()
    target = target.loc[target["industry_code"].isin(valid_codes)].copy()
    benchmark = annual_benchmarks(annual).loc[lambda d: d["industry_code"].isin(valid_codes) & d["year"].isin(YEARS), ["year", "industry_code", "seoul_sales_amount"]]
    series = target.merge(benchmark, on=["year", "industry_code"], how="left", validate="one_to_one")
    base = series.loc[series["year"].eq(2021), ["industry_code", "sales_amount", "seoul_sales_amount"]].rename(columns={"sales_amount": "sales_2021", "seoul_sales_amount": "seoul_sales_2021"})
    series = series.merge(base, on="industry_code", how="left", validate="many_to_one")
    series["absolute_sales_index_2021_100"] = series["sales_amount"] / series["sales_2021"] * 100
    series["relative_sales_index_2021_100"] = (series["sales_amount"] / series["sales_2021"]) / (series["seoul_sales_amount"] / series["seoul_sales_2021"]) * 100
    series["relative_log_performance_vs_seoul"] = np.log(series["relative_sales_index_2021_100"] / 100)
    endpoint = series.loc[series["year"].eq(2021), ["industry_code", "industry_name", "sales_amount", "seoul_sales_amount"]].rename(columns={"sales_amount": "target_sales_2021", "seoul_sales_amount": "seoul_sales_2021"})
    endpoint = endpoint.merge(
        series.loc[series["year"].eq(2025), ["industry_code", "sales_amount", "seoul_sales_amount", "relative_sales_index_2021_100", "relative_log_performance_vs_seoul"]].rename(columns={"sales_amount": "target_sales_2025", "seoul_sales_amount": "seoul_sales_2025", "relative_sales_index_2021_100": "relative_index_2025_2021_100", "relative_log_performance_vs_seoul": "relative_log_change_2021_2025"}),
        on="industry_code", how="inner", validate="one_to_one",
    )
    endpoint["target_absolute_change_2021_2025"] = endpoint["target_sales_2025"] - endpoint["target_sales_2021"]
    endpoint["target_absolute_pct_change_2021_2025"] = endpoint.apply(lambda row: _pct(row["target_sales_2021"], row["target_sales_2025"]), axis=1)
    endpoint["seoul_pct_change_2021_2025"] = endpoint.apply(lambda row: _pct(row["seoul_sales_2021"], row["seoul_sales_2025"]), axis=1)
    return series.sort_values(["industry_code", "year"]), endpoint.sort_values("industry_code"), target


def _plot_absolute(series: pd.DataFrame) -> None:
    _configure_plot()
    industries = list(series[["industry_code", "industry_name"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(1, len(industries), figsize=(13, 4.2), sharex=True)
    for axis, (code, name) in zip(np.ravel(axes), industries):
        part = series.loc[series["industry_code"].eq(code)]
        axis.plot(part["year"], part["sales_amount"] / 100_000_000, marker="o", linewidth=1.8)
        axis.set_title(name)
        axis.set_xticks(YEARS)
        axis.set_ylabel("매출액 (억원)")
        axis.grid(alpha=.2)
    fig.suptitle("업종별 절대 매출 추이 (2021~2025, 각 패널 축 독립)", y=1.02)
    _savefig(FIGURE_DIR / "industry_2021_2025_absolute_sales.png")


def _plot_relative(series: pd.DataFrame) -> None:
    _configure_plot()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for _, part in series.groupby("industry_code"):
        name = part["industry_name"].iloc[0]
        ax.plot(part["year"], part["relative_sales_index_2021_100"], marker="o", linewidth=1.8, label=name)
    ax.axhline(100, color="grey", linestyle="--", linewidth=.9)
    ax.set_xticks(YEARS)
    ax.set_ylabel("서울 동일 업종 대비 상대 매출지수 (2021=100)")
    ax.set_title("업종별 상대 매출성과: 리움 변화율 ÷ 서울 동일 업종 변화율")
    ax.legend()
    ax.grid(alpha=.2)
    _savefig(FIGURE_DIR / "industry_2021_2025_relative_sales_index.png")


def run() -> dict[str, object]:
    annual = _read_annual()
    series, endpoint, _ = _build(annual)
    if series["industry_code"].nunique() != 3:
        raise ValueError("Expected exactly three complete 2021→2025 industries.")
    save_csv(series, LEEUM_DIR / "industry_2021_2025_absolute_relative_series.csv")
    save_csv(endpoint, LEEUM_DIR / "industry_2021_2025_absolute_relative_endpoints.csv")
    _plot_absolute(series)
    _plot_relative(series)
    report = f"""# 11. 2021~2025 완결 업종: 절대·상대 매출 시각화

## 대상과 전제

대상은 2021년과 2025년이 모두 4개 분기로 완결된 **일식음식점·양식음식점·커피·음료**다. 한식음식점은 2021년이 불완전하여 이 비교에서 제외했다.

## 절대 매출 그림: `industry_2021_2025_absolute_sales.png`

각 패널은 해당 업종의 리움 상권 연간 매출액(억원)이다. 패널마다 세로축 범위가 독립적이므로, 선의 기울기는 업종 내부 추이를 읽는 데 사용하고 업종 간 절대 규모는 표로 확인한다.

## 상대 매출 그림: `industry_2021_2025_relative_sales_index.png`

상대 매출지수는 `(리움 업종 매출 / 2021년 리움 업종 매출) ÷ (서울 동일 업종 매출 / 2021년 서울 동일 업종 매출) × 100`이다.

- 100: 2021년 이후 서울 동일 업종과 같은 변화율
- 100 미만: 서울 동일 업종보다 리움 해당 업종의 변화율이 더 낮음
- 100 초과: 서울 동일 업종보다 변화율이 더 높음

이는 절대 매출액, 서울 시장점유율, 인과효과가 아니다. 서울 기준선은 완결연도 상권 단위 자료를 업종별로 합산한 값이다.

## 2021→2025 종점 수치

{markdown_table(endpoint, ['industry_name', 'target_sales_2021', 'target_sales_2025', 'target_absolute_pct_change_2021_2025', 'seoul_pct_change_2021_2025', 'relative_index_2025_2021_100', 'relative_log_change_2021_2025'], 10)}
"""
    write_text(LEEUM_REPORT_DIR / "11_industry_2021_2025_visual_guide.md", report)
    return {"industries": int(series["industry_code"].nunique()), "rows": len(series)}


if __name__ == "__main__":
    print(run())
