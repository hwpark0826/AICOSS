"""Compare Leeum Museum and Samcheong-dong on observed commercial-area data.

This is a descriptive comparison.  It does not treat similar shop mix, nearby
cultural facilities, or a shared trend as evidence of a common cause.
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

from ..config import OUTPUT_DIR, RAW_DIR, REPORT_DIR
from ..export_results import markdown_table, save_csv, write_text
from ..load_data import detect_encoding


AREAS = {"3110091": "리움미술관", "3120005": "삼청동"}
YEARS = (2022, 2023, 2024, 2025)
QUARTERS = tuple(f"{year}{quarter}" for year in YEARS for quarter in range(1, 5))
OUT_DIR = OUTPUT_DIR / "leeum" / "samcheong"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "leeum" / "13_samcheong_comparison.md"
for directory in (OUT_DIR, FIGURE_DIR, REPORT_PATH.parent):
    directory.mkdir(parents=True, exist_ok=True)

SEX = {"남성": ("남성_매출_금액", "남성_매출_건수", "남성_유동인구_수"), "여성": ("여성_매출_금액", "여성_매출_건수", "여성_유동인구_수")}
AGES = {f"{age}대": (f"연령대_{age}_매출_금액", f"연령대_{age}_매출_건수", f"연령대_{age}_유동인구_수") for age in (10, 20, 30, 40, 50)}
AGES["60대 이상"] = ("연령대_60_이상_매출_금액", "연령대_60_이상_매출_건수", "연령대_60_이상_유동인구_수")
TIMES = {
    "00–06시": ("시간대_00~06_매출_금액", "시간대_건수~06_매출_건수", "시간대_00_06_유동인구_수"),
    "06–11시": ("시간대_06~11_매출_금액", "시간대_건수~11_매출_건수", "시간대_06_11_유동인구_수"),
    "11–14시": ("시간대_11~14_매출_금액", "시간대_건수~14_매출_건수", "시간대_11_14_유동인구_수"),
    "14–17시": ("시간대_14~17_매출_금액", "시간대_건수~17_매출_건수", "시간대_14_17_유동인구_수"),
    "17–21시": ("시간대_17~21_매출_금액", "시간대_건수~21_매출_건수", "시간대_17_21_유동인구_수"),
    "21–24시": ("시간대_21~24_매출_금액", "시간대_건수~24_매출_건수", "시간대_21_24_유동인구_수"),
}


def _code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".0", "", regex=False).str.strip()


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


def _source_files(required: set[str]) -> list[tuple[Path, str, list[str]]]:
    found: list[tuple[Path, str, list[str]]] = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        encoding = detect_encoding(path)
        header = pd.read_csv(path, encoding=encoding, nrows=0).columns.tolist()
        if required.issubset(header):
            found.append((path, encoding, header))
    return found


def _read_sales() -> pd.DataFrame:
    required = {"기준_년분기_코드", "상권_코드", "서비스_업종_코드", "당월_매출_금액", "당월_매출_건수"}
    columns = ["기준_년분기_코드", "상권_코드", "상권_코드_명", "서비스_업종_코드", "서비스_업종_코드_명", "당월_매출_금액", "당월_매출_건수"]
    columns += [item for triple in [*SEX.values(), *AGES.values(), *TIMES.values()] for item in triple[:2]]
    parts: list[pd.DataFrame] = []
    for path, encoding, header in _source_files(required):
        use = [column for column in columns if column in header]
        for chunk in pd.read_csv(path, encoding=encoding, usecols=use, chunksize=100_000, low_memory=False):
            chunk["상권_코드"] = _code(chunk["상권_코드"])
            chunk["기준_년분기_코드"] = _code(chunk["기준_년분기_코드"])
            chunk = chunk.loc[chunk["상권_코드"].isin(AREAS) & chunk["기준_년분기_코드"].isin(QUARTERS)]
            if not chunk.empty:
                parts.append(chunk)
    data = pd.concat(parts, ignore_index=True).drop_duplicates(["기준_년분기_코드", "상권_코드", "서비스_업종_코드"], keep="first")
    return data.rename(columns={"기준_년분기_코드": "quarter", "상권_코드": "area_code", "상권_코드_명": "area_name", "서비스_업종_코드": "industry_code", "서비스_업종_코드_명": "industry_name", "당월_매출_금액": "sales_amount", "당월_매출_건수": "transactions"})


def _read_population() -> pd.DataFrame:
    required = {"기준_년분기_코드", "상권_코드", "총_유동인구_수"}
    columns = ["기준_년분기_코드", "상권_코드", "상권_코드_명", "총_유동인구_수"] + [item for triple in [*SEX.values(), *AGES.values(), *TIMES.values()] for item in [triple[2]]]
    parts: list[pd.DataFrame] = []
    for path, encoding, header in _source_files(required):
        use = [column for column in columns if column in header]
        piece = pd.read_csv(path, encoding=encoding, usecols=use, low_memory=False)
        piece["상권_코드"] = _code(piece["상권_코드"])
        piece["기준_년분기_코드"] = _code(piece["기준_년분기_코드"])
        parts.append(piece.loc[piece["상권_코드"].isin(AREAS) & piece["기준_년분기_코드"].isin(QUARTERS)])
    data = pd.concat(parts, ignore_index=True).drop_duplicates(["기준_년분기_코드", "상권_코드"], keep="first")
    return data.rename(columns={"기준_년분기_코드": "quarter", "상권_코드": "area_code", "상권_코드_명": "area_name", "총_유동인구_수": "floating_population"})


def _read_stores() -> pd.DataFrame:
    required = {"기준_년분기_코드", "상권_코드", "서비스_업종_코드", "전체_점포_수"}
    columns = ["기준_년분기_코드", "상권_코드", "서비스_업종_코드", "서비스_업종_코드_명", "전체_점포_수"]
    parts: list[pd.DataFrame] = []
    for path, encoding, header in _source_files(required):
        piece = pd.read_csv(path, encoding=encoding, usecols=columns, low_memory=False)
        piece["상권_코드"] = _code(piece["상권_코드"])
        piece["기준_년분기_코드"] = _code(piece["기준_년분기_코드"])
        parts.append(piece.loc[piece["상권_코드"].isin(AREAS) & piece["기준_년분기_코드"].isin(QUARTERS)])
    data = pd.concat(parts, ignore_index=True).drop_duplicates(["기준_년분기_코드", "상권_코드", "서비스_업종_코드"], keep="first")
    return data.rename(columns={"기준_년분기_코드": "quarter", "상권_코드": "area_code", "서비스_업종_코드": "industry_code", "서비스_업종_코드_명": "industry_name", "전체_점포_수": "store_count"})


def _long_components(sales: pd.DataFrame, population: pd.DataFrame, mapping: dict[str, tuple[str, str, str]], kind: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    population = population.set_index(["quarter", "area_code"])
    for segment, (sales_col, tx_col, population_col) in mapping.items():
        grouped = sales.groupby(["quarter", "area_code"], as_index=False).agg(sales_amount=(sales_col, "sum"), transactions=(tx_col, "sum"))
        grouped["floating_population"] = [population.at[(row.quarter, row.area_code), population_col] for row in grouped.itertuples()]
        grouped["segment"] = segment
        grouped["dimension"] = kind
        rows.append(grouped)
    result = pd.concat(rows, ignore_index=True)
    result["average_ticket"] = result["sales_amount"] / result["transactions"].replace(0, np.nan)
    return result


def _annual(overall: pd.DataFrame) -> pd.DataFrame:
    annual = overall.assign(year=overall["quarter"].str[:4].astype(int)).groupby(["area_code", "area_name", "year"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"), transactions=("transactions", "sum"), floating_population=("floating_population", "mean"),
    )
    annual["average_ticket"] = annual["sales_amount"] / annual["transactions"].replace(0, np.nan)
    for metric in ("sales_amount", "transactions", "floating_population", "average_ticket"):
        annual[f"{metric}_index_2022_100"] = annual.groupby("area_code")[metric].transform(lambda values: values / values.iloc[0] * 100)
    return annual


def _plot_trends(annual: pd.DataFrame) -> None:
    _configure_plot()
    metrics = [("sales_amount_index_2022_100", "매출액 지수"), ("transactions_index_2022_100", "거래건수 지수"), ("average_ticket_index_2022_100", "건당 매출 지수"), ("floating_population_index_2022_100", "유동인구 지수")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        for code, name in AREAS.items():
            frame = annual.loc[annual["area_code"].eq(code)]
            axis.plot(frame["year"], frame[metric], marker="o", linewidth=2, label=name)
        axis.axhline(100, color="grey", linewidth=.8, linestyle="--")
        axis.set_title(title)
        axis.set_xticks(YEARS)
        axis.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("리움미술관과 삼청동: 2022=100 연간 지수", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "leeum_samcheong_annual_indices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_composition(components: pd.DataFrame) -> None:
    _configure_plot()
    shown = components.loc[components["quarter"].str[:4].eq("2025")].groupby(["area_code", "dimension", "segment"], as_index=False).agg(sales_amount=("sales_amount", "sum"), transactions=("transactions", "sum"))
    shown["sales_share"] = shown["sales_amount"] / shown.groupby(["area_code", "dimension"])["sales_amount"].transform("sum") * 100
    shown["average_ticket"] = shown["sales_amount"] / shown["transactions"].replace(0, np.nan)
    dimensions = [("sex", "성별"), ("age", "연령대"), ("time", "시간대")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    colors = ["#4575b4", "#d73027"]
    for col, (dimension, title) in enumerate(dimensions):
        frame = shown.loc[shown["dimension"].eq(dimension)]
        segments = frame["segment"].drop_duplicates().tolist()
        x = np.arange(len(segments))
        for index, (code, name) in enumerate(AREAS.items()):
            part = frame.loc[frame["area_code"].eq(code)].set_index("segment").reindex(segments)
            axes[0, col].bar(x + (index - .5) * .35, part["sales_share"], .35, label=name, color=colors[index])
            axes[1, col].bar(x + (index - .5) * .35, part["average_ticket"] / 1_000, .35, label=name, color=colors[index])
        axes[0, col].set_title(f"{title}별 매출 구성 (2025)")
        axes[0, col].set_ylabel("매출 비중 (%)")
        axes[1, col].set_title(f"{title}별 건당 매출 (2025)")
        axes[1, col].set_ylabel("천 원")
        for axis in (axes[0, col], axes[1, col]):
            axis.set_xticks(x, segments, rotation=40 if dimension != "sex" else 0, ha="right")
            axis.grid(axis="y", alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("매출 구성과 건당 매출: 리움미술관 vs 삼청동", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "leeum_samcheong_demographic_time_comparison_2025.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_population_and_transactions(components: pd.DataFrame) -> None:
    _configure_plot()
    shown = components.loc[components["quarter"].str[:4].eq("2025")].groupby(["area_code", "dimension", "segment"], as_index=False).agg(
        transactions=("transactions", "sum"), floating_population=("floating_population", "mean"),
    )
    shown["transaction_share"] = shown["transactions"] / shown.groupby(["area_code", "dimension"])["transactions"].transform("sum") * 100
    shown["population_share"] = shown["floating_population"] / shown.groupby(["area_code", "dimension"])["floating_population"].transform("sum") * 100
    dimensions = [("sex", "성별"), ("age", "연령대"), ("time", "시간대")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    colors = ["#4575b4", "#d73027"]
    for col, (dimension, title) in enumerate(dimensions):
        frame = shown.loc[shown["dimension"].eq(dimension)]
        segments = frame["segment"].drop_duplicates().tolist()
        x = np.arange(len(segments))
        for index, (code, name) in enumerate(AREAS.items()):
            part = frame.loc[frame["area_code"].eq(code)].set_index("segment").reindex(segments)
            axes[0, col].bar(x + (index - .5) * .35, part["population_share"], .35, label=name, color=colors[index])
            axes[1, col].bar(x + (index - .5) * .35, part["transaction_share"], .35, label=name, color=colors[index])
        axes[0, col].set_title(f"{title}별 유동인구 구성 (2025)")
        axes[0, col].set_ylabel("유동인구 비중 (%)")
        axes[1, col].set_title(f"{title}별 거래 구성 (2025)")
        axes[1, col].set_ylabel("거래건수 비중 (%)")
        for axis in (axes[0, col], axes[1, col]):
            axis.set_xticks(x, segments, rotation=40 if dimension != "sex" else 0, ha="right")
            axis.grid(axis="y", alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("유동인구 구성과 거래 구성: 리움미술관 vs 삼청동", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "leeum_samcheong_population_transaction_comparison_2025.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _start_end_summary(annual: pd.DataFrame, components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_wide = annual.loc[annual["year"].isin([2022, 2025])].pivot(index=["area_code", "area_name"], columns="year", values=["sales_amount", "transactions", "floating_population", "average_ticket"])
    annual_wide.columns = [f"{metric}_{year}" for metric, year in annual_wide.columns]
    annual_wide = annual_wide.reset_index()
    for metric in ("sales_amount", "transactions", "floating_population", "average_ticket"):
        annual_wide[f"{metric}_change_2022_2025"] = annual_wide[f"{metric}_2025"] / annual_wide[f"{metric}_2022"] - 1
    segment = components.loc[components["year"].isin([2022, 2025])].groupby(["area_code", "dimension", "segment", "year"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"), transactions=("transactions", "sum"), floating_population=("floating_population", "mean"),
    )
    segment["sales_share"] = segment["sales_amount"] / segment.groupby(["area_code", "dimension", "year"])["sales_amount"].transform("sum")
    segment["population_share"] = segment["floating_population"] / segment.groupby(["area_code", "dimension", "year"])["floating_population"].transform("sum")
    segment["average_ticket"] = segment["sales_amount"] / segment["transactions"].replace(0, np.nan)
    return annual_wide, segment


def run() -> dict[str, object]:
    sales, population, stores = _read_sales(), _read_population(), _read_stores()
    overall = sales.groupby(["quarter", "area_code", "area_name"], as_index=False).agg(sales_amount=("sales_amount", "sum"), transactions=("transactions", "sum"))
    overall = overall.merge(population[["quarter", "area_code", "floating_population"]], on=["quarter", "area_code"], how="left")
    overall["average_ticket"] = overall["sales_amount"] / overall["transactions"].replace(0, np.nan)
    overall["year"] = overall["quarter"].str[:4].astype(int)
    annual = _annual(overall)
    components = pd.concat([_long_components(sales, population, SEX, "sex"), _long_components(sales, population, AGES, "age"), _long_components(sales, population, TIMES, "time")], ignore_index=True)
    components["year"] = components["quarter"].str[:4].astype(int)
    annual_change, segment_summary = _start_end_summary(annual, components)
    mix = stores.assign(year=stores["quarter"].str[:4].astype(int)).loc[lambda frame: frame["year"].eq(2025)].groupby(["area_code", "industry_code", "industry_name"], as_index=False).agg(mean_store_count=("store_count", "mean"))
    mix["store_share_2025"] = mix["mean_store_count"] / mix.groupby("area_code")["mean_store_count"].transform("sum")
    mix = mix.sort_values(["area_code", "mean_store_count"], ascending=[True, False])
    save_csv(overall, OUT_DIR / "quarterly_overall_metrics.csv")
    save_csv(annual, OUT_DIR / "annual_comparison.csv")
    save_csv(annual_change, OUT_DIR / "annual_change_2022_2025.csv")
    save_csv(components, OUT_DIR / "demographic_time_metrics.csv")
    save_csv(segment_summary, OUT_DIR / "segment_summary_2022_2025.csv")
    save_csv(mix, OUT_DIR / "industry_mix_2025.csv")
    _plot_trends(annual)
    _plot_composition(components)
    _plot_population_and_transactions(components)
    annual_2025 = annual.loc[annual["year"].eq(2025), ["area_name", "sales_amount_index_2022_100", "transactions_index_2022_100", "average_ticket_index_2022_100", "floating_population_index_2022_100"]]
    annual_change_report = annual_change[["area_name", "sales_amount_change_2022_2025", "transactions_change_2022_2025", "average_ticket_change_2022_2025", "floating_population_change_2022_2025"]]
    top_mix = mix.groupby("area_code", group_keys=False).head(5)
    def segment_value(code: str, dimension: str, segment: str, field: str) -> float:
        return float(segment_summary.loc[(segment_summary["area_code"].eq(code)) & (segment_summary["dimension"].eq(dimension)) & (segment_summary["segment"].eq(segment)) & (segment_summary["year"].eq(2025)), field].iloc[0])
    def annual_value(code: str, field: str) -> float:
        return float(annual_change.loc[annual_change["area_code"].eq(code), field].iloc[0])
    leeum_60_share = segment_value("3110091", "age", "60대 이상", "sales_share")
    samcheong_60_share = segment_value("3120005", "age", "60대 이상", "sales_share")
    leeum_evening_share = segment_value("3110091", "time", "17–21시", "sales_share")
    samcheong_lunch_share = segment_value("3120005", "time", "11–14시", "sales_share")
    leeum_evening_ticket = segment_value("3110091", "time", "17–21시", "average_ticket")
    samcheong_lunch_ticket = segment_value("3120005", "time", "11–14시", "average_ticket")
    write_text(REPORT_PATH, f"""# 리움미술관·삼청동 비교 분석

## 범위와 주의

비교 대상은 발달상권 **리움미술관(3110091)**과 **삼청동(3120005)**이다. 2022–2025년 전 업종을 합산해 비교했다. 유동인구는 분기별 관측값의 연평균이며, 매출·거래는 연간 합계다. 건당 매출은 `매출액 ÷ 거래건수`다. 유동인구 1명당 매출이나 거래를 구매전환율로 해석하지 않는다.

‘문화시설·분위기·점포 종류가 비슷하다’는 것은 이 데이터로 직접 측정하거나 검증한 사실이 아니다. 아래 비교는 같은 상권분석서비스의 관측지표가 유사한 변화 양상을 보이는지를 보는 기술 분석이다.

## 2025년 지수 (2022=100)

{markdown_table(annual_2025, list(annual_2025.columns), 10)}

## 2022→2025 변화율

{markdown_table(annual_change_report, list(annual_change_report.columns), 10)}

## 데이터에서 보이는 비교 결과

- **공통점:** 두 상권 모두 유동인구는 늘었지만 매출·거래는 감소했다. 그러나 감소 폭은 리움이 더 크다. 리움의 매출·거래 변화는 각각 {annual_value('3110091', 'sales_amount_change_2022_2025'):.1%}, {annual_value('3110091', 'transactions_change_2022_2025'):.1%}이고, 삼청동은 {annual_value('3120005', 'sales_amount_change_2022_2025'):.1%}, {annual_value('3120005', 'transactions_change_2022_2025'):.1%}다.
- **건당 매출:** 리움은 {annual_value('3110091', 'average_ticket_change_2022_2025'):.1%}, 삼청동은 {annual_value('3120005', 'average_ticket_change_2022_2025'):.1%} 상승했다. 거래 감소를 건당 매출 상승이 일부 완충한 공통 패턴이지만, 리움의 거래 감소가 훨씬 커 총매출 감소도 더 크다.
- **성별:** 2025년 매출에서 여성 비중은 리움 {segment_value('3110091', 'sex', '여성', 'sales_share'):.1%}, 삼청동 {segment_value('3120005', 'sex', '여성', 'sales_share'):.1%}로 둘 다 여성 비중이 높다. 이것은 카드매출 구성의 차이이며 방문객 성별을 뜻하지는 않는다.
- **연령:** 리움은 20·30대 매출 비중이 각각 {segment_value('3110091', 'age', '20대', 'sales_share'):.1%}, {segment_value('3110091', 'age', '30대', 'sales_share'):.1%}이고 60대 이상은 {leeum_60_share:.1%}다. 삼청동은 30·40·50대 비중이 각각 {segment_value('3120005', 'age', '30대', 'sales_share'):.1%}, {segment_value('3120005', 'age', '40대', 'sales_share'):.1%}, {segment_value('3120005', 'age', '50대', 'sales_share'):.1%}이며 60대 이상은 {samcheong_60_share:.1%}다.
- **시간대:** 리움의 최대 매출 시간대는 17–21시(비중 {leeum_evening_share:.1%}, 건당 매출 {leeum_evening_ticket:,.0f}원)이고, 삼청동은 11–14시(비중 {samcheong_lunch_share:.1%}, 건당 매출 {samcheong_lunch_ticket:,.0f}원)다. 두 상권의 시간대 소비 패턴은 같지 않다.

## 점포 구성: 2025년 평균 점포 수 상위 5개 업종

{markdown_table(top_mix, ['area_code', 'industry_name', 'mean_store_count', 'store_share_2025'], 10)}

## 해석 경계

두 상권의 지수·성별·연령대·시간대별 매출 구성 차이는 관측할 수 있다. 그러나 상권 경계 면적, 방문 목적, 관광객 출발지, 임대료, 개별 점포·문화시설 이용자를 연결하는 자료가 없으므로, 유사한 변화가 발견돼도 공통 원인이나 문화시설 효과로 판정하지 않는다.

## 산출물

- `quarterly_overall_metrics.csv`: 분기 총매출·거래·건당 매출·유동인구
- `annual_comparison.csv`: 연간 지수
- `annual_change_2022_2025.csv`: 2022→2025 변화율
- `demographic_time_metrics.csv`: 성별·연령대·시간대별 매출·거래·건당 매출·유동인구
- `segment_summary_2022_2025.csv`: 성별·연령대·시간대별 2022·2025 구성비와 건당 매출
- `industry_mix_2025.csv`: 2025년 평균 점포 구성
""")
    return {"areas": len(AREAS), "quarters": len(overall), "sales_rows": len(sales), "population_rows": len(population)}


if __name__ == "__main__":
    print(run())
