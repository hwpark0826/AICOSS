"""Describe Leeum Museum commercial-area floating-population patterns."""
from __future__ import annotations

import os
from pathlib import Path

_MPL_CACHE = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd

from ..config import OUTPUT_DIR, RAW_DIR, REPORT_DIR
from ..export_results import markdown_table, save_csv, write_text
from ..load_data import detect_encoding


TARGET_CODE = "3110091"
YEARS = (2022, 2023, 2024, 2025)
OUT_DIR = OUTPUT_DIR / "leeum" / "floating_population"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "leeum" / "14_leeum_floating_population.md"
for directory in (OUT_DIR, FIGURE_DIR, REPORT_PATH.parent):
    directory.mkdir(parents=True, exist_ok=True)

AGE_COLUMNS = {
    "10대": "연령대_10_유동인구_수", "20대": "연령대_20_유동인구_수", "30대": "연령대_30_유동인구_수",
    "40대": "연령대_40_유동인구_수", "50대": "연령대_50_유동인구_수", "60대 이상": "연령대_60_이상_유동인구_수",
}
TIME_COLUMNS = {
    "00–06시": "시간대_00_06_유동인구_수", "06–11시": "시간대_06_11_유동인구_수", "11–14시": "시간대_11_14_유동인구_수",
    "14–17시": "시간대_14_17_유동인구_수", "17–21시": "시간대_17_21_유동인구_수", "21–24시": "시간대_21_24_유동인구_수",
}
DAY_COLUMNS = {
    "월": "월요일_유동인구_수", "화": "화요일_유동인구_수", "수": "수요일_유동인구_수", "목": "목요일_유동인구_수",
    "금": "금요일_유동인구_수", "토": "토요일_유동인구_수", "일": "일요일_유동인구_수",
}


def _font() -> str | None:
    for path in (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/NanumGothic.ttf")):
        if path.exists():
            return font_manager.FontProperties(fname=str(path)).get_name()
    return None


def _load() -> pd.DataFrame:
    columns = ["기준_년분기_코드", "상권_코드", "상권_코드_명", "총_유동인구_수", *AGE_COLUMNS.values(), *TIME_COLUMNS.values(), *DAY_COLUMNS.values()]
    parts: list[pd.DataFrame] = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        encoding = detect_encoding(path)
        header = pd.read_csv(path, encoding=encoding, nrows=0).columns.tolist()
        if "총_유동인구_수" not in header:
            continue
        piece = pd.read_csv(path, encoding=encoding, usecols=columns, low_memory=False)
        piece["상권_코드"] = piece["상권_코드"].astype(str).str.replace(".0", "", regex=False)
        piece["기준_년분기_코드"] = piece["기준_년분기_코드"].astype(str).str.replace(".0", "", regex=False)
        parts.append(piece.loc[(piece["상권_코드"].eq(TARGET_CODE)) & piece["기준_년분기_코드"].str[:4].isin([str(year) for year in YEARS])])
    data = pd.concat(parts, ignore_index=True).drop_duplicates(["기준_년분기_코드", "상권_코드"], keep="first")
    return data.rename(columns={"기준_년분기_코드": "quarter", "상권_코드": "area_code", "상권_코드_명": "area_name", "총_유동인구_수": "total_population"})


def _annual_long(data: pd.DataFrame, mapping: dict[str, str], dimension: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    data = data.assign(year=data["quarter"].str[:4].astype(int))
    for segment, column in mapping.items():
        piece = data.groupby("year", as_index=False)[column].mean().rename(columns={column: "floating_population"})
        piece["segment"] = segment
        piece["dimension"] = dimension
        rows.append(piece)
    result = pd.concat(rows, ignore_index=True)
    result["index_2022_100"] = result.groupby("segment")["floating_population"].transform(lambda values: values / values.iloc[0] * 100)
    result["change_2022_2025"] = result["floating_population"] / result.groupby("segment")["floating_population"].transform("first") - 1
    return result


def _plot(age: pd.DataFrame, time: pd.DataFrame, day: pd.DataFrame) -> None:
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    for segment, frame in age.groupby("segment", sort=False):
        axes[0].plot(frame["year"], frame["index_2022_100"], marker="o", label=segment)
    axes[0].axhline(100, color="grey", linestyle="--", linewidth=.8)
    axes[0].set_title("연령대별 유동인구 지수")
    axes[0].set_ylabel("2022=100")
    axes[0].set_xticks(YEARS)
    axes[0].legend(ncol=2, fontsize=9)
    lunch_evening = time.loc[time["segment"].isin(["11–14시", "17–21시"])]
    for segment, frame in lunch_evening.groupby("segment"):
        axes[1].plot(frame["year"], frame["floating_population"], marker="o", linewidth=2, label=segment)
    axes[1].set_title("점심·저녁 시간대 유동인구")
    axes[1].set_ylabel("분기 평균 유동인구")
    axes[1].set_xticks(YEARS)
    axes[1].legend()
    latest = day.loc[day["year"].eq(2025)].copy()
    axes[2].bar(latest["segment"], latest["floating_population"], color="#4575b4")
    axes[2].set_title("요일별 유동인구 (2025년 분기 평균)")
    axes[2].set_ylabel("유동인구")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("리움미술관 상권 유동인구: 2022–2025", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "leeum_floating_population_age_time_day.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run() -> dict[str, object]:
    data = _load()
    age, time, day = _annual_long(data, AGE_COLUMNS, "age"), _annual_long(data, TIME_COLUMNS, "time"), _annual_long(data, DAY_COLUMNS, "day")
    total = data.assign(year=data["quarter"].str[:4].astype(int)).groupby("year", as_index=False).agg(total_floating_population=("total_population", "mean"))
    total["change_2022_2025"] = total["total_floating_population"] / total["total_floating_population"].iloc[0] - 1
    save_csv(total, OUT_DIR / "annual_total_floating_population.csv")
    save_csv(age, OUT_DIR / "annual_age_floating_population.csv")
    save_csv(time, OUT_DIR / "annual_time_floating_population.csv")
    save_csv(day, OUT_DIR / "annual_day_floating_population.csv")
    _plot(age, time, day)
    age_2025 = age.loc[age["year"].eq(2025), ["segment", "floating_population", "change_2022_2025"]]
    lunch = time.loc[(time["segment"].eq("11–14시")) & time["year"].eq(2025), "floating_population"].iloc[0]
    evening = time.loc[(time["segment"].eq("17–21시")) & time["year"].eq(2025), "floating_population"].iloc[0]
    latest_day = day.loc[day["year"].eq(2025)]
    write_text(REPORT_PATH, f"""# 리움미술관 상권 유동인구 분석

## 범위

서울시 상권분석서비스의 길단위 유동인구를 사용했다. 2022–2025년은 각 연도 4개 분기가 존재하므로, 연도 값은 **분기 평균 유동인구**다. 유동인구는 방문 목적·거주지·실제 구매자를 식별하지 않으므로, 매출 또는 거래와의 단순 비교를 구매전환율로 해석하지 않는다.

## 전체 추세

{markdown_table(total, list(total.columns), 10)}

전체 유동인구는 2022년 대비 2025년에 **{total.iloc[-1]['change_2022_2025']:.1%} 증가**했다.

## 연령대별 변화

{markdown_table(age_2025, list(age_2025.columns), 10)}

2025년 20대와 30대 유동인구는 각각 2022년보다 **{age_2025.loc[age_2025['segment'].eq('20대'), 'change_2022_2025'].iloc[0]:.1%}**, **{age_2025.loc[age_2025['segment'].eq('30대'), 'change_2022_2025'].iloc[0]:.1%} 증가**했다. 같은 기간 이들의 거래건수 감소와는 방향이 다르므로, 유동인구 감소가 20·30대 거래 감소를 직접 설명한다는 해석은 자료와 맞지 않는다.

## 점심·저녁 및 요일

2025년 분기 평균 유동인구는 점심(11–14시) **{lunch:,.0f}명**, 저녁(17–21시) **{evening:,.0f}명**으로 저녁이 약 **{evening / lunch - 1:.1%}** 많다.

요일별로는 2025년 토요일이 **{latest_day.loc[latest_day['segment'].eq('토'), 'floating_population'].iloc[0]:,.0f}명**으로 가장 많고, 월요일이 **{latest_day.loc[latest_day['segment'].eq('월'), 'floating_population'].iloc[0]:,.0f}명**으로 가장 적다. 이는 각 요일의 분기 평균이며, 특정 이벤트나 특정 점포의 효과를 뜻하지 않는다.

## 산출물

- `annual_total_floating_population.csv`: 전체 연간 평균
- `annual_age_floating_population.csv`: 연령대별 연간 평균·지수·변화율
- `annual_time_floating_population.csv`: 시간대별 연간 평균·지수·변화율
- `annual_day_floating_population.csv`: 요일별 연간 평균·지수·변화율
""")
    return {"quarters": len(data), "total_change_2022_2025": float(total.iloc[-1]["change_2022_2025"])}


if __name__ == "__main__":
    print(run())
