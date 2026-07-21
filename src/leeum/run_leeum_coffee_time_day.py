"""Visualize Leeum coffee-and-beverage sales by time band and weekday."""
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


TARGET_CODE = "3110091"
INDUSTRY_CODE = "CS100010"
INDUSTRY_NAME = "커피-음료"
YEARS = (2022, 2023, 2024, 2025)
OUT_DIR = OUTPUT_DIR / "leeum" / "coffee"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "leeum" / "15_leeum_coffee_time_day.md"
for directory in (OUT_DIR, FIGURE_DIR, REPORT_PATH.parent):
    directory.mkdir(parents=True, exist_ok=True)

TIME = {
    "00–06시": ("시간대_00~06_매출_금액", "시간대_건수~06_매출_건수"),
    "06–11시": ("시간대_06~11_매출_금액", "시간대_건수~11_매출_건수"),
    "11–14시": ("시간대_11~14_매출_금액", "시간대_건수~14_매출_건수"),
    "14–17시": ("시간대_14~17_매출_금액", "시간대_건수~17_매출_건수"),
    "17–21시": ("시간대_17~21_매출_금액", "시간대_건수~21_매출_건수"),
    "21–24시": ("시간대_21~24_매출_금액", "시간대_건수~24_매출_건수"),
}
DAY = {
    "월": ("월요일_매출_금액", "월요일_매출_건수"), "화": ("화요일_매출_금액", "화요일_매출_건수"),
    "수": ("수요일_매출_금액", "수요일_매출_건수"), "목": ("목요일_매출_금액", "목요일_매출_건수"),
    "금": ("금요일_매출_금액", "금요일_매출_건수"), "토": ("토요일_매출_금액", "토요일_매출_건수"),
    "일": ("일요일_매출_금액", "일요일_매출_건수"),
}


def _font() -> str | None:
    for path in (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/NanumGothic.ttf")):
        if path.exists():
            return font_manager.FontProperties(fname=str(path)).get_name()
    return None


def _load() -> pd.DataFrame:
    needed = ["기준_년분기_코드", "상권_코드", "서비스_업종_코드", "당월_매출_금액", "당월_매출_건수"]
    needed += [column for columns in [*TIME.values(), *DAY.values()] for column in columns]
    parts: list[pd.DataFrame] = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        encoding = detect_encoding(path)
        header = pd.read_csv(path, encoding=encoding, nrows=0).columns.tolist()
        if not {"당월_매출_금액", "서비스_업종_코드"}.issubset(header):
            continue
        for piece in pd.read_csv(path, encoding=encoding, usecols=needed, chunksize=100_000, low_memory=False):
            piece["상권_코드"] = piece["상권_코드"].astype(str).str.replace(".0", "", regex=False)
            piece["서비스_업종_코드"] = piece["서비스_업종_코드"].astype(str).str.strip()
            piece["기준_년분기_코드"] = piece["기준_년분기_코드"].astype(str).str.replace(".0", "", regex=False)
            piece = piece.loc[(piece["상권_코드"].eq(TARGET_CODE)) & (piece["서비스_업종_코드"].eq(INDUSTRY_CODE)) & piece["기준_년분기_코드"].str[:4].isin([str(year) for year in YEARS])]
            if not piece.empty:
                parts.append(piece)
    data = pd.concat(parts, ignore_index=True).drop_duplicates(["기준_년분기_코드", "상권_코드", "서비스_업종_코드"], keep="first")
    return data.rename(columns={"기준_년분기_코드": "quarter"})


def _annual_long(data: pd.DataFrame, mapping: dict[str, tuple[str, str]], dimension: str) -> pd.DataFrame:
    data = data.assign(year=data["quarter"].str[:4].astype(int))
    rows: list[pd.DataFrame] = []
    for segment, (sales_col, transaction_col) in mapping.items():
        frame = data.groupby("year", as_index=False).agg(sales_amount=(sales_col, "sum"), transactions=(transaction_col, "sum"))
        frame["segment"] = segment
        frame["dimension"] = dimension
        frame["average_ticket"] = frame["sales_amount"] / frame["transactions"].replace(0, np.nan)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _plot(time: pd.DataFrame, day: pd.DataFrame) -> None:
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for axis, data, metric, title, scale in [
        (axes[0, 0], time, "sales_amount", "시간대별 매출액", 100_000_000),
        (axes[0, 1], time, "transactions", "시간대별 거래건수", 1_000),
        (axes[1, 0], day, "sales_amount", "요일별 매출액", 100_000_000),
        (axes[1, 1], day, "transactions", "요일별 거래건수", 1_000),
    ]:
        segments = data["segment"].drop_duplicates().tolist()
        x = np.arange(len(segments))
        width = .19
        for index, year in enumerate(YEARS):
            frame = data.loc[data["year"].eq(year)].set_index("segment").reindex(segments)
            axis.bar(x + (index - 1.5) * width, frame[metric] / scale, width, label=str(year))
        axis.set_title(title)
        axis.set_xticks(x, segments)
        axis.set_ylabel("억 원" if metric == "sales_amount" else "천 건")
        axis.grid(axis="y", alpha=.2)
    axes[0, 0].legend(title="연도")
    fig.suptitle("리움미술관 커피·음료 업종: 시간대·요일별 매출과 거래", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "leeum_coffee_time_day_sales_transactions_2022_2025.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run() -> dict[str, object]:
    data = _load()
    time, day = _annual_long(data, TIME, "time"), _annual_long(data, DAY, "day")
    save_csv(time, OUT_DIR / "annual_time_metrics_2022_2025.csv")
    save_csv(day, OUT_DIR / "annual_day_metrics_2022_2025.csv")
    _plot(time, day)
    last_time = time.loc[time["year"].eq(2025)]
    last_day = day.loc[day["year"].eq(2025)]
    peak_time = last_time.loc[last_time["sales_amount"].idxmax()]
    peak_day = last_day.loc[last_day["sales_amount"].idxmax()]
    write_text(REPORT_PATH, f"""# 리움미술관 커피·음료: 시간대·요일별 매출 분석

2022–2025년 각 연도의 4개 분기 매출을 합산했다. 대상은 상권 코드 `{TARGET_CODE}`의 `커피-음료` 업종(`{INDUSTRY_CODE}`)이다. 각 시간대·요일 값은 해당 연도 내 합계이며, 시간대·요일별 합계가 총매출·총거래와 약간 다를 수 있는지 여부는 원자료의 비공개·반올림 처리와 별도로 확인해야 한다.

2025년 매출액 기준 최대 시간대는 **{peak_time['segment']}**({peak_time['sales_amount'] / 100_000_000:.2f}억 원), 최대 요일은 **{peak_day['segment']}요일**({peak_day['sales_amount'] / 100_000_000:.2f}억 원)이다.

## 시간대별 연간 합계

{markdown_table(time, ['year', 'segment', 'sales_amount', 'transactions', 'average_ticket'], 30)}

## 요일별 연간 합계

{markdown_table(day, ['year', 'segment', 'sales_amount', 'transactions', 'average_ticket'], 35)}
""")
    return {"quarters": len(data), "time_rows": len(time), "day_rows": len(day)}


if __name__ == "__main__":
    print(run())
