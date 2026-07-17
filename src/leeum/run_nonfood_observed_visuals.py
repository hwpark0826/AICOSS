"""Plot Leeum non-food service-industry series without imputing missing quarters."""
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

from ..config import OUTPUT_DIR, RAW_DIR


TARGET_CODE = "3110091"
QUARTERS = tuple(f"{year}{quarter}" for year in range(2022, 2026) for quarter in range(1, 5))
TARGET_INDUSTRIES = {"CS300011": "일반의류", "CS300022": "화장품"}
OUT_DIR = OUTPUT_DIR / "leeum"
FIGURE_PATH = OUT_DIR / "figures" / "nonfood_observed_trends_2022_2025.png"
EVENT_FIGURE_PATH = OUT_DIR / "figures" / "nonfood_sales_store_events_2022_2025.png"
TABLE_PATH = OUT_DIR / "nonfood_observed_quarterly_metrics_2022_2025.csv"


def _code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".0", "", regex=False).str.strip()


def _font() -> str | None:
    for path in (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/NanumGothic.ttf")):
        if path.exists():
            return font_manager.FontProperties(fname=str(path)).get_name()
    return None


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    sales_parts: list[pd.DataFrame] = []
    store_parts: list[pd.DataFrame] = []
    for path in RAW_DIR.glob("*.csv"):
        try:
            header = pd.read_csv(path, encoding="cp949", nrows=0)
        except UnicodeDecodeError:
            continue
        if len(header.columns) == 55:
            piece = pd.read_csv(path, encoding="cp949", usecols=[0, 3, 5, 6, 7, 8], low_memory=False)
            piece.columns = ["quarter", "area_code", "industry_code", "industry_name", "sales_amount", "transactions"]
            sales_parts.append(piece)
        elif len(header.columns) == 14:
            # The 2021–2024 files place opening/closing counts at positions
            # 10 and 12, while the consolidated later file uses 11 and 13.
            opening_index, closing_index = (10, 12) if "개업_율" in str(header.columns[9]) else (11, 13)
            piece = pd.read_csv(path, encoding="cp949", usecols=[0, 3, 5, 6, 7, opening_index, closing_index], low_memory=False)
            piece.columns = ["quarter", "area_code", "industry_code", "industry_name", "store_count", "open_count", "close_count"]
            store_parts.append(piece)

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for column in ("quarter", "area_code", "industry_code"):
            frame[column] = _code(frame[column])
        return frame.loc[
            frame["area_code"].eq(TARGET_CODE)
            & frame["quarter"].isin(QUARTERS)
            & frame["industry_code"].isin(TARGET_INDUSTRIES)
        ]

    # Annual source files overlap the consolidated source.  Values are exact
    # duplicates for this target, so one copy is retained without aggregation.
    sales = select(pd.concat(sales_parts, ignore_index=True)).drop_duplicates(
        ["quarter", "area_code", "industry_code"], keep="first"
    )
    stores = select(pd.concat(store_parts, ignore_index=True)).drop_duplicates(
        ["quarter", "area_code", "industry_code"], keep="first"
    )
    return sales, stores


def build_series() -> pd.DataFrame:
    sales, stores = _load_raw()
    grid = pd.MultiIndex.from_product(
        [QUARTERS, TARGET_INDUSTRIES], names=["quarter", "industry_code"]
    ).to_frame(index=False)
    merged = grid.merge(
        sales[["quarter", "industry_code", "industry_name", "sales_amount", "transactions"]],
        on=["quarter", "industry_code"], how="left",
    ).merge(
        stores[["quarter", "industry_code", "store_count", "open_count", "close_count"]],
        on=["quarter", "industry_code"], how="left",
    )
    merged["industry_name"] = merged["industry_name"].fillna(merged["industry_code"].map(TARGET_INDUSTRIES))
    merged["observed_sales_and_store"] = merged[["sales_amount", "transactions", "store_count"]].notna().all(axis=1)
    merged["quarter_label"] = merged["quarter"].str[:4] + "Q" + merged["quarter"].str[-1]
    return merged


def plot(series: pd.DataFrame) -> None:
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False

    metrics = [
        ("sales_amount", "분기 매출액 (억원)", 100_000_000),
        ("transactions", "분기 거래건수", 1),
        ("store_count", "점포 수", 1),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 7.6), sharex=True)
    x = np.arange(len(QUARTERS))
    tick_labels = [f"{quarter[:4]}\nQ{quarter[-1]}" for quarter in QUARTERS]

    for row, (code, name) in enumerate(TARGET_INDUSTRIES.items()):
        data = series.loc[series["industry_code"].eq(code)].set_index("quarter").reindex(QUARTERS)
        for col, (metric, label, scale) in enumerate(metrics):
            axis = axes[row, col]
            values = data[metric].astype(float) / scale
            # NaN values deliberately break the line; no interpolation occurs.
            axis.plot(x, values, marker="o", linewidth=2, color="#2563eb")
            missing = values.isna().to_numpy()
            for index in np.flatnonzero(missing):
                axis.axvline(index, color="#dc2626", linestyle=":", linewidth=1)
                axis.text(index, 0.98, "결측", color="#b91c1c", ha="center", va="top", transform=axis.get_xaxis_transform(), fontsize=9)
            axis.set_title(f"{name} · {label}")
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xticks(x, tick_labels, fontsize=8)
            if metric == "sales_amount":
                axis.yaxis.set_major_formatter(lambda value, _: f"{value:,.1f}")

    figure.suptitle("리움미술관 상권: 일반의류·화장품 관측 분기 추이 (2022Q1–2025Q4)", fontsize=15, y=1.02)
    figure.text(0.5, -0.01, "빨간 점선은 원자료 결측 분기이며, 선을 연결하거나 값을 보간하지 않았습니다.", ha="center", fontsize=10)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_sales_and_store_events(series: pd.DataFrame) -> None:
    """Compare sales volatility with observed store openings and closings."""
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False

    figure, axes = plt.subplots(2, 1, figsize=(15, 7.8), sharex=True)
    x = np.arange(len(QUARTERS))
    tick_labels = [f"{quarter[:4]}\nQ{quarter[-1]}" for quarter in QUARTERS]
    for axis, (code, name) in zip(axes, TARGET_INDUSTRIES.items()):
        data = series.loc[series["industry_code"].eq(code)].set_index("quarter").reindex(QUARTERS)
        sales_billion = data["sales_amount"].astype(float) / 100_000_000
        stores = data["store_count"].astype(float)
        axis.plot(x, sales_billion, color="#2563eb", marker="o", linewidth=2, label="매출액")
        axis.set_ylabel("매출액 (억원)", color="#2563eb")
        axis.tick_params(axis="y", labelcolor="#2563eb")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top"]].set_visible(False)
        axis.set_title(name)

        secondary = axis.twinx()
        secondary.plot(x, stores, color="#475569", marker="s", linestyle="--", linewidth=1.5, label="점포 수")
        secondary.set_ylabel("점포 수", color="#475569")
        secondary.tick_params(axis="y", labelcolor="#475569")
        secondary.spines[["top"]].set_visible(False)

        for index, row in enumerate(data.itertuples()):
            if pd.notna(row.sales_amount) and row.open_count:
                axis.annotate(f"개업 +{int(row.open_count)}", (index, row.sales_amount / 100_000_000),
                              xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8, color="#15803d")
            if pd.notna(row.sales_amount) and row.close_count:
                axis.annotate(f"폐업 -{int(row.close_count)}", (index, row.sales_amount / 100_000_000),
                              xytext=(0, -16), textcoords="offset points", ha="center", fontsize=8, color="#b91c1c")
            if pd.isna(row.sales_amount):
                axis.axvline(index, color="#dc2626", linestyle=":", linewidth=1)
                axis.text(index, 0.94, "매출·거래 결측", color="#b91c1c", ha="center", va="top",
                          transform=axis.get_xaxis_transform(), fontsize=8)

        handles, labels = [], []
        for item in (axis, secondary):
            h, l = item.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        axis.legend(handles, labels, loc="upper left", frameon=False)

    axes[-1].set_xticks(x, tick_labels, fontsize=8)
    figure.suptitle("리움미술관 상권: 매출 급변과 점포 개·폐업의 시점 비교", fontsize=15, y=1.01)
    figure.text(0.5, -0.01, "점포 수와 개·폐업은 업종 합계이며, 개별 매장명·주소는 원자료에 없습니다.", ha="center", fontsize=10)
    EVENT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(EVENT_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    series = build_series()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series.to_csv(TABLE_PATH, index=False, encoding="utf-8-sig")
    plot(series)
    plot_sales_and_store_events(series)
    print(FIGURE_PATH)
    print(EVENT_FIGURE_PATH)
    print(TABLE_PATH)


if __name__ == "__main__":
    main()
