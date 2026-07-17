"""Compare Leeum with the three nearest non-overlapping commercial areas."""
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
import shapefile

from ..config import OUTPUT_DIR, RAW_DIR


TARGET_CODE = "3110091"
YEARS = (2022, 2023, 2024, 2025)
QUARTERS = tuple(f"{year}{quarter}" for year in YEARS for quarter in range(1, 5))
INDUSTRIES = {"CS300011": "일반의류", "CS300022": "화장품"}
LEEUM_DIR = OUTPUT_DIR / "leeum"
SCOPE_PATH = LEEUM_DIR / "nearest3_independent_apparel_cosmetics_scope.csv"
SERIES_PATH = LEEUM_DIR / "nearest3_independent_apparel_cosmetics_quarterly.csv"
FIGURE_PATH = LEEUM_DIR / "figures" / "nearest3_independent_apparel_cosmetics_sales_transactions.png"


def _code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".0", "", regex=False).str.strip()


def _font() -> str | None:
    for path in (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/NanumGothic.ttf")):
        if path.exists():
            return font_manager.FontProperties(fname=str(path)).get_name()
    return None


def _contains(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        crosses = (y2 > y) != (y1 > y)
        if y1 != y2:
            intersection_x = (x1 - x2) * (y - y2) / (y1 - y2) + x2
            inside ^= crosses & (x < intersection_x)
        x1, y1 = x2, y2
    return inside


def _overlap_area_estimate(first: np.ndarray, second: np.ndarray, resolution: int = 300) -> float:
    min_x, max_x = min(first[:, 0].min(), second[:, 0].min()), max(first[:, 0].max(), second[:, 0].max())
    min_y, max_y = min(first[:, 1].min(), second[:, 1].min()), max(first[:, 1].max(), second[:, 1].max())
    x = np.linspace(min_x, max_x, resolution, endpoint=False) + (max_x - min_x) / resolution / 2
    y = np.linspace(min_y, max_y, resolution, endpoint=False) + (max_y - min_y) / resolution / 2
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return float((_contains(points, first) & _contains(points, second)).sum() * (max_x - min_x) * (max_y - min_y) / resolution**2)


def _nearest_independent_areas() -> pd.DataFrame:
    candidates = pd.read_csv(LEEUM_DIR / "neighbor_areas.csv", dtype={"area_code": str}).sort_values("centroid_distance_m")
    reader = shapefile.Reader(str(next(RAW_DIR.glob("*.shp"))), encoding="utf-8")
    polygons: dict[str, np.ndarray] = {}
    for record, shape in zip(reader.iterRecords(), reader.iterShapes()):
        code = str(record.as_dict()["TRDAR_CD"])
        polygons[code] = np.asarray(shape.points, dtype=float)
    selected: list[str] = []
    overlaps: list[dict[str, float]] = []
    for row in candidates.itertuples():
        code = str(row.area_code)
        if code not in polygons:
            continue
        checked = [TARGET_CODE, *selected]
        measures = {other: _overlap_area_estimate(polygons[code], polygons[other]) for other in checked}
        if any(value > 0 for value in measures.values()):
            continue
        selected.append(code)
        overlaps.append(measures)
        if len(selected) == 3:
            break
    result = candidates.loc[candidates["area_code"].astype(str).isin(selected)].copy()
    result["area_code"] = result["area_code"].astype(str)
    result["selection_rank"] = result["area_code"].map({code: index + 1 for index, code in enumerate(selected)})
    result["overlap_with_leeum_est_m2"] = result["area_code"].map({code: values[TARGET_CODE] for code, values in zip(selected, overlaps)})
    return result.sort_values("selection_rank")[[
        "selection_rank", "area_code", "area_name", "area_type", "district", "administrative_dong",
        "centroid_distance_m", "overlap_with_leeum_est_m2",
    ]]


def _load_sales(area_codes: set[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in RAW_DIR.glob("*.csv"):
        try:
            header = pd.read_csv(path, encoding="cp949", nrows=0)
        except UnicodeDecodeError:
            continue
        if len(header.columns) != 55:
            continue
        for chunk in pd.read_csv(path, encoding="cp949", usecols=[0, 3, 4, 5, 6, 7, 8], chunksize=100_000, low_memory=False):
            chunk.columns = ["quarter", "area_code", "area_name", "industry_code", "industry_name", "sales_amount", "transactions"]
            for column in ("quarter", "area_code", "industry_code"):
                chunk[column] = _code(chunk[column])
            chunk = chunk.loc[
                chunk["area_code"].isin(area_codes)
                & chunk["industry_code"].isin(INDUSTRIES)
                & chunk["quarter"].isin(QUARTERS)
            ]
            if not chunk.empty:
                pieces.append(chunk)
    return pd.concat(pieces, ignore_index=True).drop_duplicates(["quarter", "area_code", "industry_code"], keep="first")


def build_series(scope: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    areas = pd.concat([
        pd.DataFrame([{
            "selection_rank": 0, "area_code": TARGET_CODE, "area_name": "리움미술관", "area_type": "target",
            "district": "용산구", "administrative_dong": "한남동", "centroid_distance_m": 0.0,
            "overlap_with_leeum_est_m2": 0.0,
        }]),
        scope,
    ], ignore_index=True)
    raw = _load_sales(set(areas["area_code"]))
    grid = pd.MultiIndex.from_product([QUARTERS, areas["area_code"], INDUSTRIES], names=["quarter", "area_code", "industry_code"]).to_frame(index=False)
    series = grid.merge(areas, on="area_code", how="left").merge(
        raw[["quarter", "area_code", "industry_code", "industry_name", "sales_amount", "transactions"]],
        on=["quarter", "area_code", "industry_code"], how="left",
    )
    series["industry_name"] = series["industry_name"].fillna(series["industry_code"].map(INDUSTRIES))
    series["observed"] = series[["sales_amount", "transactions"]].notna().all(axis=1)
    series["quarter_label"] = series["quarter"].str[:4] + "Q" + series["quarter"].str[-1]
    coverage = series.groupby(["area_code", "area_name", "industry_code", "industry_name"], as_index=False).agg(observed_quarters=("observed", "sum"))
    scope = scope.merge(coverage.pivot(index="area_code", columns="industry_name", values="observed_quarters").reset_index(), on="area_code", how="left")
    return series, scope


def plot(series: pd.DataFrame) -> None:
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    x = np.arange(len(QUARTERS))
    labels = [f"{quarter[:4]}\nQ{quarter[-1]}" for quarter in QUARTERS]
    areas = series.drop_duplicates("area_code").sort_values("selection_rank")
    colors = ["#dc2626", "#2563eb", "#16a34a", "#ea580c"]
    color_map = dict(zip(areas["area_code"], colors))
    name_map = areas.set_index("area_code")["area_name"].to_dict()
    for row, (industry_code, industry_name) in enumerate(INDUSTRIES.items()):
        for col, (metric, title, scale) in enumerate([
            ("sales_amount", "분기 매출액 (억원)", 100_000_000),
            ("transactions", "분기 거래건수 (천 건)", 1_000),
        ]):
            axis = axes[row, col]
            for area_code in areas["area_code"]:
                data = series.loc[
                    series["industry_code"].eq(industry_code) & series["area_code"].eq(area_code)
                ].set_index("quarter").reindex(QUARTERS)
                axis.plot(x, data[metric].astype(float) / scale, marker="o", linewidth=2 if area_code == TARGET_CODE else 1.5,
                          label=name_map[area_code], color=color_map[area_code])
            axis.set_title(f"{industry_name} · {title}")
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xticks(x, labels, fontsize=8)
            if row == 0 and col == 1:
                axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.suptitle("리움미술관과 폴리곤이 중첩되지 않는 가장 가까운 3개 상권", fontsize=15, y=1.02)
    figure.text(0.5, -0.01, "현재 제공된 폴리곤 기준으로 리움 및 상호 간 중첩이 0㎡인 상권만 선택했습니다. 결측은 보간하지 않아 선이 끊깁니다.", ha="center", fontsize=10)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    scope = _nearest_independent_areas()
    series, scope = build_series(scope)
    scope.to_csv(SCOPE_PATH, index=False, encoding="utf-8-sig")
    series.to_csv(SERIES_PATH, index=False, encoding="utf-8-sig")
    plot(series)
    print(SCOPE_PATH)
    print(SERIES_PATH)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
