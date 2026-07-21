"""Render two readable all-industry revenue-index charts for Leeum."""
from __future__ import annotations

import os
from pathlib import Path

_MPL_CACHE = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd

from ..config import OUTPUT_DIR


OUT_DIR = OUTPUT_DIR / "leeum" / "all_industry_neighbor_index"
FIGURE_DIR = OUT_DIR / "figures"
ANNUAL_PATH = OUT_DIR / "all_industry_neighbor_annual_metrics.csv"
GROUP_PATH = OUT_DIR / "all_industry_neighbor_equal_weight_index.csv"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _font() -> str | None:
    for path in (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/NanumGothic.ttf")):
        if path.exists():
            return font_manager.FontProperties(fname=str(path)).get_name()
    return None


def _base_axes(ax) -> None:
    ax.set_facecolor("#F7F9FC")
    ax.axhline(100, color="#8A9099", linewidth=1.2, zorder=0)
    ax.set_ylim(50, 150)
    ax.set_xlim(2020.8, 2025.45)
    ax.set_xticks([2021, 2022, 2023, 2024, 2025])
    ax.set_ylabel("매출지수 (2022=100)")
    ax.grid(axis="y", color="#D9DEE7", linewidth=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AEB6C2")
    ax.tick_params(colors="#3A4656")


def run() -> dict[str, str]:
    font = _font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    annual = pd.read_csv(ANNUAL_PATH, dtype={"area_code": str})
    groups = pd.read_csv(GROUP_PATH)
    leeum = annual.loc[annual["group"].eq("leeum")].sort_values("year")
    neighbors = groups.loc[groups["group"].eq("independent_neighbors_equal_weight")].sort_values("year")

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    _base_axes(ax)
    color = "#2459A6"
    ax.fill_between(leeum["year"], leeum["sales_index_2022_100"], 100, color=color, alpha=.10)
    ax.plot(leeum["year"], leeum["sales_index_2022_100"], marker="o", markersize=9, color=color, linewidth=3.2)
    ax.set_title("리움미술관 상권 | 전체 업종 연간 매출지수", loc="center", pad=18, fontweight="bold")
    ax.text(2024.72, 142, "기준선", color="#6A7280", fontsize=11, ha="right")
    ax.annotate(
        "2025\n73.2", xy=(2025, leeum.iloc[-1]["sales_index_2022_100"]), xytext=(2024.72, 65),
        ha="right", va="center", fontsize=19, fontweight="bold", color=color,
        bbox={"boxstyle": "round,pad=.45", "facecolor": "white", "edgecolor": color, "linewidth": 1.2},
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.4},
    )
    ax.text(2021, 54, "2022년 매출=100", color="#6A7280", fontsize=11, ha="left")
    fig.tight_layout()
    leeum_path = FIGURE_DIR / "leeum_all_industry_revenue_index.png"
    fig.savefig(leeum_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    _base_axes(ax)
    leeum_color, neighbor_color = "#2459A6", "#D66A45"
    ax.plot(leeum["year"], leeum["sales_index_2022_100"], marker="o", markersize=9, color=leeum_color, linewidth=3.2, label="리움미술관")
    ax.plot(neighbors["year"], neighbors["equal_weight_mean_index"], marker="s", markersize=9, color=neighbor_color, linewidth=3.2, label="인접 5개 상권 평균")
    last_leeum = leeum.iloc[-1]
    last_neighbor = neighbors.iloc[-1]
    ax.annotate(
        f"리움미술관\n{last_leeum['sales_index_2022_100']:.1f}", xy=(last_leeum["year"], last_leeum["sales_index_2022_100"]), xytext=(2024.72, 65),
        ha="right", va="center", fontsize=17, fontweight="bold", color=leeum_color,
        bbox={"boxstyle": "round,pad=.42", "facecolor": "white", "edgecolor": leeum_color, "linewidth": 1.2},
        arrowprops={"arrowstyle": "-", "color": leeum_color, "linewidth": 1.4},
    )
    ax.annotate(
        f"인접 5개 상권 평균\n{last_neighbor['equal_weight_mean_index']:.1f}", xy=(last_neighbor["year"], last_neighbor["equal_weight_mean_index"]), xytext=(2024.72, 124),
        ha="right", va="center", fontsize=16, fontweight="bold", color=neighbor_color,
        bbox={"boxstyle": "round,pad=.42", "facecolor": "white", "edgecolor": neighbor_color, "linewidth": 1.2},
        arrowprops={"arrowstyle": "-", "color": neighbor_color, "linewidth": 1.4},
    )
    ax.set_title("리움미술관과 인접 상권 | 전체 업종 연간 매출지수", loc="center", pad=18, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.text(2021, 54, "2022년 매출=100", color="#6A7280", fontsize=11, ha="left")
    fig.tight_layout()
    comparison_path = FIGURE_DIR / "leeum_vs_independent_neighbors_all_industry_index.png"
    fig.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"leeum": str(leeum_path), "comparison": str(comparison_path)}


if __name__ == "__main__":
    print(run())
