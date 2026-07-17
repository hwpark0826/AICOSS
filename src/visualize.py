"""Small dependency-light PNG visualizations for the final outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _canvas(title: str, width: int = 1500, height: int = 900) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 35), title, fill="#14213d", font=_font(30, True))
    return image, draw


def _bar_chart(data: pd.DataFrame, label: str, value: str, title: str, output: Path, color: str = "#d1495b") -> None:
    subset = data.sort_values(value, ascending=False).head(20).iloc[::-1].copy()
    image, draw = _canvas(title)
    left, top, right, bottom = 390, 115, 1420, 840
    maximum = max(float(subset[value].abs().max()), 1e-9)
    row_height = (bottom - top) / max(len(subset), 1)
    draw.line((left, top, left, bottom), fill="#6b7280", width=2)
    for index, (_, row) in enumerate(subset.iterrows()):
        y = top + index * row_height + 6
        amount = float(row[value])
        width = (right - left) * abs(amount) / maximum
        draw.rectangle((left, y, left + width, y + row_height - 12), fill=color)
        text = str(row[label])[:33]
        draw.text((45, y), text, fill="#1f2937", font=_font(16))
        draw.text((left + width + 8, y), f"{amount:,.2f}", fill="#1f2937", font=_font(15))
    draw.text((left, 855), "점수가 클수록 서울 동일 업종 대비 쇠퇴 신호가 강함", fill="#4b5563", font=_font(15))
    image.save(output)


def create_figures(ranking: pd.DataFrame, stability: pd.DataFrame, correlations: pd.DataFrame, figure_dir: Path) -> None:
    """Write the requested five transparent, dependency-light PNG figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    named = ranking.copy()
    _bar_chart(named, "area_name", "CoreDeclineScore", "쇠퇴 신호 상위 20개 상권", figure_dir / "top20_decline_score.png")
    components = named.sort_values("CoreDeclineScore", ascending=False).head(20).copy()
    component_columns = ["long_score", "medium_score", "recent_score"]
    rows = []
    for _, row in components.iterrows():
        for column in component_columns:
            rows.append({"area_name": row["area_name"], "component": column.replace("_score", ""), "value": row.get(column, np.nan)})
    comp = pd.DataFrame(rows)
    # Present a concise total component comparison by retaining each area's long score.
    _bar_chart(comp.loc[comp["component"].eq("long")], "area_name", "value", "상위 20개 상권: 장기 구간 점수", figure_dir / "top20_score_components.png", "#457b9d")
    merged = ranking[["area_code", "area_name", "overall_rank"]].merge(stability, on="area_code", how="left")
    _bar_chart(merged, "area_name", "top20_appearance_rate", "민감도 시나리오별 상위 20위 출현 비율", figure_dir / "rank_stability.png", "#2a9d8f")
    period = ranking[["area_name", "long_score", "medium_score", "recent_score"]].melt(id_vars="area_name", var_name="period", value_name="value")
    _bar_chart(period.loc[period["period"].eq("long_score")], "area_name", "value", "장기 구간 순위 비교용 점수", figure_dir / "period_rank_comparison.png", "#7b2cbf")
    high = correlations.loc[correlations["metric_a"].eq(correlations["metric_b"]) == False].copy()
    if high.empty:
        high = pd.DataFrame({"metric_a": ["상관계수 없음"], "spearman_rho": [0.0]})
    _bar_chart(high, "metric_a", "spearman_rho", "지표 간 Spearman 상관계수", figure_dir / "metric_correlation.png", "#f4a261")
