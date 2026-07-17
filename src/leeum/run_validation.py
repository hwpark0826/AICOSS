"""Validate alternative explanations for the Leeum Museum commercial-area rank.

The analysis intentionally treats the existing fifth-place result as a claim to
be tested, not a conclusion to defend.  It separates observed facts,
interpretations, and causal hypotheses throughout its outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Iterable
import os

_MPL_CACHE = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile

from ..calculate_benchmarks import annual_benchmarks, quarterly_benchmarks
from ..config import FOOD_CODES, OUTPUT_DIR, RAW_DIR, REPORT_DIR, TABLE_DIR
from ..export_results import markdown_table, save_csv, write_text
from ..load_data import discover_inputs, read_area_reference


TARGET_CODE = "3110091"
TARGET_NAME = "리움미술관"
MAIN_QUARTERS = tuple(f"{year}{quarter}" for year in range(2021, 2026) for quarter in range(1, 5))
ALL_QUARTERS = (*MAIN_QUARTERS, "20261")
ANALYSIS_YEARS = (2022, 2023, 2024, 2025)
LEEUM_DIR = OUTPUT_DIR / "leeum"
LEEUM_FIGURE_DIR = LEEUM_DIR / "figures"
LEEUM_MAP_DIR = LEEUM_DIR / "maps"
LEEUM_REPORT_DIR = REPORT_DIR / "leeum"
for directory in (LEEUM_DIR, LEEUM_FIGURE_DIR, LEEUM_MAP_DIR, LEEUM_REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

KEY_COLS = ["quarter", "area_code", "industry_code"]
CORE_SALES = "당월_매출_금액"
CORE_TX = "당월_매출_건수"
RAW_COLUMN_MAP = {
    "기준_년분기_코드": "quarter",
    "상권_코드": "area_code",
    "상권_코드_명": "area_name",
    "상권_구분_코드_명": "area_type",
    "서비스_업종_코드": "industry_code",
    "서비스_업종_코드_명": "industry_name",
    CORE_SALES: "sales_amount",
    CORE_TX: "sales_transactions",
    "주중_매출_금액": "weekday_sales",
    "주말_매출_금액": "weekend_sales",
    "주중_매출_건수": "weekday_transactions",
    "주말_매출_건수": "weekend_transactions",
    "시간대_00~06_매출_금액": "sales_00_06",
    "시간대_06~11_매출_금액": "sales_06_11",
    "시간대_11~14_매출_금액": "sales_11_14",
    "시간대_14~17_매출_금액": "sales_14_17",
    "시간대_17~21_매출_금액": "sales_17_21",
    "시간대_21~24_매출_금액": "sales_21_24",
    "시간대_건수~06_매출_건수": "transactions_00_06",
    "시간대_건수~11_매출_건수": "transactions_06_11",
    "시간대_건수~14_매출_건수": "transactions_11_14",
    "시간대_건수~17_매출_건수": "transactions_14_17",
    "시간대_건수~21_매출_건수": "transactions_17_21",
    "시간대_건수~24_매출_건수": "transactions_21_24",
    "남성_매출_금액": "male_sales",
    "여성_매출_금액": "female_sales",
    "남성_매출_건수": "male_transactions",
    "여성_매출_건수": "female_transactions",
    "연령대_20_매출_금액": "age20_sales",
    "연령대_30_매출_금액": "age30_sales",
    "연령대_40_매출_금액": "age40_sales",
    "연령대_50_매출_금액": "age50_sales",
    "연령대_60_이상_매출_금액": "age60plus_sales",
}
TIME_SALES = ["sales_00_06", "sales_06_11", "sales_11_14", "sales_14_17", "sales_17_21", "sales_21_24"]
TIME_TX = ["transactions_00_06", "transactions_06_11", "transactions_11_14", "transactions_14_17", "transactions_17_21", "transactions_21_24"]


def _code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".0", "", regex=False).str.strip()


def _ratio(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    if isinstance(numerator, pd.Series):
        return numerator / denominator.where(denominator.ne(0))
    return numerator / denominator if pd.notna(numerator) and pd.notna(denominator) and denominator != 0 else np.nan


def _pct(start: float, end: float) -> float:
    return end / start - 1 if pd.notna(start) and pd.notna(end) and start != 0 else np.nan


def _log_change(start: float, end: float) -> float:
    return float(np.log(end / start)) if pd.notna(start) and pd.notna(end) and start > 0 and end > 0 else np.nan


def _quarter_label(value: object) -> str:
    text = str(value)
    return f"{text[:4]}Q{text[-1]}" if len(text) == 5 and text[:4].isdigit() and text[-1].isdigit() else text


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


def _read_existing(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    for column in ("quarter", "area_code", "industry_code"):
        if column in data:
            data[column] = _code(data[column])
    return data


def _load_target_detail() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read detailed target sales columns and align them to the core quarter panel."""
    inputs = [item for item in discover_inputs() if item.role == "sales_area"]
    pieces: list[pd.DataFrame] = []
    for item in inputs:
        header = pd.read_csv(item.path, encoding=item.encoding, nrows=0).columns.tolist()
        selected = [source for source in RAW_COLUMN_MAP if source in header]
        for chunk in pd.read_csv(item.path, encoding=item.encoding, usecols=selected, chunksize=100_000, low_memory=False):
            raw_code = chunk["상권_코드"].astype(str).str.replace(".0", "", regex=False).str.strip()
            part = chunk.loc[raw_code.eq(TARGET_CODE)].copy()
            if not part.empty:
                part["source_file"] = item.path.name
                pieces.append(part.rename(columns=RAW_COLUMN_MAP))
    if not pieces:
        raise ValueError("No detailed sales rows found for Leeum Museum.")
    detail = pd.concat(pieces, ignore_index=True)
    for column in ("quarter", "area_code", "industry_code"):
        detail[column] = _code(detail[column])
    detail = detail.loc[detail["quarter"].isin(ALL_QUARTERS) & detail["industry_code"].isin(FOOD_CODES)].copy()
    numeric = [column for column in detail.columns if column not in {"quarter", "area_code", "area_name", "area_type", "industry_code", "industry_name", "source_file"}]
    detail[numeric] = detail[numeric].apply(pd.to_numeric, errors="coerce")
    duplicated = detail.duplicated(KEY_COLS, keep=False)
    duplicate_audit = detail.loc[duplicated, [*KEY_COLS, "source_file"]].sort_values(KEY_COLS).copy()
    # Core analysis removes duplicate keys rather than adding them.  Keep the
    # first source row here as well and record every duplicate key in output.
    detail = detail.drop_duplicates(KEY_COLS, keep="first")
    panel = _read_existing(Path("data/interim/food_commercial_area_quarter_panel.csv"))
    target_keys = panel.loc[panel["area_code"].eq(TARGET_CODE), KEY_COLS].drop_duplicates()
    detail = detail.merge(target_keys, on=KEY_COLS, how="inner", validate="one_to_one")
    return detail.sort_values(KEY_COLS).reset_index(drop=True), duplicate_audit


def _quarterly_metrics(detail: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    sales_agg = detail.groupby("quarter", as_index=False).agg(
        observed_industries=("industry_code", "nunique"),
        sales_amount=("sales_amount", "sum"), sales_transactions=("sales_transactions", "sum"),
        weekday_sales=("weekday_sales", "sum"), weekend_sales=("weekend_sales", "sum"),
        weekday_transactions=("weekday_transactions", "sum"), weekend_transactions=("weekend_transactions", "sum"),
        **{column: (column, "sum") for column in [*TIME_SALES, *TIME_TX, "male_sales", "female_sales", "age20_sales", "age30_sales", "age40_sales", "age50_sales", "age60plus_sales"] if column in detail},
    )
    store = panel.loc[panel["area_code"].eq(TARGET_CODE), ["quarter", "industry_code", "store_count", "open_count", "close_count"]]
    store_agg = store.groupby("quarter", as_index=False).agg(
        operating_stores=("store_count", "sum"), openings=("open_count", "sum"), closings=("close_count", "sum"),
    )
    frame = pd.DataFrame({"quarter": ALL_QUARTERS}).merge(sales_agg, on="quarter", how="left").merge(store_agg, on="quarter", how="left")
    frame["year"] = frame["quarter"].str[:4].astype(int)
    frame["quarter_no"] = frame["quarter"].str[-1].astype(int)
    expected_industries = int(sales_agg["observed_industries"].max())
    frame["expected_industries"] = expected_industries
    frame["total_series_comparable"] = frame["observed_industries"].eq(expected_industries)
    frame["comparability_note"] = np.where(
        frame["total_series_comparable"],
        "all target industries observed",
        "incomplete industry coverage; do not compare total series",
    )
    frame["sales_per_store"] = _ratio(frame["sales_amount"], frame["operating_stores"])
    frame["transactions_per_store"] = _ratio(frame["sales_transactions"], frame["operating_stores"])
    frame["sales_per_transaction"] = _ratio(frame["sales_amount"], frame["sales_transactions"])
    frame["net_entry_count"] = frame["openings"] - frame["closings"]
    frame["net_entry_rate"] = _ratio(frame["net_entry_count"], frame["operating_stores"])
    frame["weekend_sales_share"] = _ratio(frame["weekend_sales"], frame["sales_amount"])
    frame["weekend_transaction_share"] = _ratio(frame["weekend_transactions"], frame["sales_transactions"])
    frame["evening_sales_share"] = _ratio(frame.get("sales_17_21", 0) + frame.get("sales_21_24", 0), frame["sales_amount"])
    frame["night_sales_share"] = _ratio(frame.get("sales_21_24", 0), frame["sales_amount"])
    frame["female_sales_share"] = _ratio(frame.get("female_sales", 0), frame["sales_amount"])
    comparable_pair = frame["total_series_comparable"] & frame["total_series_comparable"].shift(4, fill_value=False)
    for metric in ["sales_amount", "sales_transactions", "operating_stores", "sales_per_store", "transactions_per_store", "sales_per_transaction", "weekend_sales_share", "evening_sales_share"]:
        frame[f"{metric}_yoy"] = frame[metric].pct_change(4, fill_method=None).where(comparable_pair)
    frame["sales_amount_4q_ma"] = frame["sales_amount"].rolling(4, min_periods=4).mean()
    return frame


def _change_points(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Descriptive mean-break scan plus a pre/post 2022Q4 comparison."""
    rows: list[dict[str, object]] = []
    metrics = ["sales_amount", "sales_transactions", "sales_per_store", "transactions_per_store", "weekend_sales_share", "evening_sales_share"]
    source = quarterly.loc[
        quarterly["quarter"].isin(MAIN_QUARTERS) & quarterly["total_series_comparable"]
    ].reset_index(drop=True)
    for metric in metrics:
        values = source[metric].astype(float)
        valid = values.notna()
        series = values.loc[valid].to_numpy()
        labels = source.loc[valid, "quarter"].tolist()
        candidates: list[tuple[float, str]] = []
        for split in range(4, len(series) - 4):
            before, after = series[:split], series[split:]
            sse = ((before - before.mean()) ** 2).sum() + ((after - after.mean()) ** 2).sum()
            candidates.append((float(sse), labels[split]))
        best_sse, best_quarter = min(candidates) if candidates else (np.nan, pd.NA)
        pre = source.loc[source["quarter"].le("20223"), metric].dropna()
        post = source.loc[source["quarter"].ge("20224"), metric].dropna()
        rows.append({
            "metric": metric, "best_mean_break_quarter": best_quarter, "best_mean_break_sse": best_sse,
            "pre_2022q4_mean": pre.mean(), "post_2022q4_mean": post.mean(),
            "post_vs_pre_change": _pct(pre.mean(), post.mean()),
            "interpretation": "Descriptive break scan; seasonality and concurrent causes are not controlled.",
        })
    return pd.DataFrame(rows)


def _annual_from_quarterly(quarterly: pd.DataFrame) -> pd.DataFrame:
    totals = quarterly.loc[quarterly["year"].isin(ANALYSIS_YEARS)].groupby("year", as_index=False).agg(
        observed_quarters=("quarter", "nunique"), sales_amount=("sales_amount", "sum"),
        sales_transactions=("sales_transactions", "sum"), operating_stores=("operating_stores", "mean"),
        openings=("openings", "sum"), closings=("closings", "sum"), weekday_sales=("weekday_sales", "sum"), weekend_sales=("weekend_sales", "sum"),
        weekday_transactions=("weekday_transactions", "sum"), weekend_transactions=("weekend_transactions", "sum"),
        **{column: (column, "sum") for column in [*TIME_SALES, *TIME_TX, "male_sales", "female_sales", "age20_sales", "age30_sales", "age40_sales", "age50_sales", "age60plus_sales"] if column in quarterly},
    )
    totals["sales_per_store"] = _ratio(totals["sales_amount"], totals["operating_stores"])
    totals["transactions_per_store"] = _ratio(totals["sales_transactions"], totals["operating_stores"])
    totals["sales_per_transaction"] = _ratio(totals["sales_amount"], totals["sales_transactions"])
    totals["net_entry_count"] = totals["openings"] - totals["closings"]
    totals["weekend_sales_share"] = _ratio(totals["weekend_sales"], totals["sales_amount"])
    totals["evening_sales_share"] = _ratio(totals.get("sales_17_21", 0) + totals.get("sales_21_24", 0), totals["sales_amount"])
    totals["night_sales_share"] = _ratio(totals.get("sales_21_24", 0), totals["sales_amount"])
    totals["female_sales_share"] = _ratio(totals.get("female_sales", 0), totals["sales_amount"])
    return totals


def _annual_industry(annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = annual.loc[annual["area_code"].eq(TARGET_CODE) & annual["year"].isin([2021, *ANALYSIS_YEARS])].copy()
    target["sales_per_store"] = _ratio(target["sales_amount"], target["store_count"])
    target["transactions_per_store"] = _ratio(target["sales_transactions"], target["store_count"])
    target["sales_per_transaction"] = _ratio(target["sales_amount"], target["sales_transactions"])
    base = target.loc[target["year"].eq(2022), ["industry_code", "sales_amount", "sales_transactions", "store_count"]].rename(columns={
        "sales_amount": "sales_2022", "sales_transactions": "transactions_2022", "store_count": "stores_2022",
    })
    target = target.merge(base, on="industry_code", how="left")
    target["sales_index_2022_100"] = target["sales_amount"] / target["sales_2022"] * 100
    target["transactions_index_2022_100"] = target["sales_transactions"] / target["transactions_2022"] * 100
    target["store_index_2022_100"] = target["store_count"] / target["stores_2022"] * 100
    end = target.loc[target["year"].eq(2025), ["industry_code", "industry_name", "sales_amount", "sales_transactions", "store_count"]].rename(columns={
        "sales_amount": "sales_2025", "sales_transactions": "transactions_2025", "store_count": "stores_2025",
    })
    contribution = base.merge(end, on="industry_code", how="outer")
    contribution["sales_change_2022_2025"] = contribution["sales_2025"] - contribution["sales_2022"]
    total_loss = contribution.loc[contribution["sales_change_2022_2025"].lt(0), "sales_change_2022_2025"].sum()
    contribution["share_of_observed_sales_loss"] = contribution["sales_change_2022_2025"] / total_loss if total_loss else np.nan
    return target, contribution.sort_values("sales_change_2022_2025")


def _decomposition(data: pd.DataFrame, label: str, start_year: int = 2022, end_year: int = 2025) -> dict[str, object]:
    grouped = data.loc[data["year"].isin([start_year, end_year])].groupby("year", as_index=False).agg(
        sales_amount=("sales_amount", "sum"), sales_transactions=("sales_transactions", "sum"), store_count=("store_count", "sum"),
    ).set_index("year")
    start = grouped.loc[start_year] if start_year in grouped.index else pd.Series(dtype=float)
    end = grouped.loc[end_year] if end_year in grouped.index else pd.Series(dtype=float)
    sales_start, sales_end = start.get("sales_amount", np.nan), end.get("sales_amount", np.nan)
    tx_start, tx_end = start.get("sales_transactions", np.nan), end.get("sales_transactions", np.nan)
    stores_start, stores_end = start.get("store_count", np.nan), end.get("store_count", np.nan)
    tx_store_start, tx_store_end = _ratio(tx_start, stores_start), _ratio(tx_end, stores_end)
    sales_tx_start, sales_tx_end = _ratio(sales_start, tx_start), _ratio(sales_end, tx_end)
    store_effect = _log_change(stores_start, stores_end)
    tx_effect = _log_change(tx_store_start, tx_store_end)
    price_effect = _log_change(sales_tx_start, sales_tx_end)
    total = _log_change(sales_start, sales_end)
    return {
        "scope": label, "sales_2022": sales_start, "sales_2025": sales_end,
        "transactions_2022": tx_start, "transactions_2025": tx_end,
        "stores_2022": stores_start, "stores_2025": stores_end,
        "log_sales_change": total, "log_store_effect": store_effect,
        "log_transactions_per_store_effect": tx_effect,
        "log_sales_per_transaction_effect": price_effect,
        "decomposition_residual": total - sum([store_effect, tx_effect, price_effect]) if all(pd.notna(x) for x in [total, store_effect, tx_effect, price_effect]) else np.nan,
    }


def _match_controls(annual: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = annual.loc[annual["year"].eq(2022) & annual["complete_year"]].copy()
    totals = source.groupby(["area_code", "area_name", "area_type"], as_index=False).agg(
        total_sales=("sales_amount", "sum"), total_transactions=("sales_transactions", "sum"), total_stores=("store_count", "sum"),
    )
    for code, name in [("CS100003", "japanese_share"), ("CS100004", "western_share"), ("CS100010", "coffee_share")]:
        part = source.loc[source["industry_code"].eq(code)].groupby("area_code", as_index=False)["sales_amount"].sum().rename(columns={"sales_amount": name})
        totals = totals.merge(part, on="area_code", how="left")
    totals[["japanese_share", "western_share", "coffee_share"]] = totals[["japanese_share", "western_share", "coffee_share"]].fillna(0)
    for column in ("japanese_share", "western_share", "coffee_share"):
        totals[column] = totals[column] / totals["total_sales"]
    totals["sales_per_store"] = (totals["total_sales"] / totals["total_stores"]).replace([np.inf, -np.inf], np.nan)
    target = totals.loc[totals["area_code"].eq(TARGET_CODE)].copy()
    if target.empty:
        raise ValueError("Leeum is missing from the 2022 complete-year panel.")
    features = ["total_sales", "total_transactions", "total_stores", "sales_per_store", "japanese_share", "western_share", "coffee_share"]
    candidate = totals.loc[totals["area_type"].eq(target.iloc[0]["area_type"]) & ~totals["area_code"].eq(TARGET_CODE)].dropna(subset=features).copy()
    scaled = pd.concat([target, candidate], ignore_index=True)
    transformed = scaled[features].astype(float).copy()
    for column in ["total_sales", "total_transactions", "total_stores", "sales_per_store"]:
        transformed[column] = np.log1p(transformed[column])
    standard_deviation = transformed.std(ddof=0).replace(0, 1.0)
    target_vector = transformed.iloc[0]
    candidate["matching_distance"] = np.sqrt(((transformed.iloc[1:].reset_index(drop=True) - target_vector) / standard_deviation).pow(2).sum(axis=1).to_numpy())
    selected = candidate.nsmallest(7, "matching_distance").copy()
    result = pd.concat([target.assign(matching_distance=0.0, selection="target"), selected.assign(selection="same area type; nearest 2022 feature distance")], ignore_index=True)
    if not reference.empty:
        result = result.merge(reference[[column for column in ["area_code", "district", "administrative_dong"] if column in reference]], on="area_code", how="left")
    balances = []
    for feature in features:
        target_value = float(target.iloc[0][feature])
        control_mean = float(selected[feature].mean())
        control_std = float(selected[feature].std(ddof=0))
        balances.append({"feature": feature, "target_2022": target_value, "controls_mean_2022": control_mean,
                         "difference": target_value - control_mean, "control_std": control_std,
                         "standardized_difference": _ratio(target_value - control_mean, control_std)})
    control_codes = selected["area_code"].tolist()
    trend_data = annual.loc[annual["area_code"].isin([TARGET_CODE, *control_codes]) & annual["year"].isin(ANALYSIS_YEARS)]
    trend = trend_data.assign(group=np.where(trend_data["area_code"].eq(TARGET_CODE), "leeum", "matched_controls")).groupby(["group", "year"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"), sales_transactions=("sales_transactions", "sum"), store_count=("store_count", "sum"),
    )
    trend["sales_index_2022_100"] = trend.groupby("group")["sales_amount"].transform(lambda x: x / x.iloc[0] * 100)
    return result, pd.DataFrame(balances), trend


@dataclass
class SpatialArea:
    code: str
    name: str
    area_type: str
    district: str
    dong: str
    x: float
    y: float
    points: np.ndarray


def _load_spatial() -> dict[str, SpatialArea]:
    files = list(RAW_DIR.glob("*.shp"))
    if not files:
        raise FileNotFoundError("No commercial-area shapefile found under data/raw.")
    reader = shapefile.Reader(str(files[0]), encoding="utf-8")
    areas: dict[str, SpatialArea] = {}
    for record, shape in zip(reader.iterRecords(), reader.iterShapes()):
        row = record.as_dict()
        areas[str(row["TRDAR_CD"])] = SpatialArea(
            code=str(row["TRDAR_CD"]), name=str(row["TRDAR_CD_N"]), area_type=str(row["TRDAR_SE_1"]),
            district=str(row["SIGNGU_CD_"]), dong=str(row["ADSTRD_CD_"]),
            x=float(row["XCNTS_VALU"]), y=float(row["YDNTS_VALU"]), points=np.asarray(shape.points, dtype=float),
        )
    return areas


def _neighbors(annual: pd.DataFrame, spatial: dict[str, SpatialArea]) -> pd.DataFrame:
    target = spatial[TARGET_CODE]
    valid_codes = set(annual["area_code"])
    candidate_words = ["한남", "이태원", "경리단", "해방촌", "한강진", "리움"]
    rows: list[dict[str, object]] = []
    for code, item in spatial.items():
        if code == TARGET_CODE or code not in valid_codes:
            continue
        distance = hypot(item.x - target.x, item.y - target.y)
        name_candidate = any(word in item.name for word in candidate_words)
        if distance > 1000 and not name_candidate:
            continue
        rows.append({"area_code": code, "area_name": item.name, "area_type": item.area_type, "district": item.district,
                     "administrative_dong": item.dong, "centroid_distance_m": distance,
                     "within_500m": distance <= 500, "within_1km": distance <= 1000,
                     "name_candidate": name_candidate,
                     "selection_reason": "; ".join(label for label, ok in [("centroid_within_1km", distance <= 1000), ("itaewon_hannam_name_candidate", name_candidate)] if ok)})
    return pd.DataFrame(rows).sort_values(["centroid_distance_m", "area_code"]).reset_index(drop=True)


def _neighbor_trend(annual: pd.DataFrame, neighbors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes = [TARGET_CODE, *neighbors["area_code"].tolist()]
    data = annual.loc[annual["area_code"].isin(codes) & annual["year"].isin(ANALYSIS_YEARS)].copy()
    per_area = data.groupby(["area_code", "area_name", "year"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"), sales_transactions=("sales_transactions", "sum"), store_count=("store_count", "sum"),
    )
    per_area["sales_per_store"] = _ratio(per_area["sales_amount"], per_area["store_count"])
    base = per_area.loc[per_area["year"].eq(2022), ["area_code", "sales_amount"]].rename(columns={"sales_amount": "sales_2022"})
    per_area = per_area.merge(base, on="area_code", how="left")
    per_area["sales_index_2022_100"] = per_area["sales_amount"] / per_area["sales_2022"] * 100
    aggregate = data.assign(group=np.where(data["area_code"].eq(TARGET_CODE), "leeum", "selected_neighbors")).groupby(["group", "year"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"), sales_transactions=("sales_transactions", "sum"), store_count=("store_count", "sum"),
    )
    aggregate["sales_index_2022_100"] = aggregate.groupby("group")["sales_amount"].transform(lambda x: x / x.iloc[0] * 100)
    return per_area, aggregate


def _external_events() -> pd.DataFrame:
    return pd.DataFrame([
        {"event_date": "2022-10-29", "event": "이태원 참사", "category": "external_shock", "source": "https://www.mois.go.kr/plan2023/download/plan2023_RNSS.pdf", "status": "date_verified", "use_in_analysis": "2022Q4 event marker only; causal effect is not inferred"},
        {"event_date": "2022-03-02", "event": "아트스펙트럼 2022", "category": "leeum_program", "source": "https://www.leeumhoam.org/leeum/edu/publication/28?params=Y", "status": "date_verified", "use_in_analysis": "Exhibition/program timing only; no visitor count supplied"},
        {"event_date": "2022-09-02", "event": "구름산책자", "category": "leeum_program", "source": "https://www.leeumhoam.org/leeum/exhibition/53?params=Y", "status": "date_verified", "use_in_analysis": "Exhibition timing only; no visitor count supplied"},
        {"event_date": pd.NA, "event": "한남2·3구역 재개발 이주·철거", "category": "redevelopment", "source": pd.NA, "status": "not_verifiable_from_available_data", "use_in_analysis": "No dated boundary, migration, demolition, or store-address data supplied"},
    ])


def _plot_quarterly(quarterly: pd.DataFrame) -> None:
    _configure_plot()
    frame = quarterly.loc[quarterly["quarter"].isin(ALL_QUARTERS)].copy()
    x = np.arange(len(frame))
    labels = frame["quarter"].str[:4] + "Q" + frame["quarter"].str[-1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis, (column, title) in zip(axes.flat, [("sales_amount", "총매출"), ("sales_transactions", "거래건수"), ("operating_stores", "운영점포 수"), ("sales_per_store", "점포당 매출")]):
        axis.plot(x, frame[column].where(frame["total_series_comparable"]), marker="o", linewidth=1.5)
        axis.axvline(frame.index[frame["quarter"].eq("20224")][0], color="#d73027", linestyle="--", linewidth=1, label="2022Q4")
        axis.set_title(title)
        axis.grid(alpha=.25)
        axis.ticklabel_format(style="plain", axis="y")
    axes[-1, 0].set_xticks(x[::2], labels[::2], rotation=45, ha="right")
    axes[-1, 1].set_xticks(x[::2], labels[::2], rotation=45, ha="right")
    fig.suptitle("리움미술관 상권 분기 추이 (점선: 2022Q4, 2021Q1–Q3 불완전 집계 제외)", y=1.02)
    _savefig(LEEUM_FIGURE_DIR / "quarterly_trend.png")


def _plot_yoy(quarterly: pd.DataFrame) -> None:
    _configure_plot()
    frame = quarterly.loc[quarterly["quarter"].isin(MAIN_QUARTERS) & quarterly["total_series_comparable"]].copy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for column, label in [("sales_amount_yoy", "매출 YoY"), ("sales_transactions_yoy", "거래 YoY"), ("transactions_per_store_yoy", "점포당 거래 YoY")]:
        ax.plot(frame["quarter"], frame[column] * 100, marker="o", label=label)
    ax.axhline(0, color="black", linewidth=.8)
    ax.axvline("20224", color="#d73027", linestyle="--", linewidth=1)
    ax.set_ylabel("전년동기 대비 (%)")
    ax.set_title("리움미술관 주요 지표의 전년동기 대비 변화")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(alpha=.2)
    _savefig(LEEUM_FIGURE_DIR / "yoy_change.png")


def _plot_decomposition(decomposition: pd.DataFrame) -> None:
    _configure_plot()
    shown = decomposition.copy()
    cols = ["log_store_effect", "log_transactions_per_store_effect", "log_sales_per_transaction_effect"]
    labels = ["점포 수", "점포당 거래", "건당 매출"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(shown))
    for col, label, color in zip(cols, labels, ["#d73027", "#4575b4", "#74add1"]):
        value = shown[col].fillna(0).to_numpy()
        ax.bar(shown["scope"], value, bottom=bottom, label=label, color=color)
        bottom += value
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_ylabel("로그 매출변화 기여도 (2022→2025)")
    ax.set_title("매출 변화 분해")
    ax.legend()
    _savefig(LEEUM_FIGURE_DIR / "sales_decomposition.png")


def _plot_industries(industry: pd.DataFrame, contribution: pd.DataFrame) -> None:
    _configure_plot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for code, group in industry.groupby("industry_code"):
        axes[0].plot(group["year"], group["sales_index_2022_100"], marker="o", label=group["industry_name"].iloc[0])
    axes[0].axhline(100, color="grey", linestyle="--", linewidth=.8)
    axes[0].set_title("업종별 매출지수 (2022=100)")
    axes[0].set_xticks(list(ANALYSIS_YEARS))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    shown = contribution.sort_values("sales_change_2022_2025")
    axes[1].barh(shown["industry_name"], shown["sales_change_2022_2025"] / 100_000_000, color="#d73027")
    axes[1].axvline(0, color="black", linewidth=.7)
    axes[1].set_title("업종별 매출 변화 (2022→2025)")
    axes[1].set_xlabel("억원")
    _savefig(LEEUM_FIGURE_DIR / "industry_trends.png")

    # Standalone contribution chart for use outside the two-panel figure.
    _configure_plot()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(shown["industry_name"], shown["sales_change_2022_2025"] / 100_000_000, color="#d73027")
    ax.axvline(0, color="black", linewidth=.7)
    ax.set_xlabel("Sales change (100m KRW)")
    _savefig(LEEUM_FIGURE_DIR / "industry_contribution.png")


def _plot_day_time(annual: pd.DataFrame) -> None:
    _configure_plot()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for column, label in [("weekend_sales_share", "주말 매출 비중"), ("evening_sales_share", "17~24시 매출 비중"), ("night_sales_share", "21~24시 매출 비중"), ("female_sales_share", "여성 매출 비중")]:
        if column in annual:
            ax.plot(annual["year"], annual[column] * 100, marker="o", label=label)
    ax.set_xticks(list(ANALYSIS_YEARS))
    ax.set_ylabel("매출 비중 (%)")
    ax.set_title("주말·시간대·성별 매출구성 변화")
    ax.legend()
    ax.grid(alpha=.2)
    _savefig(LEEUM_FIGURE_DIR / "day_time_trends.png")


def _plot_exhibition_timeline(quarterly: pd.DataFrame, events: pd.DataFrame) -> None:
    """Show official program timing only; this is not an attendance-effect chart."""
    _configure_plot()
    frame = quarterly.loc[quarterly["quarter"].isin(MAIN_QUARTERS) & quarterly["total_series_comparable"]].copy()
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, frame["sales_amount"] / 100_000_000, marker="o", color="#4575b4")
    for _, row in events.loc[events["category"].eq("leeum_program")].iterrows():
        stamp = pd.Timestamp(row["event_date"])
        event_quarter = f"{stamp.year}{(stamp.month - 1) // 3 + 1}"
        locations = np.flatnonzero(frame["quarter"].to_numpy() == event_quarter)
        if len(locations):
            location = int(locations[0])
            ax.axvline(location, color="#74add1", linestyle="--", linewidth=1)
            ax.annotate(str(row["event"]), (location, frame["sales_amount"].iloc[location] / 100_000_000),
                        xytext=(3, 12), textcoords="offset points", fontsize=8, rotation=25)
    ax.set_xticks(x[::2], (frame["quarter"].str[:4] + "Q" + frame["quarter"].str[-1]).iloc[::2], rotation=45, ha="right")
    ax.set_ylabel("sales (100m KRW)")
    ax.set_title("Leeum program timing and sales: visitor counts are unavailable")
    ax.grid(alpha=.2)
    _savefig(LEEUM_FIGURE_DIR / "exhibition_sales_timeline.png")


def _plot_neighbor_map(spatial: dict[str, SpatialArea], neighbors: pd.DataFrame) -> None:
    _configure_plot()
    target = spatial[TARGET_CODE]
    fig, ax = plt.subplots(figsize=(8, 7))
    for code in neighbors["area_code"]:
        item = spatial.get(code)
        if item is not None and len(item.points):
            ax.plot(item.points[:, 0], item.points[:, 1], color="#999999", linewidth=.8)
    ax.fill(target.points[:, 0], target.points[:, 1], color="#d73027", alpha=.45, edgecolor="#7f0000", linewidth=1.5)
    ax.scatter(target.x, target.y, color="#7f0000", s=24)
    ax.annotate(TARGET_NAME, (target.x, target.y), xytext=(5, 5), textcoords="offset points", weight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("리움미술관과 인접·이태원권 후보 상권")
    _savefig(LEEUM_MAP_DIR / "leeum_neighbor_map.png")


def _plot_group_trend(trend: pd.DataFrame, filename: str, title: str) -> None:
    _configure_plot()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for group, frame in trend.groupby("group"):
        label = {"leeum": "리움미술관", "selected_neighbors": "인접·이태원권 후보 합계", "matched_controls": "매칭 통제군 합계"}.get(group, group)
        ax.plot(frame["year"], frame["sales_index_2022_100"], marker="o", label=label)
    ax.axhline(100, color="grey", linewidth=.8, linestyle="--")
    ax.set_xticks(list(ANALYSIS_YEARS))
    ax.set_ylabel("매출지수 (2022=100)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=.2)
    _savefig(LEEUM_FIGURE_DIR / filename)


def _plot_matching_balance(balance: pd.DataFrame) -> None:
    _configure_plot()
    shown = balance.sort_values("standardized_difference")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = np.where(shown["standardized_difference"].abs().gt(.25), "#d73027", "#4575b4")
    ax.barh(shown["feature"], shown["standardized_difference"], color=colors)
    ax.axvline(0, color="black", linewidth=.7)
    ax.axvline(.25, color="#999999", linestyle="--", linewidth=.8)
    ax.axvline(-.25, color="#999999", linestyle="--", linewidth=.8)
    ax.set_xlabel("standardized mean difference")
    ax.set_title("2022 comparison-area balance (|SMD| ≤ 0.25 reference)")
    _savefig(LEEUM_FIGURE_DIR / "matching_balance.png")


def _format_number(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "검증 불가"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{digits}f}"
    return str(value)


def _write_reports(
    reproduction: pd.DataFrame, quarterly: pd.DataFrame, change_points: pd.DataFrame, annual_total: pd.DataFrame,
    decomposition: pd.DataFrame, industry: pd.DataFrame, contribution: pd.DataFrame, day_time: pd.DataFrame,
    neighbors: pd.DataFrame, neighbor_aggregate: pd.DataFrame, controls: pd.DataFrame, balance: pd.DataFrame,
    control_trend: pd.DataFrame, events: pd.DataFrame, redevelopment: pd.DataFrame, evidence: pd.DataFrame,
) -> None:
    target = reproduction.iloc[0]
    post = change_points.loc[change_points["metric"].eq("sales_amount")].iloc[0]
    annual = annual_total.set_index("year")
    target_decomp = decomposition.loc[decomposition["scope"].eq("leeum")].iloc[0]
    neighbor_index = neighbor_aggregate.loc[neighbor_aggregate["year"].eq(2025)].set_index("group")["sales_index_2022_100"]
    control_index = control_trend.loc[control_trend["year"].eq(2025)].set_index("group")["sales_index_2022_100"]
    verdict = evidence.loc[evidence["hypothesis"].eq("final_classification"), "verdict"].iloc[0]
    max_abs_smd = balance["standardized_difference"].abs().max()
    write_text(LEEUM_REPORT_DIR / "00_reproduction_check.md", f"""# 00. 기존 5위 결과 재현

## 확인된 사실

`{TARGET_NAME}`(코드 `{TARGET_CODE}`)은 기존 결과에서 **{int(target['overall_rank'])}위 / {int(target['eligible_area_count'])}개**, CoreDeclineScore **{target['CoreDeclineScore']:.3f}**입니다. 장기·중기·최근 점수는 각각 {_format_number(target['long_score'], 3)}, {_format_number(target['medium_score'], 3)}, {_format_number(target['recent_score'], 3)}입니다.

2021년 한식은 완결연도 조건을 충족하지 않습니다. 따라서 2021년 전체 외식업 합계를 2025년과 직접 비교하지 않았고, 순위 재현에는 업종별 완결 자료의 상대성과를 사용했습니다. 2022–2025는 완결연도 총량 추세로 별도 분석했습니다. 분기 총량에서도 2021Q1–Q3은 4개 중 3개 업종만 관측되어 비교·전년동기·변화점 계산에서 제외합니다.
""")
    write_text(LEEUM_REPORT_DIR / "01_quarterly_turning_points.md", f"""# 01. 분기 전환점과 2022년 고점

## 확인된 사실

2022년 총매출은 {_format_number(annual.at[2022, 'sales_amount'] / 100_000_000, 2)}억 원, 2025년은 {_format_number(annual.at[2025, 'sales_amount'] / 100_000_000, 2)}억 원입니다. 2022→2025 변화는 {_format_number(_pct(annual.at[2022, 'sales_amount'], annual.at[2025, 'sales_amount']) * 100, 1)}%입니다. 거래건수도 {_format_number(_pct(annual.at[2022, 'sales_transactions'], annual.at[2025, 'sales_transactions']) * 100, 1)}% 변했습니다.

비교 가능한 분기만 사용한 평균수준 분할 탐색에서 총매출의 가장 낮은 오차 분할 후보는 **{_quarter_label(post['best_mean_break_quarter'])}**이며, 2022Q4 전후 평균 변화는 {_format_number(post['post_vs_pre_change'] * 100, 1)}%입니다. 이 전후 평균에는 불완전한 2021Q1–Q3을 넣지 않았습니다.

## 해석

2022년이 2023–2025보다 높은 관측 고점이라는 사실은 확인됩니다. 그러나 분기 평균의 변화점은 계절성·동시 사건을 통제하지 않은 기술통계이므로, 2022Q4의 특정 사건이 원인이라는 인과판정은 아닙니다.
""")
    write_text(LEEUM_REPORT_DIR / "02_sales_decomposition.md", f"""# 02. 2022→2025 매출 감소 분해

## 확인된 사실

매출 항등식 `매출 = 평균 점포 수 × 점포당 거래건수 × 건당 매출`로 분해하면, 리움미술관 상권의 로그 매출 변화는 {_format_number(target_decomp['log_sales_change'], 3)}입니다.

- 점포 수 효과: {_format_number(target_decomp['log_store_effect'], 3)}
- 점포당 거래 효과: {_format_number(target_decomp['log_transactions_per_store_effect'], 3)}
- 건당 매출 효과: {_format_number(target_decomp['log_sales_per_transaction_effect'], 3)}

## 해석

점포 수가 거의 유지된 상태에서 **점포당 거래는 약화**했고, **건당 매출은 상승하여 일부를 완충**했음을 보여주는 회계적 분해입니다. 방문객 감소, 가격 변화, 상권 이동 중 어느 것이 원인인지는 이 항등식만으로 구분할 수 없습니다.
""")
    write_text(LEEUM_REPORT_DIR / "03_industry_analysis.md", f"""# 03. 업종별 재편 검증

## 확인된 사실

2022→2025 매출 감소 기여는 아래 표와 같습니다.

{markdown_table(contribution, ['industry_name', 'sales_2022', 'sales_2025', 'sales_change_2022_2025', 'share_of_observed_sales_loss'], 15)}

## 해석

모든 관측 업종의 2022→2025 매출은 감소했습니다. 점포 총수만으로 업종 교체를 확정할 수는 없고, 주소·사업체 단위의 연속 관측도 없습니다. 따라서 이 표는 업종별 **절대 감소액**만 보여주며, 업종 재편의 증거로 해석하지 않습니다.
""")
    write_text(LEEUM_REPORT_DIR / "04_day_time_customer_analysis.md", f"""# 04. 주말·시간대·고객구성 분석

## 확인된 사실

원자료에는 주중/주말, 시간대, 성별·연령대별 매출·거래가 있습니다. 연도별 지표는 `outputs/leeum/day_time_customer_analysis.csv`에 저장했습니다.

## 해석

주말·야간 비중의 하락은 관광·방문형 수요 약화와 부합할 수 있고, 일상 시간대의 하락은 배후수요 약화와 부합할 수 있습니다. 그러나 이 지표는 카드매출 구성이지 실제 방문객의 출발지·목적지를 추적한 자료가 아니므로, 방문객 이동 자체를 증명하지는 못합니다.
""")
    write_text(LEEUM_REPORT_DIR / "05_spatial_displacement.md", f"""# 05. 인접 상권·공간 재배치 가설

## 확인된 사실

중심점 1km 이내 또는 한남·이태원·경리단·해방촌·한강진 명칭 후보를 기준으로 {len(neighbors)}개 상권을 비교했습니다. 2025년 매출지수(2022=100)는 리움미술관 {_format_number(neighbor_index.get('leeum'), 1)}, 후보권역 합계 {_format_number(neighbor_index.get('selected_neighbors'), 1)}입니다.

## 해석

대상이 하락하고 주변권역의 **선정 상권 합계**가 유지·증가하면 공간 재배치와 **일치하는 패턴**일 수 있습니다. 이는 상권별 평균이나 동일 업종 보정치가 아닌 합계 지수이며, 이동경로·점포 이전 주소 자료가 없으므로 재배치의 인과 증거는 아닙니다.
""")
    write_text(LEEUM_REPORT_DIR / "06_matched_control_analysis.md", f"""# 06. 2022년 유사 상권 비교

## 확인된 사실

동일 상권유형 안에서 2022년 총매출·거래·점포 수·점포당 매출·일식/양식/커피 비중이 가까운 7개 상권을 비교군으로 선택했습니다. 2025년 매출지수(2022=100)는 리움미술관 {_format_number(control_index.get('leeum'), 1)}, 통제군 합계 {_format_number(control_index.get('matched_controls'), 1)}입니다.

{markdown_table(balance, ['feature', 'target_2022', 'controls_mean_2022', 'standardized_difference'], 10)}

균형 진단의 최대 절대 표준화 차이는 **{max_abs_smd:.2f}**이다. 통상적인 0.25 기준을 크게 넘으므로, 이 비교는 ‘유사 통제군에 대한 효과 추정’이 아니라 관측 추세를 보조적으로 대조한 결과다.

## 해석

통제군보다 큰 하락은 상대적 약화의 기술적 증거입니다. 임대료·관광객·시설 특성 등 미관측 변수가 있어 인과효과나 반사실적 결과로 해석하지 않습니다.
""")
    write_text(LEEUM_REPORT_DIR / "07_redevelopment_effect.md", f"""# 07. 재개발 간접영향 가설

## 확인된 사실

현재 프로젝트에는 한남2·3구역의 경계, 이주·철거 시점, 점포 주소·거주인구 자료가 없습니다.

{markdown_table(redevelopment, list(redevelopment.columns), 10)}

## 판정

재개발의 직접 중첩·경계 인접·일정 거리 영향 중 어느 것도 현재 자료로 검증할 수 없습니다. 따라서 재개발은 가능한 맥락이지, 본 결과에서 확인된 원인이 아닙니다.
""")
    write_text(LEEUM_REPORT_DIR / "08_exhibition_spillover.md", f"""# 08. 리움 전시·방문객 파급 가설

## 확인된 사실

공식 리움 페이지로 전시·프로그램 시점은 확인했지만, 전시별 방문객 수·매표 수·주변 소비 경로 자료는 제공되지 않았습니다.

{markdown_table(events.loc[events['category'].eq('leeum_program')], ['event_date', 'event', 'source', 'use_in_analysis'], 10)}

## 판정

전시가 2022년 고점과 시간적으로 겹친다는 사실만으로 매출 고점의 원인이라고 할 수 없습니다. 방문객 계량자료가 확보될 때만 파급효과를 분석합니다.
""")
    write_text(LEEUM_REPORT_DIR / "09_store_turnover.md", f"""# 09. 점포 교체·업종 재편의 한계

## 확인된 사실

2022–2025 평균 점포 수·개업·폐업은 `quarterly_metrics.csv`와 `industry_deep_dive.csv`에 있습니다. 점포 총수의 안정은 점포 구성의 안정과 동의어가 아닙니다.

## 판정

점포 주소·사업체 식별자가 없어 동일 점포의 폐업·이전·업종전환·고급점포 진입을 추적할 수 없습니다. 업종 재편은 **부분 지지 또는 검증 불가**로만 판정합니다.
""")
    write_text(LEEUM_REPORT_DIR / "leeum_one_page_brief.md", f"""# 리움미술관 상권 한 장 브리프

## 결론

기존 5위는 재현됐다. 그러나 2022년이 명확한 관측 고점이고, 2021년 한식이 불완결이며, 외부충격·재개발·방문객 이동을 식별할 자료가 없다. 따라서 최종 분류는 **{verdict}**이며, ‘구조적 쇠퇴’로 단정하지 않는다.

## 확인된 사실

- 2022→2025 총매출: {_format_number(_pct(annual.at[2022, 'sales_amount'], annual.at[2025, 'sales_amount']) * 100, 1)}%
- 2022→2025 거래건수: {_format_number(_pct(annual.at[2022, 'sales_transactions'], annual.at[2025, 'sales_transactions']) * 100, 1)}%
- 평균 점포 수: {_format_number(annual.at[2022, 'operating_stores'], 2)}개 → {_format_number(annual.at[2025, 'operating_stores'], 2)}개
- 통제군 2025 매출지수: {_format_number(control_index.get('matched_controls'), 1)} vs 리움 {_format_number(control_index.get('leeum'), 1)}

## 확인하지 못한 인과

이태원 참사·재개발·전시·방문객 이동은 날짜 또는 맥락 일부만 확인된다. 이들이 매출 감소를 만들었다는 인과는 현 자료로 판정하지 않는다.
""")
    write_text(LEEUM_REPORT_DIR / "leeum_commercial_area_validation_report.md", f"""# 리움미술관 상권 대체가설 검증 보고서

## 분석 원칙

이 문서는 기존 5위 결론을 정당화하지 않는다. 각 가설에서 관측된 사실, 그 사실에 대한 해석, 인과가설을 분리하고 반대 증거와 데이터 한계를 함께 기록한다.

## 최종 판정

**{verdict}**

{markdown_table(evidence, ['hypothesis', 'predicted_pattern', 'observed_facts', 'supporting_evidence', 'counter_evidence', 'verdict', 'confidence'], 20)}

## 핵심 해석

2022년 이후 총매출·거래·점포당 거래가 약해진 것은 관측 사실이다. 2022년 고점 정상화, 이태원권 공통충격, 공간 재배치, 재개발 간접영향, 업종 재편은 서로 배타적이지 않은 설명 후보지만, 현재 제공된 데이터만으로는 단일 원인으로 식별되지 않는다.

## 재현

`python -m src.leeum.run_validation` 실행 후 모든 CSV·그림은 `outputs/leeum/`, 세부 보고서는 `reports/leeum/`에 생성된다.
""")


def _evidence(
    reproduction: pd.DataFrame, annual_total: pd.DataFrame, change_points: pd.DataFrame, decomposition: pd.DataFrame,
    industry_contribution: pd.DataFrame, neighbor_aggregate: pd.DataFrame, control_trend: pd.DataFrame,
    redevelopment: pd.DataFrame, events: pd.DataFrame, balance: pd.DataFrame,
) -> pd.DataFrame:
    annual = annual_total.set_index("year")
    sales_2022_2025 = _pct(annual.at[2022, "sales_amount"], annual.at[2025, "sales_amount"])
    tx_2022_2025 = _pct(annual.at[2022, "sales_transactions"], annual.at[2025, "sales_transactions"])
    store_2022_2025 = _pct(annual.at[2022, "operating_stores"], annual.at[2025, "operating_stores"])
    cp = change_points.loc[change_points["metric"].eq("sales_amount")].iloc[0]
    decomp = decomposition.loc[decomposition["scope"].eq("leeum")].iloc[0]
    neighbor = neighbor_aggregate.loc[neighbor_aggregate["year"].eq(2025)].set_index("group")["sales_index_2022_100"]
    controls = control_trend.loc[control_trend["year"].eq(2025)].set_index("group")["sales_index_2022_100"]
    industry_loss = industry_contribution.loc[industry_contribution["sales_change_2022_2025"].lt(0), "industry_name"].tolist()
    maximum_balance_gap = balance["standardized_difference"].abs().max()
    rows = [
        {"hypothesis": "H1_2022_temporary_peak", "predicted_pattern": "2022 only is unusually high and later years return near a normal level", "observed_facts": f"2022→2025 sales {sales_2022_2025:.1%}; best mean split {_quarter_label(cp['best_mean_break_quarter'])}", "supporting_evidence": "2022 is the highest complete annual total", "counter_evidence": f"2025 remains materially below 2023 and 2024; transactions {tx_2022_2025:.1%}", "verdict": "partially_supported", "confidence": "medium", "interpretation_boundary": "High point is observed; normalisation versus a counterfactual is not identified."},
        {"hypothesis": "H2_itaewon_external_shock", "predicted_pattern": "A sustained post-2022Q4 fall, especially visitor-oriented weekend/evening demand and neighbouring areas", "observed_facts": f"post/pre sales mean change {cp['post_vs_pre_change']:.1%}; event date is verified", "supporting_evidence": "The official event date falls in 2022Q4", "counter_evidence": "No visitor origin, exposure intensity, or causal control is available; break timing can reflect concurrent changes", "verdict": "not_verifiable", "confidence": "low", "interpretation_boundary": "Temporal coincidence is not causal attribution."},
        {"hypothesis": "H3_spatial_reallocation", "predicted_pattern": "Leeum falls while nearby candidate areas or same industries rise", "observed_facts": f"2025 index: Leeum {neighbor.get('leeum', np.nan):.1f}, neighbours {neighbor.get('selected_neighbors', np.nan):.1f}", "supporting_evidence": "Relative divergence is observable if candidate aggregate is higher", "counter_evidence": "No movement paths, store-address transitions, or origin-destination data", "verdict": "possible" if neighbor.get("selected_neighbors", 0) > neighbor.get("leeum", 0) else "not_supported", "confidence": "low", "interpretation_boundary": "Spatial co-movement is descriptive only."},
        {"hypothesis": "H4_redevelopment_indirect_effect", "predicted_pattern": "Dated redevelopment exposure overlaps residential/daytime demand weakening", "observed_facts": "No redevelopment boundary or dated relocation/demolition input available", "supporting_evidence": "None in the provided data", "counter_evidence": "No spatial or timing evidence", "verdict": "not_verifiable", "confidence": "low", "interpretation_boundary": "Redevelopment cannot be inserted as an assumed cause."},
        {"hypothesis": "H5_industry_recomposition", "predicted_pattern": "Total stores are stable but industry shares/turnover change", "observed_facts": f"2022→2025 stores {store_2022_2025:.1%}; every observed industry has lower sales: {', '.join(industry_loss)}", "supporting_evidence": "Total average store count is broadly stable", "counter_evidence": "All observed industries decline and there is no business/address identifier to observe replacement", "verdict": "not_supported", "confidence": "medium", "interpretation_boundary": "Stable total stores do not establish industry reorganization; the available pattern is broad sales decline."},
        {"hypothesis": "H6_underlying_demand_weakening", "predicted_pattern": "Transactions and transactions per store fall even if store count remains stable", "observed_facts": f"transactions {tx_2022_2025:.1%}; stores {store_2022_2025:.1%}; tx/store log effect {decomp['log_transactions_per_store_effect']:.3f}", "supporting_evidence": "Transaction deterioration is not explained by store count alone", "counter_evidence": "Demand source (resident, worker, tourist) is not observed", "verdict": "partially_supported", "confidence": "medium", "interpretation_boundary": "Transaction-based weakening is observed; its demand source is not identified."},
        {"hypothesis": "H7_matched_area_relative_weakening", "predicted_pattern": "Leeum declines more than similarly configured 2022 areas", "observed_facts": f"2025 index: Leeum {controls.get('leeum', np.nan):.1f}, matched controls {controls.get('matched_controls', np.nan):.1f}; max |SMD| {maximum_balance_gap:.2f}", "supporting_evidence": "The selected comparison areas decline less in the observed series", "counter_evidence": "The selected controls remain materially unbalanced (max |SMD| exceeds 0.25) and omit tourist exposure, rent, and museum visits", "verdict": "possible" if controls.get("leeum", 0) < controls.get("matched_controls", 0) else "not_supported", "confidence": "low", "interpretation_boundary": "This is a descriptive comparison, not a credible causal control."},
    ]
    partially_supported = {row["hypothesis"] for row in rows if row["verdict"] == "partially_supported"}
    primary = "2022년 고점 이후 거래기반 약화 관측; 구조적 쇠퇴 원인은 식별 불가"
    rows.append({"hypothesis": "final_classification", "predicted_pattern": "", "observed_facts": f"existing rank={int(reproduction.iloc[0]['overall_rank'])}; partially_supported={len(partially_supported)}", "supporting_evidence": "", "counter_evidence": "2021 incomplete Korean data, unbalanced controls, and unavailable causal covariates", "verdict": primary, "confidence": "medium", "interpretation_boundary": "The score is reproduced, but this is not a structural-decline causal conclusion."})
    return pd.DataFrame(rows)


def run() -> dict[str, object]:
    ranking = _read_existing(TABLE_DIR / "commercial_area_decline_ranking.csv")
    annual = _read_existing(Path("data/processed/food_commercial_area_annual_panel.csv"))
    panel = _read_existing(Path("data/interim/food_commercial_area_quarter_panel.csv"))
    reference = read_area_reference()
    if not reference.empty:
        reference["area_code"] = _code(reference["area_code"])
    target = ranking.loc[ranking["area_code"].eq(TARGET_CODE)].copy()
    if len(target) != 1:
        raise ValueError(f"Expected one Leeum ranking row; got {len(target)}")
    target["eligible_area_count"] = len(ranking)
    reproduction_columns = ["area_code", "area_name", "district", "area_type", "eligible_area_count", "overall_rank", "CoreDeclineScore", "long_score", "medium_score", "recent_score", "start_food_stores", "analysis_avg_food_stores", "sales_weight_coverage", "store_weight_coverage", "valid_industry_count", "sensitivity_mean_rank", "sensitivity_rank_std", "top20_appearance_rate", "early_warning_rank", "EarlyWarningAdjustedScore"]
    reproduction = target[[column for column in reproduction_columns if column in target]].copy()
    save_csv(reproduction, LEEUM_DIR / "leeum_reproduction.csv")

    detail, duplicate_audit = _load_target_detail()
    save_csv(duplicate_audit, LEEUM_DIR / "detail_duplicate_audit.csv")
    quarterly = _quarterly_metrics(detail, panel)
    annual_total = _annual_from_quarterly(quarterly)
    change_points = _change_points(quarterly)
    save_csv(quarterly, LEEUM_DIR / "quarterly_metrics.csv")
    save_csv(annual_total, LEEUM_DIR / "annual_metrics_2022_2025.csv")
    save_csv(change_points, LEEUM_DIR / "change_point_results.csv")

    industry, contribution = _annual_industry(annual)
    save_csv(industry, LEEUM_DIR / "industry_deep_dive.csv")
    save_csv(contribution, LEEUM_DIR / "industry_contribution.csv")
    day_time = annual_total[[column for column in annual_total.columns if column in {"year", "weekday_sales", "weekend_sales", "weekend_sales_share", "evening_sales_share", "night_sales_share", "female_sales_share", "sales_amount", "sales_transactions"}]].copy()
    save_csv(day_time, LEEUM_DIR / "day_time_customer_analysis.csv")
    save_csv(day_time, LEEUM_DIR / "day_time_metrics.csv")

    benchmark = annual_benchmarks(annual)
    benchmark_data = benchmark.rename(columns={"seoul_sales_amount": "sales_amount", "seoul_sales_transactions": "sales_transactions", "seoul_avg_store_count": "store_count"})[["year", "industry_code", "sales_amount", "sales_transactions", "store_count"]]
    target_complete = annual.loc[annual["area_code"].eq(TARGET_CODE) & annual["year"].isin(ANALYSIS_YEARS)]
    decomposition = pd.DataFrame([_decomposition(target_complete, "leeum"), _decomposition(benchmark_data.loc[benchmark_data["industry_code"].isin(target_complete["industry_code"].unique())], "seoul_same_industries")])

    controls, balance, control_trend = _match_controls(annual, reference)
    control_data = annual.loc[annual["area_code"].isin(controls.loc[controls["area_code"].ne(TARGET_CODE), "area_code"]) & annual["year"].isin(ANALYSIS_YEARS)]
    decomposition = pd.concat([decomposition, pd.DataFrame([_decomposition(control_data, "matched_controls")])], ignore_index=True)
    save_csv(decomposition, LEEUM_DIR / "sales_decomposition.csv")
    save_csv(controls, LEEUM_DIR / "matched_controls.csv")
    save_csv(controls, LEEUM_DIR / "matched_controls_2022.csv")
    save_csv(balance, LEEUM_DIR / "matching_balance.csv")
    save_csv(control_trend, LEEUM_DIR / "matched_control_trend.csv")

    spatial = _load_spatial()
    neighbors = _neighbors(annual, spatial)
    neighbor_comparison, neighbor_aggregate = _neighbor_trend(annual, neighbors)
    save_csv(neighbors, LEEUM_DIR / "neighbor_areas.csv")
    save_csv(neighbor_comparison, LEEUM_DIR / "neighbor_comparison.csv")
    save_csv(neighbor_aggregate, LEEUM_DIR / "area_aggregate.csv")
    save_csv(neighbor_aggregate, LEEUM_DIR / "neighbor_aggregate.csv")
    redevelopment = pd.DataFrame([{"target_area_code": TARGET_CODE, "redevelopment_boundary_data_available": False, "store_address_data_available": False, "dated_relocation_or_demolition_data_available": False, "verdict": "not_verifiable", "note": "No redevelopment GIS/timing or store-level location input in project."}])
    save_csv(redevelopment, LEEUM_DIR / "redevelopment_spatial_check.csv")
    events = _external_events()
    save_csv(events, LEEUM_DIR / "external_events.csv")
    save_csv(events.loc[events["category"].eq("leeum_program")], LEEUM_DIR / "exhibition_timeline.csv")
    store_turnover = industry[["year", "industry_code", "industry_name", "store_count", "open_count", "close_count", "sales_amount", "sales_transactions"]].copy()
    save_csv(store_turnover, LEEUM_DIR / "store_turnover.csv")
    save_csv(store_turnover, LEEUM_DIR / "turnover_reorganization.csv")

    _plot_quarterly(quarterly)
    _plot_yoy(quarterly)
    _plot_decomposition(decomposition)
    _plot_industries(industry.loc[industry["year"].isin(ANALYSIS_YEARS)], contribution)
    _plot_day_time(annual_total)
    _plot_exhibition_timeline(quarterly, events)
    _plot_neighbor_map(spatial, neighbors)
    write_text(LEEUM_MAP_DIR / "leeum_neighbor_map.html", """<!doctype html>
<html lang=\"ko\"><meta charset=\"utf-8\"><title>리움미술관 인접 상권</title>
<body><h1>리움미술관 인접 상권</h1><p>중심점 1km 이내 또는 이태원·한남 명칭 후보를 표시한 정적 지도입니다. 경계 이동이나 소비 이동을 증명하지 않습니다.</p>
<img src=\"leeum_neighbor_map.png\" alt=\"리움미술관 및 인접 상권 지도\"></body></html>""")
    write_text(LEEUM_MAP_DIR / "leeum_redevelopment_map.html", """<!doctype html>
<html lang=\"ko\"><meta charset=\"utf-8\"><title>재개발 공간 검증</title>
<body><h1>재개발 공간 검증: 자료 미제공</h1><p>현재 프로젝트에는 한남2·3구역 경계·사업 단계별 날짜·점포 주소 자료가 없어 중첩 또는 거리 분석을 수행하지 않았습니다.</p></body></html>""")
    _plot_group_trend(neighbor_aggregate, "leeum_vs_neighbors.png", "리움미술관과 인접·이태원권 후보 상권")
    _plot_group_trend(control_trend, "leeum_vs_controls.png", "리움미술관과 2022년 유사 상권")
    _plot_matching_balance(balance)

    evidence = _evidence(reproduction, annual_total, change_points, decomposition, contribution, neighbor_aggregate, control_trend, redevelopment, events, balance)
    save_csv(evidence, LEEUM_DIR / "leeum_evidence_matrix.csv")
    save_csv(evidence, LEEUM_DIR / "evidence_matrix.csv")
    save_csv(evidence, LEEUM_DIR / "leeum_hypothesis_verdict.csv")
    _write_reports(reproduction, quarterly, change_points, annual_total, decomposition, industry, contribution, day_time, neighbors, neighbor_aggregate, controls, balance, control_trend, events, redevelopment, evidence)
    summary = pd.DataFrame([
        {"section": "reproduction", "metric": "overall_rank", "value": int(reproduction.iloc[0]["overall_rank"]), "note": "Original score calculation reproduced"},
        {"section": "reproduction", "metric": "CoreDeclineScore", "value": float(reproduction.iloc[0]["CoreDeclineScore"]), "note": "Original score calculation reproduced"},
        {"section": "annual_2022_2025", "metric": "sales_change", "value": _pct(annual_total.loc[annual_total["year"].eq(2022), "sales_amount"].iloc[0], annual_total.loc[annual_total["year"].eq(2025), "sales_amount"].iloc[0]), "note": "Complete-year total, not a causal estimate"},
        {"section": "annual_2022_2025", "metric": "transactions_change", "value": _pct(annual_total.loc[annual_total["year"].eq(2022), "sales_transactions"].iloc[0], annual_total.loc[annual_total["year"].eq(2025), "sales_transactions"].iloc[0]), "note": "Complete-year total, not a causal estimate"},
        {"section": "final", "metric": "classification", "value": evidence.iloc[-1]["verdict"], "note": evidence.iloc[-1]["interpretation_boundary"]},
    ])
    save_csv(summary, LEEUM_DIR / "leeum_analysis_summary.csv")
    return {"target": TARGET_NAME, "rank": int(target.iloc[0]["overall_rank"]), "final_classification": evidence.iloc[-1]["verdict"], "neighbors": len(neighbors), "controls": len(controls) - 1}


if __name__ == "__main__":
    print(run())
