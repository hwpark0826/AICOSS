"""Input discovery and schema-normalized loading utilities."""
from __future__ import annotations

import csv
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import CSV_ENCODINGS, FOOD_CODES, RAW_DIR


@dataclass(frozen=True)
class InputFile:
    path: Path
    role: str
    encoding: str
    columns: tuple[str, ...]


def detect_encoding(path: Path) -> str:
    """Return the first configured CSV encoding that parses the header."""
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                next(csv.reader(handle))
            return encoding
        except (UnicodeDecodeError, StopIteration):
            continue
    raise UnicodeError(f"지원 인코딩으로 읽을 수 없습니다: {path.name}")


def classify_file(path: Path, columns: Iterable[str]) -> str:
    """Classify source files from actual Korean schema, never filename alone."""
    column_set = set(columns)
    if {"당월_매출_금액", "당월_매출_건수"}.issubset(column_set):
        return "sales_area"
    if "개업_점포_수" in column_set and ("점포_수" in column_set or "전체_점포_수" in column_set):
        return "stores_area"
    if "상권_변화_지표" in column_set:
        return "district_change_indicator"
    return "other"


def discover_inputs(root: Path = RAW_DIR) -> list[InputFile]:
    """Recursively discover CSV inputs and record their parsed headers."""
    inputs: list[InputFile] = []
    for path in sorted(root.rglob("*.csv")):
        relative_parts = path.relative_to(root).parts
        # Generated files are deliberately never allowed to become inputs on a
        # rerun. This preserves a strict raw-input/derived-output separation.
        if any(part in {"data", "outputs", "reports", "src", "tests", ".git", ".deps"} for part in relative_parts):
            continue
        encoding = detect_encoding(path)
        with path.open("r", encoding=encoding, newline="") as handle:
            columns = tuple(next(csv.reader(handle)))
        inputs.append(InputFile(path, classify_file(path, columns), encoding, columns))
    return inputs


def _resolve_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise KeyError(f"필수 컬럼을 찾지 못했습니다. 후보: {list(candidates)}; 실제: {list(columns)}")


def _read_filtered(path: Path, role: str, encoding: str, chunksize: int = 100_000) -> pd.DataFrame:
    """Load only selected food-service rows and normalize source schema."""
    header = pd.read_csv(path, encoding=encoding, nrows=0).columns.tolist()
    quarter_col = _resolve_column(header, ["기준_년분기_코드"])
    area_code_col = _resolve_column(header, ["상권_코드"])
    area_name_col = _resolve_column(header, ["상권_코드_명"])
    area_type_col = _resolve_column(header, ["상권_구분_코드_명"])
    industry_code_col = _resolve_column(header, ["서비스_업종_코드"])
    industry_name_col = _resolve_column(header, ["서비스_업종_코드_명"])
    common = [quarter_col, area_code_col, area_name_col, area_type_col, industry_code_col, industry_name_col]
    if role == "sales_area":
        measure_cols = [_resolve_column(header, ["당월_매출_금액"]), _resolve_column(header, ["당월_매출_건수"])]
        names = ["sales_amount", "sales_transactions"]
    else:
        measure_cols = [
            _resolve_column(header, ["전체_점포_수", "점포_수"]),
            _resolve_column(header, ["개업_점포_수"]),
            _resolve_column(header, ["폐업_점포_수"]),
        ]
        names = ["store_count", "open_count", "close_count"]
    selected = common + measure_cols
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, encoding=encoding, usecols=selected, chunksize=chunksize, low_memory=False):
        chunk[industry_code_col] = chunk[industry_code_col].astype(str).str.strip()
        chunk = chunk.loc[chunk[industry_code_col].isin(FOOD_CODES)].copy()
        if chunk.empty:
            continue
        chunk = chunk.rename(columns={
            quarter_col: "quarter", area_code_col: "area_code", area_name_col: "area_name",
            area_type_col: "area_type", industry_code_col: "industry_code",
            industry_name_col: "industry_name", **dict(zip(measure_cols, names)),
        })
        chunk["source_file"] = path.name
        parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=["quarter", "area_code", "area_name", "area_type", "industry_code", "industry_name", *names, "source_file"])
    data = pd.concat(parts, ignore_index=True)
    data["quarter"] = data["quarter"].astype(str).str.replace(".0", "", regex=False).str.strip()
    data["area_code"] = data["area_code"].astype(str).str.replace(".0", "", regex=False).str.strip()
    for name in names:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    return data


def load_area_data(inputs: list[InputFile], role: str) -> pd.DataFrame:
    """Load all source files with a given role and append a source identifier."""
    files = [item for item in inputs if item.role == role]
    if not files:
        raise FileNotFoundError(f"{role} 역할의 데이터 파일이 없습니다.")
    return pd.concat([_read_filtered(item.path, role, item.encoding) for item in files], ignore_index=True)


def read_area_reference(root: Path = RAW_DIR) -> pd.DataFrame:
    """Read the available DBF commercial-area reference without external GIS packages."""
    dbfs = sorted(root.rglob("*영역-상권*.dbf"))
    if not dbfs:
        return pd.DataFrame(columns=["area_code", "district", "administrative_dong"])
    path = dbfs[0]
    with path.open("rb") as handle:
        header = handle.read(32)
        record_count, header_length, record_length = struct.unpack("<xxxxIHH20x", header)
        fields: list[tuple[str, str, int]] = []
        while True:
            descriptor = handle.read(32)
            if descriptor[0] == 0x0D:
                break
            name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
            fields.append((name, chr(descriptor[11]), descriptor[16]))
        handle.seek(header_length)
        records: list[dict[str, str]] = []
        for _ in range(record_count):
            raw = handle.read(record_length)
            if len(raw) < record_length or raw[:1] == b"*":
                continue
            offset = 1
            row: dict[str, str] = {}
            for name, _, length in fields:
                value = raw[offset: offset + length]
                offset += length
                # The accompanying .cpg declares UTF-8; older DBF field names
                # are truncated to ten characters, so names are matched below.
                row[name] = value.decode("utf-8", errors="replace").strip()
            records.append(row)
    data = pd.DataFrame(records)
    lookup = {
        "TRDAR_CD": "area_code", "TRDAR_CD_N": "area_name_reference",
        "SIGNGU_CD_N": "district", "SIGNGU_CD_": "district",
        "ADSTRD_CD_N": "administrative_dong", "ADSTRD_CD_": "administrative_dong",
        "TRDAR_SE_1": "area_type_reference",
        "XCNTS_VALU": "centroid_x", "YDNTS_VALU": "centroid_y",
    }
    existing = {source: target for source, target in lookup.items() if source in data.columns}
    data = data.rename(columns=existing)
    result = data[list(existing.values())].drop_duplicates("area_code") if "area_code" in data else pd.DataFrame()
    for coordinate in ("centroid_x", "centroid_y"):
        if coordinate in result:
            result[coordinate] = pd.to_numeric(result[coordinate], errors="coerce")
    return result


def quarter_to_year(quarter: pd.Series) -> pd.Series:
    """Extract a four-digit year from the service's YYYYQ quarter code."""
    return pd.to_numeric(quarter.astype(str).str[:4], errors="coerce").astype("Int64")
