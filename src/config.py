"""Central analysis configuration.

All criteria specified in the project brief live here so a rerun can be
configured without changing calculation code.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
VALIDATION_REPORT_DIR = REPORT_DIR
TARGET_AREA_NAME = "증산역 4번"
TARGET_AREA_CODE = "3110451"
VALIDATION_QUARTERS = tuple(f"{year}{quarter}" for year in range(2021, 2026) for quarter in range(1, 5))
VALIDATION_ALL_QUARTERS = (*VALIDATION_QUARTERS, "20261")
MATCHED_CONTROL_COUNT = 7

FOOD_CODES = {
    "CS100001": "한식음식점",
    "CS100002": "중식음식점",
    "CS100003": "일식음식점",
    "CS100004": "양식음식점",
    "CS100005": "제과점",
    "CS100006": "패스트푸드점",
    "CS100007": "치킨전문점",
    "CS100008": "분식전문점",
    "CS100009": "호프·간이주점",
    "CS100010": "커피·음료",
}

ANALYSIS_QUARTERS = tuple(f"{year}{q}" for year in range(2021, 2026) for q in range(1, 5))
ANALYSIS_YEARS = tuple(range(2021, 2026))
PERIODS = {
    "long": (2021, 2025),
    "medium": (2023, 2025),
    "recent": (2024, 2025),
}
EARLY_WARNING_QUARTERS = ("20251", "20261")

METRIC_WEIGHTS = {
    "sales_rel": 0.25,
    "transactions_rel": 0.20,
    "sales_per_store_rel": 0.15,
    "store_rel": 0.20,
    "net_entry_rel": 0.20,
}
WEIGHT_SCENARIOS = {
    "기본": METRIC_WEIGHTS,
    "동일가중": {k: 0.20 for k in METRIC_WEIGHTS},
    "수요중심": {
        "sales_rel": 0.35,
        "transactions_rel": 0.35,
        "sales_per_store_rel": 0.30,
        "store_rel": 0.0,
        "net_entry_rel": 0.0,
    },
    "점포생태계중심": {
        "sales_rel": 0.0,
        "transactions_rel": 0.0,
        "sales_per_store_rel": 0.0,
        "store_rel": 0.50,
        "net_entry_rel": 0.50,
    },
    "점포당매출제외": {
        "sales_rel": 0.30,
        "transactions_rel": 0.25,
        "sales_per_store_rel": 0.0,
        "store_rel": 0.225,
        "net_entry_rel": 0.225,
    },
}
PERIOD_WEIGHTS = {"long": 0.50, "medium": 0.25, "recent": 0.25}
WINSOR_LIMITS = (0.01, 0.99)
MIN_START_STORES = 20
SIZE_SENSITIVITY = (20, 30, 50)
MIN_OBSERVATION_RATE = 0.80
MIN_WEIGHT_COVERAGE = 0.70
MIN_VALID_INDUSTRIES = 3
CSV_ENCODINGS = ("utf-8-sig", "cp949", "euc-kr")

KEY_COLUMNS = ["quarter", "area_code", "industry_code"]

for directory in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, REPORT_DIR, OUTPUT_DIR, TABLE_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
