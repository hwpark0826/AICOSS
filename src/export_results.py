"""CSV and Markdown reporting helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def save_csv(data: pd.DataFrame, path: Path) -> None:
    """Save analysis tables in Excel-friendly UTF-8 with BOM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, content: str) -> None:
    """Save a UTF-8 Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def markdown_table(data: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    """Render a small Markdown table without adding an optional tabulate dependency."""
    if data.empty:
        return "자료 없음"
    frame = data.loc[:, [column for column in columns if column in data.columns]].head(limit).copy()
    frame = frame.fillna("-")
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def format_inventory(inventory: pd.DataFrame) -> str:
    """Create the data-inventory report body."""
    visible = ["file", "role", "encoding", "raw_rows", "period", "food_rows", "food_areas", "food_industries", "duplicate_keys", "missing_required"]
    return "# 데이터 인벤토리\n\n원본 파일은 변경하지 않았습니다. CSV 인코딩은 `utf-8-sig → cp949 → euc-kr` 순으로 판별했습니다.\n\n" + markdown_table(inventory, visible, 50) + "\n\n## 실제 분석 컬럼\n\n- 매출: `기준_년분기_코드`, `상권_코드`, `상권_코드_명`, `상권_구분_코드_명`, `서비스_업종_코드`, `서비스_업종_코드_명`, `당월_매출_금액`, `당월_매출_건수`\n- 점포: 위 식별 컬럼과 `점포_수` 또는 `전체_점포_수`, `개업_점포_수`, `폐업_점포_수`\n- 기준정보(DBF): `TRDAR_CD`, `TRDAR_CD_N`, `SIGNGU_CD_N`, `ADSTRD_CD_N`\n"


def format_quality(summary: dict[str, Any], issues: pd.DataFrame) -> str:
    """Create a concise quality report from check results."""
    counts = issues.groupby("issue_type").size().reset_index(name="rows") if not issues.empty else pd.DataFrame(columns=["issue_type", "rows"])
    values = "\n".join(f"- {key}: {value}" for key, value in summary.items())
    return "# 데이터 품질 보고서\n\n## 요약\n\n" + values + "\n\n## 이슈 건수\n\n" + markdown_table(counts, ["issue_type", "rows"], 30) + "\n\n원본의 결측·0·중복·이상값은 임의 보정하지 않았습니다. 충돌 중복키가 있으면 분석 대상에서 제외하고, 정확히 동일한 중복행은 한 행만 사용합니다. 상세 행은 `outputs/tables/data_quality_issues.csv`에 있습니다.\n"


def format_methodology(parquet_note: str) -> str:
    """Write methods and explicit implementation conventions."""
    return f"""# 방법론\n\n## 목적\n\n서울시 외식업 상권의 원인을 단정하지 않고, 서울 동일 세부업종 대비 상대적 악화 신호를 복합지수로 순위화했습니다.\n\n## 분석 범위\n\n- 업종: CS100001~CS100010의 10개 외식 세부업종\n- 주 분석 기간: 2021Q1~2025Q4\n- 분석 단위: 분기 × 상권코드 × 세부업종, 연간 패널은 연도 × 상권코드 × 세부업종\n- 기준선: 공식 서울시 업종 자료가 없어, 상권 단위 자료를 업종별로 합산한 서울 동일 업종 기준선\n\n## 연간화\n\n매출액·거래건수·개업·폐업은 4개 분기 합계, 점포수는 연평균, 운영점포수 성과에는 4분기 말 점포수를 사용했습니다. 4개 분기가 모두 없는 상권-업종-연도는 불완전 연도로 표시하고 합계에 포함하지 않았습니다.\n\n## 상대성과와 점수\n\n매출액·거래건수·점포당매출·연말점포수는 `ln(상권 종료/상권 시작) - ln(서울동일업종 종료/서울동일업종 시작)`으로 계산했습니다. 순진입률은 기간 평균 상권 순진입률에서 서울 동일업종 기간 평균 순진입률을 뺐습니다. 0 또는 결측이 있는 로그 지표는 결측 처리했습니다. 업종별 값은 시작연도 매출 또는 점포 비중으로 가중평균했습니다.\n\n기간별 점수는 1%·99% 윈저라이징 후 median/MAD 기반 robust Z-score의 부호를 뒤집어(낮은 상대성과일수록 높은 점수) 25%·20%·15%·20%·20%로 합산했습니다. 최종 CoreDeclineScore는 장기 50%, 중기 25%, 최근 25%입니다.\n\n## 적격성 및 강건성\n\n주 순위는 2021년 외식업 점포 20개 이상, 분석기간 평균 점포 20개 이상, 유효 업종 3개 이상, 분기 관측률 80% 이상, 매출·점포 가중치 커버리지 각각 70% 이상을 충족한 상권만 포함합니다. 가중치·점포기준·표준화 방법을 바꾼 30개 시나리오로 순위 안정성을 점검했습니다.\n\n## 파일 형식 참고\n\n{parquet_note}\n"""
