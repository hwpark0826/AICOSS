"""Compare Leeum and the five retained independent neighbours using all observed industries."""
from __future__ import annotations

import pandas as pd

from ..build_panel import remove_duplicate_keys
from ..config import ANALYSIS_QUARTERS, OUTPUT_DIR, REPORT_DIR
from ..export_results import markdown_table, save_csv, write_text
from ..load_data import discover_inputs, load_area_data


TARGET_CODE = "3110091"
NEIGHBOR_PATH = OUTPUT_DIR / "leeum" / "neighbor_areas.csv"
OUT_DIR = OUTPUT_DIR / "leeum" / "all_industry_neighbor_index"
REPORT_PATH = REPORT_DIR / "leeum" / "19_all_industry_neighbor_index.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def run() -> dict[str, object]:
    neighbors = pd.read_csv(NEIGHBOR_PATH, dtype={"area_code": str})
    scope = pd.concat([
        pd.DataFrame({"area_code": [TARGET_CODE], "area_name": ["리움미술관"], "group": ["leeum"]}),
        neighbors.loc[:, ["area_code", "area_name"]].assign(group="independent_neighbors_equal_weight"),
    ], ignore_index=True)
    sales = load_area_data(discover_inputs(), "sales_area", industry_codes=None)
    sales, duplicate_audit = remove_duplicate_keys(sales, ["sales_amount", "sales_transactions"])
    if not duplicate_audit.empty:
        save_csv(duplicate_audit, OUT_DIR / "sales_duplicate_resolution.csv")
    sales = sales.loc[sales["area_code"].isin(scope["area_code"]) & sales["quarter"].isin(ANALYSIS_QUARTERS)].copy()
    sales["year"] = sales["quarter"].str[:4].astype(int)
    annual = sales.groupby(["area_code", "year"], as_index=False).agg(
        observed_all_industry_sales=("sales_amount", "sum"),
        observed_all_industry_transactions=("sales_transactions", "sum"),
        observed_industries=("industry_code", "nunique"),
    ).merge(scope, on="area_code", how="left")
    base = annual.loc[annual["year"].eq(2022), ["area_code", "observed_all_industry_sales"]].rename(
        columns={"observed_all_industry_sales": "sales_2022"}
    )
    annual = annual.merge(base, on="area_code", how="left")
    annual["sales_index_2022_100"] = annual["observed_all_industry_sales"] / annual["sales_2022"] * 100
    group = annual.groupby(["group", "year"], as_index=False).agg(
        equal_weight_mean_index=("sales_index_2022_100", "mean"),
        equal_weight_median_index=("sales_index_2022_100", "median"),
        areas=("area_code", "nunique"),
    )
    current = annual.loc[annual["year"].eq(2025)].sort_values("group")
    leeum_index = float(current.loc[current["group"].eq("leeum"), "sales_index_2022_100"].iloc[0])
    neighbor_group = group.loc[(group["group"].eq("independent_neighbors_equal_weight")) & group["year"].eq(2025)].iloc[0]
    save_csv(scope, OUT_DIR / "comparison_scope.csv")
    save_csv(annual, OUT_DIR / "all_industry_neighbor_annual_metrics.csv")
    save_csv(group, OUT_DIR / "all_industry_neighbor_equal_weight_index.csv")
    save_csv(current, OUT_DIR / "all_industry_neighbor_individual_indices_2025.csv")
    write_text(REPORT_PATH, f"""# 리움미술관과 인접 5개 상권: 전체 업종 매출지수

## 2022년 매출=100 기준, 2025년 지수

- 리움미술관: **{leeum_index:.1f}**
- 인접 5개 상권의 동일가중 평균: **{neighbor_group['equal_weight_mean_index']:.1f}**
- 인접 5개 상권의 중앙값: **{neighbor_group['equal_weight_median_index']:.1f}**

비교 상권은 한강진역 3번, 이태원(이태원역), 우사단길, 경리단길남측, 이태원엔틱가구거리다. 이태원 관광특구는 이태원역과의 검증된 폴리곤 중첩 때문에 제외했다.

## 계산 기준

- 각 상권에서 원자료로 관측된 모든 서비스업종 매출을 연간 합산했다.
- 각 상권의 2022년 합계를 100으로 표준화한 뒤, 5개 인접 상권의 지수를 동일가중 평균했다. 큰 상권(이태원역 등)의 매출 규모가 평균을 지배하지 않도록 하기 위함이다.
- 리움은 2022년에 7개, 2025년에 6개 업종이 관측됐다. 각 인접 상권도 관측 업종 수가 다르므로, 이 결과는 **각 상권의 원자료 관측 업종 합계에 대한 기술적 비교**다.

## 상권별 값

{markdown_table(current, ['area_name', 'observed_all_industry_sales', 'observed_all_industry_transactions', 'observed_industries', 'sales_index_2022_100'], 10)}

## 해석 경계

리움의 전체 업종 지수 73.2는 외식업만의 지수 58.1보다 높다. 화장품·일반의류 등 비외식업의 변동이 외식업 감소 일부를 상쇄했기 때문이다. 그러나 인접 5개 상권 평균 104.1보다 낮다는 사실은 리움의 변화가 인접권 평균보다 컸다는 기술적 차이를 뜻할 뿐, 소비가 인접 상권으로 이동했다거나 특정 원인이 입증됐다는 뜻은 아니다.
""")
    return {"leeum_index_2025": leeum_index, "neighbor_mean_index_2025": float(neighbor_group["equal_weight_mean_index"]), "neighbor_median_index_2025": float(neighbor_group["equal_weight_median_index"])}


if __name__ == "__main__":
    print(run())
