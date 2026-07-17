# 01. 결측·업종 이탈 가설 검증

대상 상권의 2021Q1–2026Q1, 10개 외식업종 전체 격자를 원시 매출·점포 파일에서 다시 만들었습니다. **행이 없다는 사실을 0으로 바꾸지 않았습니다.** `both_rows_missing`은 양쪽 원시 파일에 행이 없음을 뜻하고, `store_only`/`sales_only`는 한쪽 파일만 존재함을 뜻합니다.

## 대상 격자 상태

| state | cells |
| --- | --- |
| both_present_positive | 90 |
| both_rows_missing | 7 |
| store_only | 113 |

## 2021Q4·2025Q4 업종 상태

| quarter | industry_code | industry_name | state |
| --- | --- | --- | --- |
| 20214 | CS100001 | 한식음식점 | both_present_positive |
| 20214 | CS100002 | 중식음식점 | both_present_positive |
| 20214 | CS100003 | 일식음식점 | store_only |
| 20214 | CS100004 | 양식음식점 | store_only |
| 20214 | CS100005 | 제과점 | store_only |
| 20214 | CS100006 | 패스트푸드점 | store_only |
| 20214 | CS100007 | 치킨전문점 | both_present_positive |
| 20214 | CS100008 | 분식전문점 | both_present_positive |
| 20214 | CS100009 | 호프-간이주점 | both_present_positive |
| 20214 | CS100010 | 커피-음료 | store_only |
| 20254 | CS100001 | 한식음식점 | both_present_positive |
| 20254 | CS100002 | 중식음식점 | store_only |
| 20254 | CS100003 | - | both_rows_missing |
| 20254 | CS100004 | 양식음식점 | store_only |
| 20254 | CS100005 | 제과점 | store_only |
| 20254 | CS100006 | 패스트푸드점 | store_only |
| 20254 | CS100007 | 치킨전문점 | both_present_positive |
| 20254 | CS100008 | 분식전문점 | both_present_positive |
| 20254 | CS100009 | 호프-간이주점 | store_only |
| 20254 | CS100010 | 커피-음료 | store_only |

도시 전체에서도 2021년에 매출이 있던 상권-업종이 2025년에 사라지는 빈도는 업종별로 아래와 같습니다. 따라서 이런 패턴은 대상 하나에만 있는 현상인지 분리해 볼 수 있습니다.

| industry_code | area_industry_with_2021_sales | absent_sales_2025 | absent_2025_share |
| --- | --- | --- | --- |
| CS100001 | 1429 | 41 | 0.028691392582225334 |
| CS100002 | 488 | 81 | 0.16598360655737704 |
| CS100003 | 369 | 56 | 0.15176151761517614 |
| CS100004 | 356 | 49 | 0.13764044943820225 |
| CS100005 | 476 | 85 | 0.17857142857142858 |
| CS100006 | 465 | 89 | 0.1913978494623656 |
| CS100007 | 664 | 150 | 0.22590361445783133 |
| CS100008 | 860 | 103 | 0.11976744186046512 |
| CS100009 | 955 | 88 | 0.09214659685863874 |
| CS100010 | 1094 | 63 | 0.05758683729433273 |

## 순위 민감도

| scenario | target_included | target_rank | rank_change_vs_baseline | eligible_areas | top20 |
| --- | --- | --- | --- | --- | --- |
| A1_common_industries_all_2021_2025 | True | 1.0 | 0.0 | 815 | True |
| A2_baseline_drop_and_reweight | True | 1.0 | 0.0 | 847 | True |
| A3_missing_industry_equals_seoul_average | True | 1.0 | 0.0 | 891 | True |
| A4_sales_coverage_at_least_80pct | False | - | - | 825 | False |

- A1은 2021–2025 모든 해에 양 끝점 지표가 가능한 공통 업종만 사용합니다.
- A2는 기존 방식(끝점이 없는 업종은 상대지표 계산에서 제외하고 남은 업종 가중치를 재정규화)입니다.
- A3는 끝점 상대지표가 없는 시작 업종을 서울 평균과 같음(상대지표 0)으로 둡니다. 이는 실제 매출을 보정한 것이 아니라 순위 민감도용 반대 가정입니다.
- A4는 기존 방식에서 장기 매출가중 커버리지가 80% 이상인 상권만 비교합니다.

결측 처리 하나만으로 원 결론을 확정하거나 무효화하지 않으며, A4에서 대상이 제외되면 그 자체가 데이터 신뢰도 제한입니다.
