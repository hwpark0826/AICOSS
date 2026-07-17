# 데이터 품질 보고서

## 요약

- sales_rows: 142554
- sales_duplicate_rows: 0
- stores_rows: 258931
- stores_duplicate_rows: 0
- missing_analysis_quarters: []
- observed_analysis_quarters: ['20211', '20212', '20213', '20214', '20221', '20222', '20223', '20224', '20231', '20232', '20233', '20234', '20241', '20242', '20243', '20244', '20251', '20252', '20253', '20254']
- issue_rows: 118225
- sales_store_match_rate: 0.5505482155477714

## 이슈 건수

| issue_type | rows |
| --- | --- |
| area_name_changed | 6 |
| sales_with_zero_store | 1842 |
| store_without_sales | 116377 |

원본의 결측·0·중복·이상값은 임의 보정하지 않았습니다. 충돌 중복키가 있으면 분석 대상에서 제외하고, 정확히 동일한 중복행은 한 행만 사용합니다. 상세 행은 `outputs/tables/data_quality_issues.csv`에 있습니다.
