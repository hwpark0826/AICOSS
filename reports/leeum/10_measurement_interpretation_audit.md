# 10. 측정 기준·해석 경계 재검토

## 결론

기존 순위 **5위**와 2022→2025 완결연도 절대 추이는 수치적으로 재현된다. 다만 이 둘은 같은 지표가 아니다. 순위는 서울 동일 업종 대비 **상대성과 복합점수**이고, 연도 총량은 리움의 **절대 집계값**이다. 이 문서는 두 결과를 서로의 증거로 바꾸어 해석하지 않는다.

## 즉시 정정한 항목

- 2021Q1–Q3은 관측 업종이 3개뿐이어서 총량·점포수·전년동기 비교·변화점 탐색에서 제외했다.
- 2022→2025 매출분해에서 건당 매출은 하락이 아니라 상승(+0.132 로그 기여)이며, 점포당 거래 하락(-0.721)이 매출 감소의 주된 회계 항목이다.
- 2022=100 업종지수와 2021→2025 업종 변동률은 기준연도가 다르다. 같은 문장에서 직접 대비하지 않는다.
- 모든 관측 업종의 2022→2025 매출이 감소했으므로, 현 자료는 업종 재편을 주된 설명으로 지지하지 않는다.

## 지표별 사용 규칙

| artifact | metric_type | time_or_base | allowed_interpretation | not_allowed |
| --- | --- | --- | --- | --- |
| commercial_area_decline_ranking.csv | relative composite ranking | long 2021→2025; medium 2023→2025; recent 2024→2025 | relative underperformance signal within the eligible sample | absolute sales decline or causal diagnosis |
| annual_metrics_2022_2025.csv | absolute complete-year totals | 2022→2025 | absolute change after the 2022 observed peak | 2021 total comparison or causal attribution |
| quarterly_metrics.csv | absolute quarterly series | 2021Q1→2026Q1 | only quarters with all four target industries observed are comparable | using 2021Q1–Q3 aggregate levels or YoY values as full-total evidence |
| industry_deep_dive.csv / industry_trends.png | industry index | 2022=100; chart uses 2022→2025 | within-industry change from 2022 | comparison to a 2021-based percentage without naming the different baseline |
| industry_contribution.csv / industry_contribution.png | absolute loss contribution | 2022→2025 | which industry contributes more won loss amount | proof of industry reorganization or relative percentage decline |
| sales_decomposition.csv | accounting decomposition | 2022→2025 | mechanical components associated with the sales change | customer origin, price cause, or causal mechanism |
| neighbor_aggregate.csv | independent-neighbour equal-weight index | 2022=100 | descriptive divergence from the retained independent-neighbour average | proven spatial displacement or a common-shock counterfactual |
| matched_control_trend.csv | imbalanced comparison aggregate index | 2022=100 | supplementary descriptive comparison only | causal treatment effect or counterfactual |

## 업종 기준연도 대조

| industry_name | sales_2021 | sales_2022 | sales_2025 | change_2021_2025 | change_2022_2025 | change_2021_2025_usable |
| --- | --- | --- | --- | --- | --- | --- |
| 한식음식점 | - | 3239171920.0 | 2715889412.0 | - | -0.16154823545148544 | False |
| 일식음식점 | 1858509036.0 | 2053408075.0 | 806968748.0 | -0.5657977807109247 | -0.6070100445085665 | True |
| 양식음식점 | 1581862021.0 | 740802561.0 | 265348527.0 | -0.8322555801470879 | -0.6418093821897572 | True |
| 커피-음료 | 1198271899.0 | 2283683554.0 | 1047182873.0 | -0.1260891005840069 | -0.5414500966362873 | True |

`change_2021_2025`와 `change_2022_2025`는 서로 다른 질문의 답이다. 전자는 코로나 시기를 포함한 2021년 대비 위치, 후자는 2022년 관측 고점 이후의 변화다. 분석의 총량·분해·인접 비교 그림은 후자만 사용한다.

## 수치 대사 결과

| check | expected | observed | difference | status |
| --- | --- | --- | --- | --- |
| annual sales identity 2022 | 8317066110.0 | 8317066110.0 | 0.0 | pass |
| annual transactions identity 2022 | 282150.0 | 282150.0 | 0.0 | pass |
| annual sales/store identity 2022 | 391391346.35294116 | 391391346.3529412 | 5.960464477539063e-08 | pass |
| annual sales identity 2023 | 7643848529.0 | 7643848529.0 | 0.0 | pass |
| annual transactions identity 2023 | 247271.0 | 247271.0 | 0.0 | pass |
| annual sales/store identity 2023 | 351441311.6781609 | 351441311.6781609 | 0.0 | pass |
| annual sales identity 2024 | 6552204107.0 | 6552204107.0 | 0.0 | pass |
| annual transactions identity 2024 | 204463.0 | 204463.0 | 0.0 | pass |
| annual sales/store identity 2024 | 294481083.46067417 | 294481083.46067417 | 0.0 | pass |
| annual sales identity 2025 | 4835389560.0 | 4835389560.0 | 0.0 | pass |
| annual transactions identity 2025 | 143705.0 | 143705.0 | 0.0 | pass |
| annual sales/store identity 2025 | 217320879.1011236 | 217320879.1011236 | 0.0 | pass |
| sales decomposition identity 2022→2025 | 0.0 | 0.0 | 0.0 | pass |
| incomplete total-series quarters excluded | 20211, 20212, 20213 | 20211, 20212, 20213 |  | pass |

## 표현 허용·금지 목록

| claim_or_chart | status | preferred_wording |
| --- | --- | --- |
| rank 5 / CoreDeclineScore 2.231 | valid with scope | Eligible-sample relative-underperformance rank, not an absolute-decline rank. |
| 2021 total vs 2025 total | invalid | Do not compare: 2021Q1–Q3 omit one target industry. |
| 2021→2025 industry percentages | valid only per complete industry | Label the start/end years explicitly; never compare directly with a 2022=100 chart. |
| 2022→2025 total sales, transactions, stores | valid descriptive total | Complete-year absolute trend after the observed 2022 peak; no causal claim. |
| sales decomposition | valid accounting identity | Transaction intensity is the negative component; transaction value rises and partly offsets it. |
| industry reorganization | not supported | All four observed industries decline; store/business continuity data are absent. |
| neighbor/control indices | descriptive only | Indices are 2022=100 aggregates; control balance is poor (max \|SMD\| 2.69). |
| 2022Q4 event break | not identified | 2022Q4 is an event marker; the descriptive mean-break candidate is 2024Q3 and is not causal. |

## 재현 경로

1. `python -m src.leeum.run_validation`
2. `python -m src.leeum.run_measurement_audit`

두 명령은 원자료를 0으로 보정하지 않는다. 2021 불완전 집계는 별도 표기로 보존하되, 비교 가능한 총량·전년동기·변화점 계산에서는 제외한다.
