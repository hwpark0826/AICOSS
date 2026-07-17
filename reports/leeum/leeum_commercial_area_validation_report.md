# 리움미술관 상권 대체가설 검증 보고서

## 분석 원칙

이 문서는 기존 5위 결론을 정당화하지 않는다. 각 가설에서 관측된 사실, 그 사실에 대한 해석, 인과가설을 분리하고 반대 증거와 데이터 한계를 함께 기록한다.

## 최종 판정

**2022년 고점 이후 거래기반 약화 관측; 구조적 쇠퇴 원인은 식별 불가**

| hypothesis | predicted_pattern | observed_facts | supporting_evidence | counter_evidence | verdict | confidence |
| --- | --- | --- | --- | --- | --- | --- |
| H1_2022_temporary_peak | 2022 only is unusually high and later years return near a normal level | 2022→2025 sales -41.9%; best mean split 2024Q3 | 2022 is the highest complete annual total | 2025 remains materially below 2023 and 2024; transactions -49.1% | partially_supported | medium |
| H2_itaewon_external_shock | A sustained post-2022Q4 fall, especially visitor-oriented weekend/evening demand and neighbouring areas | post/pre sales mean change -4.6%; event date is verified | The official event date falls in 2022Q4 | No visitor origin, exposure intensity, or causal control is available; break timing can reflect concurrent changes | not_verifiable | low |
| H3_spatial_reallocation | Leeum falls while nearby candidate areas or same industries rise | 2025 index: Leeum 58.1, neighbours 97.4 | Relative divergence is observable if candidate aggregate is higher | No movement paths, store-address transitions, or origin-destination data | possible | low |
| H4_redevelopment_indirect_effect | Dated redevelopment exposure overlaps residential/daytime demand weakening | No redevelopment boundary or dated relocation/demolition input available | None in the provided data | No spatial or timing evidence | not_verifiable | low |
| H5_industry_recomposition | Total stores are stable but industry shares/turnover change | 2022→2025 stores 4.7%; declining industries: 일식음식점, 커피-음료, 한식음식점, 양식음식점 | Industry-level sales changes are heterogeneous while total stores are stable | No business/address identifier to observe actual replacement | partially_supported | medium |
| H6_underlying_demand_weakening | Transactions and transactions per store fall even if store count remains stable | transactions -49.1%; stores 4.7%; tx/store log effect -0.721 | Transaction deterioration is not explained by store count alone | Demand source (resident, worker, tourist) is not observed | partially_supported | medium |
| H7_matched_area_relative_weakening | Leeum declines more than similarly configured 2022 areas | 2025 index: Leeum 58.1, matched controls 84.5; max \|SMD\| 2.69 | The selected comparison areas decline less in the observed series | The selected controls remain materially unbalanced (max \|SMD\| exceeds 0.25) and omit tourist exposure, rent, and museum visits | possible | low |
| final_classification |  | existing rank=5; partially_supported=3 |  | 2021 incomplete Korean data, unbalanced controls, and unavailable causal covariates | 2022년 고점 이후 거래기반 약화 관측; 구조적 쇠퇴 원인은 식별 불가 | medium |

## 핵심 해석

2022년 이후 총매출·거래·점포당 거래가 약해진 것은 관측 사실이다. 2022년 고점 정상화, 이태원권 공통충격, 공간 재배치, 재개발 간접영향, 업종 재편은 서로 배타적이지 않은 설명 후보지만, 현재 제공된 데이터만으로는 단일 원인으로 식별되지 않는다.

## 재현

`python -m src.leeum.run_validation` 실행 후 모든 CSV·그림은 `outputs/leeum/`, 세부 보고서는 `reports/leeum/`에 생성된다.
