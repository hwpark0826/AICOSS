# 서울시 상권 쇠퇴·리움미술관 상권 분석

서울시 상권분석서비스 원자료를 바탕으로 다음 세 범위를 관리합니다.

- 서울 전체 외식업 상권의 상대 쇠퇴 탐색
- 증산역 4번 상권의 반증 검증
- 리움미술관 인근 상권의 전체 업종·외식업·유동인구·소비행태 분석

원자료는 보존하고, 결측 행을 0으로 대체하지 않습니다. 제공되지 않은 외부 사건·점포 좌표·과거 폴리곤은 검증 불가로 표시합니다.

## 주요 실행

```powershell
# 서울 전체 외식업 상대 쇠퇴 순위
python -m src.run_analysis

# 서울 전체 업종 상대 쇠퇴 순위
python -m src.run_all_industry_analysis

# 증산역 4번 반증 검증
python -m src.validation.run_validation

# 리움미술관 종합 검증
python -m src.leeum.run_validation
```

리움미술관의 추가 분석은 `src/leeum/`의 목적별 실행 파일로 분리되어 있습니다. 파일명은 각각 유동인구, 카페 시간·요일, 외식업-비외식업 관계, 전체 업종 손실·인접상권 지수 분석을 가리킵니다.

## 폴더 구조

| 경로 | 내용 |
| --- | --- |
| `data/raw/` | 서울시 원본 CSV와 상권 폴리곤 구성파일 |
| `data/processed/` | 분석용 중간 패널 |
| `src/` | 전체 상권·증산역 4번·리움미술관 분석 코드 |
| `reports/` | 해석·방법론·검증 보고서 |
| `outputs/all_industry/` | 전체 업종 상대 쇠퇴 순위 산출물 |
| `outputs/jeungsan4/` | 증산역 4번 전용 표·그림·점검자료 |
| `outputs/leeum/` | 리움미술관 전용 표·그림·지도·세부 분석 결과 |
| `outputs/presentations/` | 최종 발표자료와 재현용 차트·집계 데이터 |

## 발표자료

현재 발표용 최종본은 `outputs/presentations/내가함 진짜_전면개편.pptx`입니다. 같은 폴더의 `mumut_redesign/`에는 해당 장표의 차트 원본(SVG·PNG), 집계 수치, 차트 생성 코드가 있습니다.
