# 서울시 외식업 상권 쇠퇴 탐색

서울시 상권분석서비스의 분기별 추정매출·점포 데이터를 사용해, 외식업 상권의 서울 동일업종 대비 상대적 약화 신호를 탐색합니다.

## 실행

```powershell
python -m src.run_analysis
```

증산역 4번의 결측·기준연도·경계·인접상권·통제군 반증 검증은 다음 명령으로 실행합니다.

```powershell
python -m src.validation.run_validation
```

## 폴더 안내

- `reports/`: 공통 방법론·품질·전체 순위 보고서
- `reports/jeungsan4/`: 증산역 4번 전용 검증 보고서. `README.md`는 요약·목차이고, `00`~`09`는 서로 다른 검증 질문만 다룹니다.
- `outputs/jeungsan4/`: 증산역 4번 전용 CSV, 그림, 현장확인 체크리스트
- `data/raw/`: 서울시 원본 CSV와 상권 폴리곤 구성파일
- `data/processed/`: 전체 상권 분석용 연간 패널

결측 행은 0으로 대체하지 않습니다. 제공되지 않은 외부 사건·점포 좌표·과거 폴리곤은 검증 불가로 보고합니다.
