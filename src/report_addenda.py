"""Small method-report addenda kept separate from the core report template."""


def confirmed_exit_methodology_note() -> str:
    """Describe the additional structural-exit sensitivity scenario."""
    return """

## 확정 업종 소멸 민감도 시나리오

기본 30개 시나리오와 별도로, 점포 자료에서 확인된 업종 소멸을 반영한 추가 1개 시나리오를 계산했습니다. 시작연도 4개 분기가 모두 관측되고 점포 수가 양수이며, 종료연도 4분기 점포 수가 0이고, 해당 기간 폐업점포 수가 하나 이상일 때만 ‘확정 업종 소멸’로 판단했습니다.

매출 행의 부재만으로는 폐업으로 판단하지 않았고 매출을 0으로 보정하지 않았습니다. 이 시나리오에서는 기본 5개 지표 점수의 80%와, 시작연도 점포 비중으로 가중한 확정 업종 소멸 신호의 20%를 합산했습니다. 소멸 신호는 희소한 0-1형 지표이므로 일반 Z-score 대신 `3 × 시작연도 점포 비중`을 사용하고 0~3 범위로 제한했습니다. 따라서 확정 소멸이 없는 상권은 이 보조 신호로 감점되지 않습니다.
"""
