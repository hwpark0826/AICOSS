import numpy as np
import pandas as pd

from src.leeum.run_validation import _change_points, _decomposition


def test_sales_decomposition_identity() -> None:
    data = pd.DataFrame({
        "year": [2022, 2025], "sales_amount": [100.0, 96.0],
        "sales_transactions": [20.0, 16.0], "store_count": [4.0, 4.0],
    })
    result = _decomposition(data, "test")
    assert np.isclose(result["decomposition_residual"], 0.0)


def test_change_point_excludes_incomplete_total_quarters() -> None:
    quarters = ["20211", "20212", "20213", "20214", "20221", "20222", "20223", "20224", "20231", "20232", "20233", "20234"]
    frame = pd.DataFrame({
        "quarter": quarters,
        "total_series_comparable": [False, False, False] + [True] * 9,
    })
    for column in ["sales_amount", "sales_transactions", "sales_per_store", "transactions_per_store", "weekend_sales_share", "evening_sales_share"]:
        frame[column] = [999.0, 999.0, 999.0] + [100.0, 110.0, 120.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0]
    result = _change_points(frame)
    sales = result.loc[result["metric"].eq("sales_amount")].iloc[0]
    # The pre-2022Q4 mean uses 2021Q4 and the three comparable 2022 quarters.
    assert np.isclose(sales["pre_2022q4_mean"], (100.0 + 110.0 + 120.0 + 80.0) / 4)
