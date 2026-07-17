import numpy as np
import pandas as pd

from src.leeum.run_validation import _decomposition


def test_sales_decomposition_identity() -> None:
    data = pd.DataFrame({
        "year": [2022, 2025], "sales_amount": [100.0, 96.0],
        "sales_transactions": [20.0, 16.0], "store_count": [4.0, 4.0],
    })
    result = _decomposition(data, "test")
    assert np.isclose(result["decomposition_residual"], 0.0)
