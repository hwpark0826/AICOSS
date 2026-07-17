"""Minimal regression test for the required score direction."""
import pandas as pd

from src.calculate_scores import standardize_decline


def test_lower_relative_performance_has_higher_decline_component() -> None:
    values = pd.Series([-0.5, 0.0, 0.5])
    score = standardize_decline(values, "robust")
    assert score.iloc[0] > score.iloc[1] > score.iloc[2]
