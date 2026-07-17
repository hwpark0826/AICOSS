"""Small deterministic tests for validation helpers."""
import numpy as np

from src.validation.run_validation import _ring_area_centroid


def test_ring_area_centroid_for_unit_square() -> None:
    ring = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    area, x, y = _ring_area_centroid(ring)
    assert area == 1.0
    assert x == 0.5
    assert y == 0.5
