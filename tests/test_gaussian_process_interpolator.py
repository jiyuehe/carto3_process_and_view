import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_patient_data_observer import _build_linear_interpolator


def test_build_linear_interpolator_returns_predictor():
    sample_points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=float)
    sample_values = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

    predictor = _build_linear_interpolator(sample_points, sample_values)
    prediction = predictor(np.array([[0.25, 0.25], [0.75, 0.75]], dtype=float))

    assert np.isfinite(prediction).all()
    assert np.ptp(prediction) > 0.2
