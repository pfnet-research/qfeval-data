from math import nan

import numpy as np
import torch

from qfeval_data import Data

from .util import timestamps


def test_maximum_drawdown_basic() -> None:
    ts = timestamps(4)
    x = torch.tensor(
        [
            [100.0, 200.0],
            [120.0, 180.0],
            [90.0, 170.0],
            [110.0, 220.0],
        ],
    )
    data = Data.from_tensors({"price": x}, ts, np.array(["A", "B"]))

    # A: peak 120 -> trough 90 => -25%; B: peak 200 -> trough 170 => -15%
    expected = [0.25, 0.15]
    np.testing.assert_allclose(
        data.maximum_drawdown().price.array, expected, atol=1e-6
    )


def test_maximum_drawdown_with_nans() -> None:
    ts = timestamps(4)
    x = torch.tensor(
        [
            [nan, 200.0, nan],
            [100.0, nan, nan],
            [90.0, 200.0, nan],
            [110.0, 190.0, nan],
        ],
    )
    data = Data.from_tensors({"price": x}, ts, np.array(["A", "B", "C"]))

    # A: from first valid 100 peak to 90 => -10%; B: 200 peak to 190 => -5%
    expected = [0.10, 0.05, nan]
    np.testing.assert_allclose(
        data.maximum_drawdown().price.array, expected, atol=1e-6
    )
