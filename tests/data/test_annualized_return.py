from math import nan

import numpy as np
import torch

from qfeval_data import Data

from .util import timestamps


def test_annualized_return_basic() -> None:
    ts = timestamps(3)
    x = torch.tensor(
        [
            [100.0, 200.0],
            [100.5, nan],
            [100.2, 200.1],
        ],
        dtype=torch.float32,
    )
    data = Data.from_tensors({"price": x}, ts, np.array(["A", "B"]))
    actual = data.annualized_return()
    expected = [
        (100.2 / 100.0) ** (365.25 / 2.0) - 1.0,
        (200.1 / 200.0) ** (365.25 / 2.0) - 1.0,
    ]
    np.testing.assert_allclose(actual.price.array, expected, atol=1e-4)


def test_annualized_return_with_nans() -> None:
    ts = timestamps(3)
    x = torch.tensor(
        [
            [nan, 200.0],
            [100.0, nan],
            [100.1, 200.1],
        ],
        dtype=torch.float32,
    )
    data = Data.from_tensors({"price": x}, ts, np.array(["A", "B"]))
    actual = data.annualized_return()
    expected = [
        (100.1 / 100.0) ** (365.25 / 2.0) - 1.0,
        (200.1 / 200.0) ** (365.25 / 2.0) - 1.0,
    ]
    np.testing.assert_allclose(actual.price.array, expected, atol=1e-4)
