import numpy as np
import torch

from qfeval_data import Data

from .util import timestamps


def test_annualized_volatility_basic() -> None:
    ts = timestamps(4)
    x = torch.tensor(
        [
            [101.0, 200.0],
            [102.0, 205.0],
            [100.0, 220.0],
            [101.0, 210.0],
        ],
        dtype=torch.float64,
    )
    data = Data.from_tensors({"price": x}, ts, np.array(["A", "B"]))
    actual = data.annualized_volatility()
    expected = np.nanstd(
        np.array(
            [
                [102.0 / 101.0 - 1.0, 205.0 / 200.0 - 1.0],
                [100.0 / 102.0 - 1.0, 220.0 / 205.0 - 1.0],
                [101.0 / 100.0 - 1.0, 210.0 / 220.0 - 1.0],
            ],
            dtype=np.float64,
        ),
        axis=0,
        ddof=0,
    ) * np.sqrt(365.25)
    np.testing.assert_allclose(actual.price.array, expected, atol=1e-5)


def test_annualized_volatility_with_nans() -> None:
    ts = timestamps(4)
    x = torch.tensor(
        [
            [float("nan"), 200.0],
            [100.0, float("nan")],
            [102.0, 210.0],
            [101.0, 205.0],
        ],
        dtype=torch.float64,
    )
    data = Data.from_tensors({"price": x}, ts, np.array(["A", "B"]))
    actual = data.annualized_volatility()
    expected = np.nanstd(
        np.array(
            [
                [102.0 / 100.0 - 1.0, 210.0 / 200.0 - 1.0],
                [101.0 / 102.0 - 1.0, 205.0 / 210.0 - 1.0],
            ],
            dtype=np.float64,
        ),
        axis=0,
        ddof=0,
    ) * np.sqrt(365.25)
    np.testing.assert_allclose(actual.price.array, expected, atol=1e-5)
