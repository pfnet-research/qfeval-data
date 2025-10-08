import numpy as np
import torch

from qfeval_data import Data

from .util import timestamps


def test_annualized_sharpe_ratio_basic() -> None:
    ts = timestamps(4)
    x = torch.tensor(
        [
            [101.0, 200.0],
            [102.0, 205.0],
            [100.0, 220.0],
            [101.0, 210.0],
        ],
        dtype=torch.float32,
    )
    data = Data.from_tensors({"price": x}, ts, np.array(["A", "B"]))

    ret = data.annualized_return().price.array
    vol = data.annualized_volatility().price.array
    exp = ret / vol

    np.testing.assert_allclose(
        data.annualized_sharpe_ratio().price.array, exp, rtol=5e-4, atol=5e-5
    )
