import numpy as np
import torch

from qfeval_data import Data

from .util import timestamps


def test_metrics_basic() -> None:
    ts = timestamps(4)
    x = torch.tensor(
        [
            [100.0, 200.0],
            [101.1, 202.5],
            [102.2, 205.1],
            [100.3, 203.2],
        ],
        dtype=torch.float32,
    )
    data = Data.from_tensors({"price": x}, ts, np.array(["A", "B"]))

    m = data.metrics()
    expected_cols = [
        "annualized_sharpe_ratio",
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
    ]
    assert m.columns == expected_cols

    np.testing.assert_allclose(
        m.get("annualized_return").array,
        data.annualized_return().array,
    )
    np.testing.assert_allclose(
        m.get("annualized_volatility").array,
        data.annualized_volatility().array,
    )
    np.testing.assert_allclose(
        m.get("maximum_drawdown").array,
        data.maximum_drawdown().array,
    )
    np.testing.assert_allclose(
        m.get("annualized_sharpe_ratio").array,
        data.annualized_sharpe_ratio().array,
    )
