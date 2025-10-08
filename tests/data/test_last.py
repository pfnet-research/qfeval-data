from math import nan

import numpy as np
import torch

from qfeval_data import Data

from .util import create_simple_dataframe
from .util import symbols
from .util import timestamps


def test_last() -> None:
    data = Data.from_dataframe(create_simple_dataframe())

    # Shape semantics
    assert data.last(None).size() == (1, 1)
    assert data.last("timestamp").size() == (1, 2)
    assert data.last("symbol").size() == (3, 1)

    # Values
    # overall last non-NaN considering ffill over time then symbols
    np.testing.assert_allclose(data.last(None).open.array, 190.0)
    # along timestamp
    np.testing.assert_allclose(
        data.last(axis=0).open.array,
        [110.0, 190.0],
    )
    # along symbol
    np.testing.assert_allclose(
        data.last(axis=1).open.array,
        [200.0, 110.0, 190.0],
        equal_nan=True,
    )
    # Column-wise (skipna=True): picks last non-NaN among columns -> equals "close"
    np.testing.assert_allclose(
        data.last(axis=2).array,
        data.close.array,
        equal_nan=True,
    )

    # skipna=False behaviors
    np.testing.assert_allclose(
        data.last(axis=1, skipna=False).open.array,
        [200.0, np.nan, 190.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        data.last(axis=2, skipna=False).array,
        data.close.array,
        equal_nan=True,
    )


def test_last_with_nan() -> None:
    # Verify ffill behavior along timestamp axis (axis=0)
    ts = timestamps(3)
    sy = symbols(2)
    x = torch.tensor(
        [
            [nan, nan],
            [10.0, nan],
            [nan, 20.0],
        ],
        dtype=torch.float32,
    )
    d0 = Data.from_tensors({"x": x}, ts, sy)
    # skipna=True uses ffill then takes last
    np.testing.assert_allclose(d0.last(axis=0).x.array, [10.0, 20.0])
    # skipna=False keeps the original last row
    np.testing.assert_allclose(
        d0.last(axis=0, skipna=False).x.array,
        [np.nan, 20.0],
        equal_nan=True,
    )

    # Verify ffill behavior along symbol axis (axis=1)
    y = torch.tensor(
        [
            [3.0, nan],
            [nan, nan],
            [5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    d1 = Data.from_tensors({"x": y}, ts, sy)
    np.testing.assert_allclose(
        d1.last(axis=1).x.array,
        [3.0, nan, 6.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        d1.last(axis=1, skipna=False).x.array,
        [nan, nan, 6.0],
        equal_nan=True,
    )

    # Verify ffill behavior along column axis (axis=2)
    a = torch.tensor(
        [
            [nan, 1.0],
            [nan, nan],
            [4.0, nan],
        ],
        dtype=torch.float32,
    )
    b = torch.tensor(
        [
            [7.0, 8.0],
            [9.0, nan],
            [nan, 10.0],
        ],
        dtype=torch.float32,
    )
    d2 = Data.from_tensors({"a": a, "b": b}, ts, sy)
    expected = np.where(~np.isnan(b.numpy()), b.numpy(), a.numpy())
    np.testing.assert_allclose(
        d2.last(axis=2).array,
        expected,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        d2.last(axis=2, skipna=False).array,
        b.numpy(),
        equal_nan=True,
    )
