from math import nan

import numpy as np
import torch

from qfeval_data import Data

from .util import create_simple_dataframe
from .util import symbols
from .util import timestamps


def test_first() -> None:
    data = Data.from_dataframe(create_simple_dataframe())

    # Shape semantics
    assert data.first(None).size() == (1, 1)
    assert data.first("timestamp").size() == (1, 2)
    assert data.first("symbol").size() == (3, 1)

    # Values
    np.testing.assert_allclose(data.first(None).open.array, 100.0)
    np.testing.assert_allclose(
        data.first(axis=0).open.array,
        [100.0, 200.0],
    )
    np.testing.assert_allclose(
        data.first(axis=1).open.array,
        [100.0, 110.0, 190.0],
    )
    # Column-wise (skipna=True): picks first non-NaN among columns -> equals "open"
    np.testing.assert_allclose(
        data.first(axis=2).array,
        data.open.array,
        equal_nan=True,
    )

    # skipna=False: symbol axis should take first symbol even if NaN appears later
    np.testing.assert_allclose(
        data.first(axis=1, skipna=False).open.array,
        [100.0, 110.0, np.nan],
        equal_nan=True,
    )
    # skipna=False: column axis should select the first column as-is
    np.testing.assert_allclose(
        data.first(axis=2, skipna=False).array,
        data.open.array,
        equal_nan=True,
    )


def test_first_with_nan() -> None:
    # Verify bfill behavior along timestamp axis (axis=0)
    ts = timestamps(3)
    sy = symbols(2)
    x = torch.tensor(
        [
            [nan, nan],  # first row all NaN
            [10.0, nan],  # next has value for col0
            [nan, 20.0],  # later has value for col1
        ],
        dtype=torch.float32,
    )
    d0 = Data.from_tensors({"x": x}, ts, sy)
    np.testing.assert_allclose(d0.first(axis=0).x.array, [10.0, 20.0])
    # skipna=False should keep the first row which is all-NaN
    np.testing.assert_allclose(
        d0.first(axis=0, skipna=False).x.array,
        [np.nan, np.nan],
        equal_nan=True,
    )

    # Verify bfill behavior along symbol axis (axis=1)
    y = torch.tensor(
        [
            [nan, 3.0],  # first symbol NaN, second valid -> 3.0
            [nan, nan],  # all NaN -> NaN
            [5.0, 6.0],  # first valid -> 5.0
        ],
        dtype=torch.float32,
    )
    d1 = Data.from_tensors({"x": y}, ts, sy)
    np.testing.assert_allclose(
        d1.first(axis=1).x.array,
        [3.0, nan, 5.0],
        equal_nan=True,
    )
    # skipna=False should take first symbol's value as-is
    np.testing.assert_allclose(
        d1.first(axis=1, skipna=False).x.array,
        [np.nan, np.nan, 5.0],
        equal_nan=True,
    )

    # Verify bfill behavior along column axis (axis=2)
    a = torch.tensor(
        [
            [nan, 1.0],  # choose a if not NaN else fallback to b
            [nan, nan],  # all NaN -> NaN
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
    expected = np.where(~np.isnan(a.numpy()), a.numpy(), b.numpy())
    np.testing.assert_allclose(
        d2.first(axis=2).array,
        expected,
        equal_nan=True,
    )
    # skipna=False should select the first column `a` unchanged
    np.testing.assert_allclose(
        d2.first(axis=2, skipna=False).array,
        a.numpy(),
        equal_nan=True,
    )
