import io
import math

import numpy as np
import pandas as pd
import torch

from qfeval_data import Data
from qfeval_data import Flattener


class TestFlattener:
    def test_flatten(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        f = Flattener(data.open)
        # Test if a Data object can be flattened.
        t = f.flatten(data.close)
        np.testing.assert_array_almost_equal(
            t.numpy(), np.array([100.0, 200.0, 110.0, 200.0])
        )
        # Test if the tensor can be reconstructed to a Data.
        result = f.unflatten(t * 2, "flatten_result")
        np.testing.assert_array_almost_equal(
            result.array,
            np.array(([[200.0, 400.0], [220.0, math.nan], [math.nan, 400.0]])),
        )
        assert result.columns == ["flatten_result"]
        # Test unflattening an extra dimension.
        result = f.unflatten(
            t[:, None] * torch.tensor([1, 2], dtype=t.dtype, device=t.device)
        )
        np.testing.assert_array_almost_equal(
            result.array,
            np.array(
                (
                    [
                        [[100.0, 200.0], [200.0, 400.0]],
                        [[110.0, 220.0], [math.nan, math.nan]],
                        [[math.nan, math.nan], [200.0, 400.0]],
                    ]
                )
            ),
        )
        # Test if adding many extra dimensions can be handled.
        result = f.unflatten(t[:, None, None, None].expand(-1, 3, 4, 5))
        assert result.array.shape == (
            3,
            2,
            3,
            4,
            5,
        )  # type:ignore[comparison-overlap]
        # Test if 0 dimension can be handled.
        result = f.unflatten(t[:, None][:, :0])
        assert result.array.shape == (
            3,
            2,
            0,
        )  # type:ignore[comparison-overlap]

    def create_simple_dataframe(self) -> pd.DataFrame:
        return (
            pd.read_csv(
                io.StringIO(
                    "timestamp,symbol,open,high,low,close\n"
                    + "2010-01-01,AAPL,100,120,90,100\n"
                    + "2010-01-01,GOOG,200,220,190,200\n"
                    + "2010-01-02,AAPL,110,130,100,110\n"
                    + "2010-01-03,GOOG,190,210,180,200\n"
                )
            )
            .set_index(["timestamp", "symbol"])
            .astype("float")
            .reset_index()
        )
