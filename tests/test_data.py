import io
import pickle
import random
import typing
from math import nan

import numpy as np
import pandas as pd
import pytest
import qfeval_functions
import scipy
import torch
from qfeval_functions import functions as QF

from qfeval_data import Data
from qfeval_data import util


# come from extensions/market.py
def timestamps(n: int) -> np.ndarray:
    r"""Returns a list of `n` fake timestamps."""
    start_timestamp: np.ndarray = np.array("2000-01-01", dtype="datetime64[D]")
    timestamps = start_timestamp + np.arange(n)
    return typing.cast(np.ndarray, timestamps.astype("datetime64[us]"))


# come from extensions/market.py
def symbols(n: int) -> np.ndarray:
    r"""Returns a list of `n` fake symbols."""
    ret: np.ndarray = np.array([f"X-TEST:{i:04}" for i in range(n)])
    return ret


# come from extensions/market.py
def reversal_market(
    timestamps: np.ndarray,
    symbols: np.ndarray,
    ratio: float = -0.1,
    halflife: float = 1.0,
) -> Data:
    r"""Returns OHLCV Data of a fake market having reversal nature."""
    # Calculate weights of the reversal effect.
    weight = 0.5 ** (torch.arange(-1, 101) / halflife / 24) * ratio
    weight[0] = 1.0

    # Calculate non-profitable returns.
    return_shape = (len(symbols), 1, len(timestamps) * 24 + 101)
    returns = QF.randn(*return_shape) * 0.02 * (1 / 24.0) ** 0.5

    # Calculate auto-regressive (reversal) returns.
    ar_returns = torch.nn.functional.conv1d(returns, weight[None, None, :])[
        :, 0, :
    ]

    # Calculate prices based on the AR returns.
    ar_prices = (1 + ar_returns).cumprod(dim=1)
    ar_prices = ar_prices * QF.randn(len(symbols), 1).exp() * 1000

    # Transform returns/prices from hourly data to daily data.
    daily_shape = (ar_returns.shape[0], -1, 24)
    ar_returns = ar_returns.reshape(daily_shape).transpose(0, 1)
    ar_prices = ar_prices.reshape(daily_shape).transpose(0, 1)

    # Use data from 9 a.m. to 15 p.m.
    ar_prices = ar_prices.round()[:, :, 8:15]

    # Build Data based on the information.
    return Data.from_tensors(
        {
            "open": ar_prices[:, :, 0],
            "high": ar_prices.max(dim=2).values,
            "low": ar_prices.min(dim=2).values,
            "close": ar_prices[:, :, -1],
            "volume": (
                ar_returns.var(dim=2) / ar_prices[:, :, -1] * 24 * 1e9
            ).round()
            * 100,
        },
        timestamps,
        symbols,
    )


class TestData:
    def test_from_dataframe(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        np.testing.assert_array_equal(
            data.timestamps,
            np.array(
                ["2010-01-01", "2010-01-02", "2010-01-03"], dtype=np.datetime64
            ),
        )
        np.testing.assert_array_equal(data.symbols, np.array(["AAPL", "GOOG"]))
        np.testing.assert_allclose(
            util.to_numpy(data.tensors["open"]),
            np.array([[100, 200], [110, nan], [nan, 190]], dtype="f"),
        )
        np.testing.assert_allclose(
            util.to_numpy(data.tensors["close"]),
            np.array([[100, 200], [110, nan], [nan, 200]], dtype="f"),
        )

    def test_repr(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        assert (
            repr(data)
            == r"""   timestamp symbol   open   high    low  close
0 2010-01-01   AAPL  100.0  120.0   90.0  100.0
1 2010-01-01   GOOG  200.0  220.0  190.0  200.0
2 2010-01-02   AAPL  110.0  130.0  100.0  110.0
3 2010-01-03   GOOG  190.0  210.0  180.0  200.0

[3 timestamps x 2 symbols]"""
        )
        data = Data.from_tensors(
            {
                "foo": torch.zeros((2, 3)),
                "bar": torch.zeros((2, 3, 2)),
                "baz": torch.zeros((2, 3, 1, 2)),
            },
            np.array(["2010-01-01", "2020-01-02"]),
            np.array(["A", "B", "C"]),
        )
        assert (
            repr(data)
            == r"""   timestamp symbol  foo  bar[0]  bar[1]  baz[0,0]  baz[0,1]
0 2010-01-01      A  0.0     0.0     0.0       0.0       0.0
1 2010-01-01      B  0.0     0.0     0.0       0.0       0.0
2 2010-01-01      C  0.0     0.0     0.0       0.0       0.0
3 2020-01-02      A  0.0     0.0     0.0       0.0       0.0
4 2020-01-02      B  0.0     0.0     0.0       0.0       0.0
5 2020-01-02      C  0.0     0.0     0.0       0.0       0.0

[2 timestamps x 3 symbols]"""
        )  # NOQA: W291 trailing whitespace

    def test_getitem(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        actual = data[:1, :1]
        assert actual.has_timestamps()
        assert actual.has_symbols()
        np.testing.assert_array_equal(
            actual.timestamps,
            np.array(["2010-01-01"], dtype=np.datetime64),
        )
        np.testing.assert_array_equal(
            actual.symbols,
            np.array(["AAPL"]),
        )
        actual = data[0, :1]
        assert not actual.has_timestamps()
        assert actual.has_symbols()
        np.testing.assert_array_equal(
            actual.close.array,
            [100],
        )
        actual = data[:, 1]
        assert actual.has_timestamps()
        assert not actual.has_symbols()
        actual = data[1, 0]
        assert not actual.has_timestamps()
        assert not actual.has_symbols()
        np.testing.assert_array_equal(
            actual.close.array,
            110,
        )
        actual = data[-1:, -1]
        assert actual.has_timestamps()
        assert not actual.has_symbols()
        np.testing.assert_array_equal(
            actual.close.array,
            [200],
        )
        actual = data["2010-01-02", :1]
        assert not actual.has_timestamps()
        assert actual.has_symbols()
        actual = data[:, "GOOG"]
        assert actual.has_timestamps()
        assert not actual.has_symbols()

    def test_getitem_with_boolean_ts_mask(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        # Test if a mask for timestamps works.
        actual = data[data.timestamps <= np.datetime64("2010-01-02")]
        np.testing.assert_array_equal(
            actual.timestamps,
            np.array(["2010-01-01", "2010-01-02"], dtype=np.datetime64),
        )
        # Test if a mask for symbols works.
        actual = data[:, data.symbols != "AAPL"]
        np.testing.assert_array_equal(actual.symbols, ["GOOG"])
        # Test if setting masks to both of timestamps and symbols works.
        actual = data[
            data.timestamps == np.datetime64("2010-01-02"), data.symbols > "B"
        ]
        np.testing.assert_array_equal(
            actual.timestamps, np.array(["2010-01-02"], dtype=np.datetime64)
        )
        np.testing.assert_array_equal(actual.symbols, ["GOOG"])
        # Test if torch.Tensor masks also work.
        actual = data[
            torch.tensor(data.timestamps == np.datetime64("2010-01-02")),
            torch.tensor(data.symbols > "B"),
        ]
        np.testing.assert_array_equal(
            actual.timestamps, np.array(["2010-01-02"], dtype=np.datetime64)
        )
        np.testing.assert_array_equal(actual.symbols, ["GOOG"])

    def test_getitem_with_boolean_mask(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        np.testing.assert_allclose(
            data.close[data.close > 150].array,
            [[nan, 200.0], [nan, nan], [nan, 200.0]],
        )
        np.testing.assert_allclose(
            data.close[data.close < 150].array,
            [[100.0, nan], [110.0, nan], [nan, nan]],
        )

    def test_copy_shallow(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        copy = data.copy()
        assert copy.equals(data)
        # Tensors and arrays must share memory.
        for k in data.columns:
            assert (
                copy.raw_tensors[k].data_ptr() == data.raw_tensors[k].data_ptr()
            )
        assert np.shares_memory(copy.timestamps, data.timestamps)
        assert np.shares_memory(copy.symbols, data.symbols)
        # Modifying a tensor of the copy must affect the original.
        copy.raw_tensors["open"][:] *= 2
        assert copy.equals(data)

    def test_copy_deep(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        copy = data.copy(deep=True)
        assert copy.equals(data)
        # Tensors and arrays must not share memory.
        for k in data.columns:
            assert (
                copy.raw_tensors[k].data_ptr() != data.raw_tensors[k].data_ptr()
            )
        assert not np.shares_memory(copy.timestamps, data.timestamps)
        assert not np.shares_memory(copy.symbols, data.symbols)
        # Modifying a tensor of the copy must not affect the original.
        copy.raw_tensors["open"][:] *= 2
        assert not copy.equals(data)

    def test_size(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        assert data.size(0) == 3
        assert data.size("timestamp") == 3
        assert data.size(1) == 2
        assert data.size("symbol") == 2
        assert data.size() == (3, 2)
        assert data.shape == (3, 2)
        assert data.sum().size() == (1, 1)
        assert data.sum().shape == (1, 1)
        assert data.sum("timestamp").size() == (1, 2)
        assert data.sum("symbol").size() == (3, 1)

    def test_to(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        data = data.to(torch.float32)
        assert data.close.tensor.dtype == torch.float32
        assert data.to(dtype=torch.float16).close.tensor.dtype == torch.float16
        assert (
            data.to(torch.zeros((), dtype=torch.int16)).close.tensor.dtype
            == torch.int16
        )
        assert (
            data.to(data.to(dtype=torch.float64)).close.tensor.dtype
            == torch.float64
        )

    def test_unordered_data(self) -> None:
        data = Data.from_tensors(
            {"price": torch.arange(12).resize_(3, 4)},
            np.array(["2010-01-01", "2020-01-03", "2020-01-02"]),
            np.array(["C", "A", "B", "D"]),
        )
        np.testing.assert_array_equal(
            data.tensor, [[1, 2, 0, 3], [9, 10, 8, 11], [5, 6, 4, 7]]
        )
        np.testing.assert_array_equal(
            data.timestamps,
            np.array(
                ["2010-01-01", "2020-01-02", "2020-01-03"], dtype=np.datetime64
            ),
        )
        np.testing.assert_array_equal(
            data.symbols,
            np.array(["A", "B", "C", "D"]),
        )

    def test_to_csv(self) -> None:
        expected = Data.from_tensors(
            {
                "foo": QF.randn(2, 3),
                "bar": QF.randn(2, 3, 2),
                "baz": QF.randn(2, 3, 1, 2),
            },
            np.array(["2010-01-01", "2020-01-02"]),
            np.array(["A", "B", "C"]),
        )
        actual = Data.from_csv(io.StringIO(expected.to_csv()))
        assert expected.allclose(actual)

        expected = Data.from_dataframe(self.create_simple_dataframe())
        actual = Data.from_csv(io.StringIO(expected.to_csv()))
        assert expected.allclose(actual)

    def test_get(self) -> None:
        data = Data.from_tensors(
            {
                "foo_1": QF.randn(2, 3),
                "foo_2": QF.randn(2, 3, 2),
                "bar_a": QF.randn(2, 3, 1, 2),
                "bar_b": QF.randn(2, 3, 1, 2),
            },
            np.array(["2010-01-01", "2020-01-02"]),
            np.array(["A", "B", "C"]),
        )
        assert data.get().columns == []
        assert data.get("foo_1").columns == ["foo_1"]
        assert data.get("foo_1", "bar_a").columns == ["foo_1", "bar_a"]
        assert data.get(["foo_1", "bar_a"]).columns == ["foo_1", "bar_a"]
        assert data.get(pattern="foo_*").columns == ["foo_1", "foo_2"]
        assert data.get(pattern="bar_*").columns == ["bar_a", "bar_b"]

    def test_add(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        np.testing.assert_allclose(
            (data + 1).close.array,
            np.array([[101.0, 201.0], [111.0, nan], [nan, 201.0]]),
        )
        np.testing.assert_allclose(
            (10 + data).close.array,
            np.array([[110.0, 210.0], [120.0, nan], [nan, 210.0]]),
        )

    def test_comparators(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        np.testing.assert_allclose(
            (data.open < data.close).array,
            np.array([[False, False], [False, False], [False, True]]),
        )
        np.testing.assert_allclose(
            (data.open > data.close).array,
            np.array([[False, False], [False, False], [False, False]]),
        )
        np.testing.assert_allclose(
            (data.open <= data.close).array,
            np.array([[True, True], [True, False], [False, True]]),
        )
        np.testing.assert_allclose(
            (data.open >= data.close).array,
            np.array([[True, True], [True, False], [False, False]]),
        )
        np.testing.assert_allclose(
            (data.open == data.close).array,  # type: ignore
            np.array([[True, True], [True, False], [False, False]]),
        )
        np.testing.assert_allclose(
            (data.open != data.close).array,  # type: ignore
            np.array([[False, False], [False, True], [True, True]]),
        )

    def test_apply_with_aggregation(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        data / data.std()
        data / data.std(axis=0)
        data / data.std(axis=1)

    def test_aggregate(self) -> None:
        """Test if the aggregate method works as expected."""
        data = Data.from_dataframe(self.create_simple_dataframe())
        np.testing.assert_array_almost_equal(
            data.mean(axis=1).open.array,
            data.aggregate(
                lambda x: QF.nanmean(x, dim=1, keepdim=True), axis=1
            ).open.array,
        )
        np.testing.assert_array_almost_equal(
            data.var(axis=1).open.array,
            data.aggregate(
                lambda x: QF.nanvar(x, dim=1, keepdim=True), axis=1
            ).open.array,
        )

    def test_aggregation(self) -> None:
        """Test if aggregation functions work as expected."""
        data = Data.from_dataframe(self.create_simple_dataframe())
        result = data.sum()
        np.testing.assert_allclose(
            result.open.array,
            600.0,
        )
        np.testing.assert_allclose(
            result.high.array,
            680.0,
        )
        np.testing.assert_allclose(
            result.low.array,
            560.0,
        )
        np.testing.assert_allclose(
            result.close.array,
            610.0,
        )
        np.testing.assert_allclose(
            data.sum(axis=0).open.array,
            [210.0, 390.0],
        )
        np.testing.assert_allclose(
            data.sum(axis=1).open.array,
            [300.0, 110.0, 190.0],
        )
        np.testing.assert_allclose(
            data.sum(axis=2).array,
            [[410.0, 810.0], [450.0, nan], [nan, 780.0]],
        )
        np.testing.assert_allclose(
            data.mean().open.array,
            150.0,
        )
        # NOTE: The result has low precision due to float32.
        np.testing.assert_allclose(
            data.var().open.array,
            np.var([100, 200, 110, 190], ddof=1),
            1e-5,
            1e-5,
        )
        np.testing.assert_allclose(
            data.var(ddof=0).open.array,
            np.var([100, 200, 110, 190], ddof=0),
            1e-5,
            1e-5,
        )
        np.testing.assert_allclose(
            data.std().open.array,
            np.std([100, 200, 110, 190], ddof=1),
            1e-5,
            1e-5,
        )
        np.testing.assert_allclose(
            data.skew(ddof=0).open.array,
            scipy.stats.skew([100, 200, 110, 190]),
            1e-5,
            1e-5,
        )
        np.testing.assert_allclose(
            data.kurt(ddof=0).open.array,
            scipy.stats.kurtosis([100, 200, 110, 190]),
            1e-5,
            1e-5,
        )
        np.testing.assert_array_equal(data.count().open.array, 4)
        np.testing.assert_array_equal(
            data.count(axis="symbol").open.array, [2, 1, 1]
        )

    def test_aggregation_with_nan(self) -> None:
        """Test if every aggregation function should not fail even if no valid
        values exist.
        """
        data = Data.from_dataframe(self.create_simple_dataframe())
        data = data * nan
        data.sum()
        data.mean()
        data.var()
        data.std()
        data.count()

    def test_to_dataframe(self) -> None:
        expected_df = self.create_simple_dataframe()
        data = Data.from_dataframe(expected_df)
        actual_df = data.to_dataframe()
        assert actual_df.to_csv() == expected_df.to_csv()

    def test_to_matrix(self) -> None:
        csv = """timestamp,symbol,return,weight
        2021-03-30,A,,0.01
        2021-03-30,B,,0.02
        2021-03-30,C,,0.03
        2021-03-31,A,,0.04
        2021-03-31,B,,0.05
        2021-03-31,C,,0.06
        """
        expected = (
            ",A,B,C\n2021-03-30,0.01,0.02,0.03\n2021-03-31,0.04,0.05,0.06\n"
        )
        df = pd.read_csv(io.StringIO(csv))
        data = Data.from_dataframe(df)
        actual_df = data.weight.to_matrix()
        assert actual_df.to_csv(line_terminator="\n") == expected

    def test_missing_to_matrix(self) -> None:
        csv = """timestamp,symbol,return,weight
        2021-03-30,A,,0.01
        2021-03-30,C,,0.03
        2021-03-31,B,,0.05
        """
        expected = ",A,B,C\n2021-03-30,0.01,,0.03\n2021-03-31,,0.05,\n"
        df = pd.read_csv(io.StringIO(csv))
        data = Data.from_dataframe(df)
        actual_df = data.weight.to_matrix()
        assert actual_df.to_csv(line_terminator="\n") == expected

    def test_rename(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        np.testing.assert_array_equal(
            data.rename(["Open", "High", "Low", "Close"]).columns,
            ["Open", "High", "Low", "Close"],
        )
        np.testing.assert_array_equal(
            data.close.rename("price").columns, ["price"]
        )
        np.testing.assert_array_equal(
            data.rename({"high": "H", "low": "L"}).columns,
            ["open", "H", "L", "close"],
        )
        with pytest.raises(KeyError):
            data.rename({"volume": "Volume"})
        with pytest.raises(ValueError):
            data.rename(["price"])

    def test_merge(self) -> None:
        qfeval_functions.random.seed()
        expected_df = self.create_simple_dataframe()
        data = Data.from_dataframe(expected_df)
        np.testing.assert_allclose(
            data[:2].merge(data[1:]).close.array, data.close.array
        )

    def test_merge_with_random_values(self) -> None:
        timestamps: np.ndarray = np.array(
            np.datetime64("2000-01-01") + np.arange(5), dtype="datetime64[us]"
        )
        symbols: np.ndarray = np.array([f"S{i}" for i in range(5)], dtype=str)

        # Test if Data.merge can merge single elements.
        expected = np.full((len(timestamps), len(symbols)), nan)
        values = []
        for _ in range(100):
            timestamp_index = random.randrange(len(timestamps))
            symbol_index = random.randrange(len(symbols))
            value = random.random()
            expected[timestamp_index, symbol_index] = value
            row = {
                "timestamp": timestamps[timestamp_index],
                "symbol": symbols[symbol_index],
                "value": value,
            }
            values.append(Data.from_dataframe(pd.DataFrame([row])))
        actual = values[0].merge(*values[1:])
        np.testing.assert_array_equal(actual.timestamps, timestamps)
        np.testing.assert_array_equal(actual.symbols, symbols)
        np.testing.assert_allclose(actual.array, expected)

        # Test if Data.merge can merge composite Data objects.
        expected = np.full((len(timestamps), len(symbols)), nan)
        values = []
        for _ in range(10):
            children = []
            for _ in range(10):
                timestamp_index = random.randrange(len(timestamps))
                symbol_index = random.randrange(len(symbols))
                value = random.random()
                expected[timestamp_index, symbol_index] = value
                row = {
                    "timestamp": timestamps[timestamp_index],
                    "symbol": symbols[symbol_index],
                    "value": value,
                }
                children.append(Data.from_dataframe(pd.DataFrame([row])))
            values.append(children[0].merge(*children[1:]))
        actual = values[0].merge(*values[1:])
        np.testing.assert_array_equal(actual.timestamps, timestamps)
        np.testing.assert_array_equal(actual.symbols, symbols)
        np.testing.assert_allclose(actual.array, expected)

    def test_dropna(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        np.testing.assert_array_equal(
            data.dropna(axis=0).timestamps,
            data[["2010-01-01"]].timestamps,
        )
        np.testing.assert_array_equal(
            data.dropna(axis=1).symbols, data.symbols[:0]
        )

    def test_fillna(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        np.testing.assert_array_equal(
            data.fillna(12345.0).close.array,
            np.array([[100.0, 200.0], [110.0, 12345.0], [12345.0, 200.0]]),
        )
        np.testing.assert_array_equal(
            data.fillna(method="ffill").close.array,
            np.array([[100.0, 200.0], [110.0, 200.0], [110.0, 200.0]]),
        )
        np.testing.assert_array_equal(
            data.fillna(method="bfill").close.array,
            np.array([[100.0, 200.0], [110.0, 200.0], [nan, 200.0]]),
        )

    def test_pct_change(self) -> None:
        data = reversal_market(
            timestamps(10),
            symbols(20),
        )
        np.testing.assert_allclose(
            data.pct_change().close.array,
            pd.DataFrame(data.close.array).pct_change().to_numpy(),
        )
        np.testing.assert_allclose(
            data.pct_change(3).close.array,
            pd.DataFrame(data.close.array).pct_change(3).to_numpy(),
        )

    def test_diff(self) -> None:
        data = reversal_market(
            timestamps(10),
            symbols(20),
        )
        np.testing.assert_allclose(
            data.diff().close.array,
            pd.DataFrame(data.close.array).diff().to_numpy(),
        )
        np.testing.assert_allclose(
            data.diff(3).close.array,
            pd.DataFrame(data.close.array).diff(3).to_numpy(),
        )
        data = Data.from_dataframe(
            pd.read_csv(
                io.StringIO(
                    "timestamp,symbol,close\n"
                    + "2018-01-01,AAPL,100\n"
                    + "2018-01-02,AAPL,200\n"
                    + "2018-01-03,AAPL,\n"
                    + "2018-01-04,AAPL,100\n"
                    + "2018-01-05,AAPL,200\n"
                )
            )
            .set_index(["timestamp", "symbol"])
            .astype("float32")
            .reset_index()
        )
        np.testing.assert_allclose(
            data.diff().close[:, "AAPL"].array,
            [nan, 100.0, nan, nan, 100.0],
        )
        np.testing.assert_allclose(
            data.diff(skipna=True).close[:, "AAPL"].array,
            [nan, 100.0, nan, -100.0, 100.0],
        )

    def test_cumsum(self) -> None:
        data = reversal_market(
            timestamps(10),
            symbols(20),
        )
        np.testing.assert_allclose(
            data.cumsum().close.array,
            pd.DataFrame(data.close.array).cumsum().to_numpy(),
        )

    def test_cumprod(self) -> None:
        data = (
            reversal_market(
                timestamps(10),
                symbols(20),
            ).pct_change()
            + 1
        )
        np.testing.assert_allclose(
            data.cumprod().close.array,
            pd.DataFrame(data.close.array).cumprod().to_numpy(),
            rtol=1e-06,
        )

    def test_subsequences(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        subsequences = data.subsequences(-1, 1)
        np.testing.assert_allclose(
            subsequences.close.array,
            np.array(
                [
                    [[nan, 100], [nan, 200]],
                    [[100, 110], [200, nan]],
                    [[110, nan], [nan, 200.0]],
                ]
            ),
        )

    def test_randseqs(self) -> None:
        qfeval_functions.random.seed()
        tensor = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16],
                [nan, 21, 22, 23, 24, 25, 26],
                [30, 31, 32, 33, nan, 35, nan],
            ]
        ).t()
        data = Data.from_tensors(
            {"close": tensor},
            np.datetime64("2000-01-01") + np.arange(tensor.size(0)),
            np.array([f"X:{i:05d}" for i in range(tensor.size(1))]),
        )
        # Generate all subsequences w/o NaNs.
        seqs = next(data.randseqs(1000, 4, skipna=True))
        orderd_seqs = sorted(
            [
                (tuple(tensor), int(ts), int(sym))
                for tensor, ts, sym in zip(
                    seqs.tensor, seqs.timestamps, seqs.symbols
                )
            ]
        )
        np.testing.assert_allclose(
            [s[0] for s in orderd_seqs],
            [
                (10, 11, 12, 13),
                (11, 12, 13, 14),
                (12, 13, 14, 15),
                (13, 14, 15, 16),
                (21, 22, 23, 24),
                (22, 23, 24, 25),
                (23, 24, 25, 26),
                (30, 31, 32, 33),
            ],
        )
        np.testing.assert_allclose(
            [(s[1], s[2]) for s in orderd_seqs],
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (1, 1),
                (2, 1),
                (3, 1),
                (0, 2),
            ],
        )
        # Generate all subsequences w/ NaNs.
        seqs = next(data.randseqs(1000, 4, skipna=False))
        # NOTE: Replace NaNs with 0s for sorting.
        orderd_seqs = sorted(
            [
                (tuple(QF.fillna(tensor)), int(ts), int(sym))
                for tensor, ts, sym in zip(
                    seqs.tensor, seqs.timestamps, seqs.symbols
                )
            ]
        )
        np.testing.assert_allclose(
            [s[0] for s in orderd_seqs],
            [
                (0, 21, 22, 23),
                (10, 11, 12, 13),
                (11, 12, 13, 14),
                (12, 13, 14, 15),
                (13, 14, 15, 16),
                (21, 22, 23, 24),
                (22, 23, 24, 25),
                (23, 24, 25, 26),
                (30, 31, 32, 33),
                (31, 32, 33, 0),
                (32, 33, 0, 35),
                (33, 0, 35, 0),
            ],
        )
        np.testing.assert_allclose(
            [(s[1], s[2]) for s in orderd_seqs],
            [
                (0, 1),
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (1, 1),
                (2, 1),
                (3, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (3, 2),
            ],
        )
        # Random two subsequences should not match.
        assert not torch.equal(
            next(data.randseqs(1000, 4)).tensor,
            next(data.randseqs(1000, 4)).tensor,
        )
        # Test how randseqs deal with batch_shape.
        np.testing.assert_equal(
            [s.tensor.size(0) for s in data.randseqs(2, 4)], (2, 2, 2, 2)
        )
        np.testing.assert_equal(
            [s.tensor.size(0) for s in data.randseqs(3, 4)], (3, 3, 2)
        )

    def test_downsample(self) -> None:
        data = Data.from_dataframe(
            pd.read_csv(
                io.StringIO(
                    "timestamp,symbol,open,high,low,close\n"
                    + "2018-01-02,AAPL,154,162,154,160\n"
                    + "2018-01-03,AAPL,159,161,154,154\n"
                    + "2018-01-04,AAPL,154,161,154,159\n"
                    + "2018-01-05,AAPL,159,163,157,163\n"
                    + "2018-01-08,AAPL,161,170,161,169\n"
                    + "2018-01-09,AAPL,168,173,164,168\n"
                    + "2018-01-10,AAPL,167,177,167,173\n"
                    + "2018-01-11,AAPL,173,173,160,168\n"
                    + "2018-01-12,AAPL,167,170,162,162\n"
                    + "2018-01-15,AAPL,162,163,154,159\n"
                    + "2018-01-16,AAPL,160,170,159,169\n"
                    + "2018-01-17,AAPL,168,170,163,168\n"
                    + "2018-01-18,AAPL,169,171,164,165\n"
                    + "2018-01-19,AAPL,164,173,160,169\n"
                )
            )
            .set_index(["timestamp", "symbol"])
            .astype("float32")
            .reset_index()
        )
        assert (
            data.downsample(
                np.timedelta64(2, "D"), origin=np.datetime64("2018-01-01")
            )
            .to_dataframe()
            .to_csv(index=False, line_terminator="\n")
            == "timestamp,symbol,open,high,low,close\n"
            + "2018-01-01,AAPL,154.0,162.0,154.0,160.0\n"
            + "2018-01-03,AAPL,159.0,161.0,154.0,159.0\n"
            + "2018-01-05,AAPL,159.0,163.0,157.0,163.0\n"
            + "2018-01-07,AAPL,161.0,170.0,161.0,169.0\n"
            + "2018-01-09,AAPL,168.0,177.0,164.0,173.0\n"
            + "2018-01-11,AAPL,173.0,173.0,160.0,162.0\n"
            + "2018-01-15,AAPL,162.0,170.0,154.0,169.0\n"
            + "2018-01-17,AAPL,168.0,171.0,163.0,165.0\n"
            + "2018-01-19,AAPL,164.0,173.0,160.0,169.0\n"
        )
        assert (
            data.weekly(origin=np.datetime64("2018-01-01"))
            .to_dataframe()
            .to_csv(index=False, line_terminator="\n")
            == "timestamp,symbol,open,high,low,close\n"
            + "2018-01-01,AAPL,154.0,163.0,154.0,163.0\n"
            + "2018-01-08,AAPL,161.0,177.0,160.0,162.0\n"
            + "2018-01-15,AAPL,162.0,173.0,154.0,169.0\n"
        )

    def test_group_shift(self) -> None:
        qfeval_functions.random.seed()
        shape = (10, 5)
        cols = ["open", "close"]

        # Data with randomly missing entries
        tensors = {
            k: torch.where(
                torch.rand(shape) <= 0.9,
                0.1 * torch.randn(shape).float(),
                torch.tensor(nan).float(),
            )
            for k in cols
        }
        # missing data patterns
        true_patterns = {k: v.isnan().any(dim=1) for k, v in tensors.items()}

        data = Data.from_tensors(
            tensors,
            timestamps(shape[0]),
            symbols(shape[1]),
        )

        # The following two should be the same:
        # - First applying group_shift and then extracting non-missing values
        # - First extracting non-missing values and then applying (usual) shift
        np.testing.assert_allclose(
            data.group_shift(1)
            .get("open")
            .tensor[~true_patterns["open"]]
            .numpy(),
            QF.shift(
                data[~true_patterns["open"]].get("open").tensor, 1, 0
            ).numpy(),
        )
        np.testing.assert_allclose(
            data.group_shift(10)
            .get("open")
            .tensor[~true_patterns["open"]]
            .numpy(),
            QF.shift(
                data[~true_patterns["open"]].get("open").tensor, 10, 0
            ).numpy(),
        )
        np.testing.assert_allclose(
            data.group_shift(-1)
            .get("close")
            .tensor[~true_patterns["close"]]
            .numpy(),
            QF.shift(
                data[~true_patterns["close"]].get("close").tensor, -1, 0
            ).numpy(),
        )
        np.testing.assert_allclose(
            data.group_shift(-5)
            .get("close")
            .tensor[~true_patterns["close"]]
            .numpy(),
            QF.shift(
                data[~true_patterns["close"]].get("close").tensor, -5, 0
            ).numpy(),
        )

        # Data with no missing entries
        weight = Data.from_tensors(
            {k: torch.ones(shape) for k in cols},
            timestamps(shape[0]),
            symbols(shape[1]),
        )

        # If reference is set, the pattern of missing values should be
        # inherited by the reference data
        np.testing.assert_array_equal(
            weight.group_shift(1, reference=data)
            .get("open")
            .tensor.isnan()
            .numpy(),
            data.group_shift(1).get("open").tensor.isnan().numpy(),
        )

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
            .astype("float32")
            .reset_index()
        )

    def test_cpu(self) -> None:
        data = Data.from_dataframe(self.create_simple_dataframe())
        assert str(data.close.tensor.device) == "cpu"

    @pytest.mark.gpu
    def test_gpu(self) -> None:
        data = Data.from_dataframe(
            self.create_simple_dataframe(), device="cuda:0"
        )
        assert str(data.close.tensor.device) == "cuda:0"

    def test_auto(self) -> None:
        data = Data.from_dataframe(
            self.create_simple_dataframe(), device="auto"
        )
        assert str(data.close.tensor.device) in ("cpu", "cuda:0")

    @pytest.mark.gpu
    def test_auto_gpu(self) -> None:
        data = Data.from_dataframe(
            self.create_simple_dataframe(), device="auto"
        )
        assert str(data.close.tensor.device) == "cuda:0"

    def test_pickle(self) -> None:
        data = Data.from_dataframe(
            self.create_simple_dataframe(), device="auto"
        )
        pickled = pickle.dumps(data)
        unpickled = pickle.loads(pickled)
        np.testing.assert_equal(data.timestamps, unpickled.timestamps)
        np.testing.assert_equal(data.symbols, unpickled.symbols)
        np.testing.assert_equal(data.columns, unpickled.columns)
        for column in data.columns:
            torch.testing.assert_close(
                data.get(column).tensor,
                unpickled.get(column).tensor,
                atol=0,
                rtol=0,
                equal_nan=True,
            )
