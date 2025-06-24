################################################################################
# Imports of standard libraries
################################################################################

from __future__ import annotations

import fnmatch
import io
import itertools
import logging
import math
import os
import re
import sys
import typing
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib
    from matplotlib import pyplot as plt
except ImportError:
    matplotlib = None  # type:ignore[assignment]
    plt = None  # type:ignore[assignment]


from qfeval_functions import functions

from . import plot
from . import util

logger = logging.getLogger(__name__)


def _define_binary_operator(
    f: typing.Callable[..., typing.Any],
) -> typing.Callable[[Data, typing.Any], Data]:
    def op(self: Data, other: typing.Any) -> Data:
        def g(x: typing.Any, y: typing.Any) -> torch.Tensor:
            return typing.cast(torch.Tensor, f(x, y))

        return self.apply(g, other)

    return op


def _define_unary_operator(
    f: typing.Callable[..., typing.Any],
) -> typing.Callable[[Data], Data]:
    def op(self: Data) -> Data:
        def g(x: typing.Any) -> torch.Tensor:
            return typing.cast(torch.Tensor, f(x))

        return self.apply(g)

    return op


Axis = typing.Union[
    int,
    typing.Literal["timestamp"],
    typing.Literal["symbol"],
    typing.Literal["column"],
]


# TODO(imos): Re-enable @typechecked once some workaround is found.
# @typechecked
class Data(object):
    """Manages numerical tensors, each of which is indexed by a timestamp
    (np.datetime64) and a symbol (str).  Tensors should represent numerical
    information like OHLC (open, high, low, close).  It can slice tensors in a
    low computational cost via the [] operator (i.e., Tensors can also be
    accessible without transfering from the main memory).  Tensors are managed
    as a map from a string name (column name) to a multi-dimensional tensor.
    The tensor should have two or more dimensions (i.e., a feature would have
    other dimensions in addition to a timestamp and a symbol).  The tensor's
    leading two dimensions corresponds to timestamps and symbols repspectively.
    """

    ############################################################################
    # Initialization and builder methods
    ############################################################################

    def __init__(self, data: Data):
        """Initializes a new Data object.  Use from_tensors when creating a new
        Data object from tensors."""
        super().__init__()
        assert isinstance(data, Data)
        self.__timestamps: np.ndarray = data.__timestamps
        self.__symbols: np.ndarray = data.__symbols
        self.__tensors: typing.Dict[str, torch.Tensor] = data.__tensors

    @classmethod
    def from_preset(
        cls,
        name: str = "pfn-topix500",
        dtype: typing.Any = None,
        device: typing.Any = None,
        paths: typing.List[str] = [],
    ) -> Data:
        # TODO(masanori): check compatibility with actual data path
        paths += sys.path
        for path in paths:
            f = os.path.join(path, "data", f"{name}.csv")
            if os.path.exists(f):
                return cls.from_csv(f, dtype=dtype, device=device)
            f = os.path.join(path, "data", f"{name}.csv.xz")
            if os.path.exists(f):
                return cls.from_csv(f, dtype=dtype, device=device)
        raise FileNotFoundError(f"No such preset: {name}")

    @classmethod
    def from_csv(
        cls,
        input: typing.Union[str, io.IOBase],
        dtype: typing.Any = None,
        device: typing.Any = None,
    ) -> Data:
        logger.debug(f"Reading CSV data from: {input}")
        return cls.from_dataframe(
            pd.read_csv(input)
            .set_index(["timestamp", "symbol"])
            .astype(torch.zeros((), dtype=dtype).numpy().dtype)
            .reset_index(),
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        dtype: typing.Any = None,
        device: typing.Any = None,
    ) -> Data:
        """Builds and returns Data based on the given DataFrame object.  The
        DataFrame object must have timestamp and symbol columns.
        """
        # 1. Preprocess the given data frame.  This removes duplicates if any
        # because df.pivot fails when it has duplicates.
        orig_shape = df.shape
        df = df.drop_duplicates(("timestamp", "symbol"), keep="last")
        if df.shape != orig_shape:
            logger.warning(
                "Removed duplicates in the given DataFrame: "
                f"{orig_shape} => {df.shape}"
            )
        # 2. Build NumPy arrays, each of which represents exactly one column of
        # the source DataFrame.
        arrays: typing.Dict[
            str,
            typing.Union[
                np.ndarray, typing.Dict[typing.Tuple[int, ...], np.ndarray]
            ],
        ] = {}
        for column in df.columns:
            if column in ("timestamp", "symbol"):
                continue
            m = re.match(r"^([^\[\]]*)(?:\[(.*)\])?$", column)
            if m is None:
                raise ValueError(f"Invalid column: {column}")
            name, index = typing.cast(
                typing.Tuple[str, typing.Optional[str]], m.groups()
            )
            table = df.pivot(index="timestamp", columns="symbol", values=column)
            value = table.values
            if index is None:
                arrays[name] = value
            else:
                arrays.setdefault(name, {})
                arrays[name][tuple(map(int, index.split(",")))] = value

        # 3. Unite multi-dimensional columns.
        for k, v in arrays.items():
            if not isinstance(v, dict):
                continue
            items = list(v.items())
            shape = list(items[0][0])
            for item in items:
                for dim, size in enumerate(item[0]):
                    shape[dim] = max(shape[dim], size + 1)
            value = np.full(
                items[0][1].shape + tuple(shape),
                math.nan,
                dtype=items[0][1].dtype,
            )
            for item in items:
                value[(slice(None), slice(None)) + item[0]] = item[1]
            arrays[k] = value

        # 4. Build Data.
        torch_device = util.torch_device(device)
        tensors = {
            k: torch.tensor(v, dtype=dtype, device=torch_device)
            for k, v in arrays.items()
        }
        timestamps: np.ndarray = np.array(table.index, dtype=np.datetime64)
        symbols: np.ndarray = np.array(table.columns, dtype=np.str_)
        return cls.from_tensors(tensors, timestamps, symbols)

    @classmethod
    def from_tensors(
        cls,
        tensors: typing.Dict[str, torch.Tensor],
        timestamps: np.ndarray,
        symbols: np.ndarray,
    ) -> Data:
        """Returns Data with the given tensors.  This is the most primitive
        builder of Data class, and the other builders should not access internal
        properties without calling Data.from_tensors.
        """
        assert len(timestamps.shape) == 1
        assert len(symbols.shape) == 1  # type:ignore[unreachable]
        assert isinstance(tensors, dict)

        result: Data = object.__new__(cls)

        timestamps_index = (
            slice(None)
            if np.array_equal(timestamps, np.sort(timestamps))
            else np.argsort(timestamps)
        )
        timestamps = timestamps[timestamps_index]  # type: ignore
        result.__timestamps = timestamps.astype("datetime64", copy=False)

        symbols_index = (
            slice(None)
            if np.array_equal(symbols, np.sort(symbols))
            else np.argsort(symbols)
        )
        symbols = symbols[symbols_index]  # type: ignore
        result.__symbols = symbols.astype(np.str_, copy=False)

        device: typing.Optional[torch.device] = None
        for k, v in tensors.items():
            assert isinstance(k, str)
            assert isinstance(v, torch.Tensor)
            assert len(v.shape) >= 2
            assert v.shape[0] == timestamps.shape[0]
            assert v.shape[1] == symbols.shape[0]
            if device is None:
                device = v.device
            else:
                assert v.device == device
            tensors[k] = tensors[k][timestamps_index, :][:, symbols_index]  # type: ignore # TODO(masanori): fix type error
        result.__tensors = tensors

        return result

    ############################################################################
    # Python special methods
    ############################################################################

    def __repr__(self) -> str:
        """Returns the Data's summary."""
        return (
            repr(self.to_dataframe())
            + f"\n\n[{self.__timestamps.shape[0]} timestamps "
            + f"x {self.__symbols.shape[0]} symbols]"
        )

    def __getitem__(self, key: typing.Any) -> Data:
        """Returns the Data's slice specified by the given index(es)."""
        # Get items by a boolean mask.
        if isinstance(key, Data):
            assert key.dtype == torch.bool
            assert self.dtype.is_floating_point
            mask = key.like(self).raw_tensor
            return self.from_tensors(
                {
                    k: torch.where(
                        mask,
                        v,
                        torch.as_tensor(
                            math.nan, dtype=v.dtype, device=v.device
                        ),
                    )
                    for k, v in self.__tensors.items()
                },
                self.__timestamps,
                self.__symbols,
            )

        if isinstance(key, tuple):
            assert len(key) == 2
            timestamp_index, symbol_index = key
        else:
            timestamp_index, symbol_index = key, slice(None)

        if timestamp_index is None or symbol_index is None:
            raise KeyError("Data cannot be indexed by None")

        timestamp_index, collapse_timestamp = _to_index(
            timestamp_index, self.timestamp_index
        )
        symbol_index, collapse_symbol = _to_index(
            symbol_index, self.symbol_index
        )

        result = self.from_tensors(
            {
                k: v[timestamp_index][:, symbol_index]
                for k, v in self.__tensors.items()
            },
            self.__timestamps[timestamp_index],
            self.__symbols[symbol_index],
        )
        if collapse_timestamp:
            result = result.sum(axis="timestamp")
        if collapse_symbol:
            result = result.sum(axis="symbol")
        return result

    def __getattr__(self, key: str) -> Data:
        try:
            return self.get(key)
        except KeyError:
            raise AttributeError(key)

    def __float__(self) -> float:
        return float(self.tensor)

    def __getstate__(self) -> typing.Dict[str, typing.Any]:
        # We explicitly define __getstate__ and __setstate__ to avoid a maximum
        # recursion error caused by self.__getattr__ and self.get.

        # If we do not define __getstate__ and __setstate__, self.__getattr__
        # is called when unpickling a Data object because its attributes such
        # as self.__tensors are not set. This causes a maximum recursion error
        # because self.__getattr__ calls self.get, which tries to access
        # attributes, which again calls self.__getattr__.
        return {
            "timestamps": self.__timestamps,
            "symbols": self.__symbols,
            "tensors": self.__tensors,
        }

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
        self.__timestamps = state["timestamps"]
        self.__symbols = state["symbols"]
        self.__tensors = state["tensors"]

    ############################################################################
    # Python operators
    ############################################################################

    eq = _define_binary_operator(lambda x, y: x == y)
    ne = _define_binary_operator(lambda x, y: x != y)

    __add__ = _define_binary_operator(lambda x, y: x + y)
    __radd__ = _define_binary_operator(lambda x, y: y + x)
    __sub__ = _define_binary_operator(lambda x, y: x - y)
    __rsub__ = _define_binary_operator(lambda x, y: y - x)
    __mul__ = _define_binary_operator(lambda x, y: x * y)
    __rmul__ = _define_binary_operator(lambda x, y: y * x)
    __matmul__ = _define_binary_operator(lambda x, y: x @ y)
    __rmatmul__ = _define_binary_operator(lambda x, y: y @ x)
    __truediv__ = _define_binary_operator(lambda x, y: x / y)
    __rtruediv__ = _define_binary_operator(lambda x, y: y / x)
    __floordiv__ = _define_binary_operator(lambda x, y: x // y)
    __rfloordiv__ = _define_binary_operator(lambda x, y: y // x)
    __mod__ = _define_binary_operator(lambda x, y: x % y)
    __rmod__ = _define_binary_operator(lambda x, y: y % x)
    __pow__ = _define_binary_operator(lambda x, y: x**y)
    __rpow__ = _define_binary_operator(lambda x, y: y**x)
    __lshift__ = _define_binary_operator(lambda x, y: x << y)
    __rlshift__ = _define_binary_operator(lambda x, y: y << x)
    __rshift__ = _define_binary_operator(lambda x, y: x >> y)
    __rrshift__ = _define_binary_operator(lambda x, y: y >> x)
    __eq__ = eq  # type: ignore
    __ne__ = ne  # type: ignore
    __gt__ = _define_binary_operator(lambda x, y: x > y)
    __lt__ = _define_binary_operator(lambda x, y: x < y)
    __ge__ = _define_binary_operator(lambda x, y: x >= y)
    __le__ = _define_binary_operator(lambda x, y: x <= y)
    __and__ = _define_binary_operator(lambda x, y: x & y)
    __or__ = _define_binary_operator(lambda x, y: x | y)
    __xor__ = _define_binary_operator(lambda x, y: x ^ y)
    __invert__ = _define_unary_operator(lambda x: ~x)
    __neg__ = _define_unary_operator(lambda x: -x)
    __pos__ = _define_unary_operator(lambda x: +x)
    __abs__ = _define_unary_operator(abs)

    ############################################################################
    # Properties
    ############################################################################

    @property
    def raw_tensors(self) -> typing.Dict[str, torch.Tensor]:
        return self.__tensors

    @property
    def raw_tensor(self) -> torch.Tensor:
        if len(self.raw_tensors) != 1:
            raise RuntimeError(
                "Data.raw_tensor can be used only when Data has exactly one "
                + f"tensor, but it has {len(self.raw_tensors)} tensors."
            )
        return next(iter(self.raw_tensors.values()))

    @property
    def tensors(self) -> typing.Dict[str, torch.Tensor]:
        return {
            k: v[self.__index_slices()] for k, v in self.raw_tensors.items()
        }

    @property
    def tensor(self) -> torch.Tensor:
        return self.raw_tensor[self.__index_slices()]

    @property
    def arrays(self) -> typing.Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.tensors.items()}

    @property
    def array(self) -> np.ndarray:
        return typing.cast(np.ndarray, self.tensor.detach().cpu().numpy())

    @property
    def timestamps(self) -> np.ndarray:
        if self.has_timestamps():
            return self.__timestamps
        raise ValueError("Data does not have valid timestamps")

    @timestamps.setter
    def timestamps(self, timestamps: np.ndarray) -> None:
        if self.__timestamps.shape != timestamps.shape:
            raise ValueError(
                f"Inconsistent shape: expected={self.__timestamps.shape}, "
                + f"actual={self.timestamps.shape}"
            )
        assert np.array_equal(timestamps, np.sort(timestamps))
        self.__timestamps = timestamps.astype("datetime64", copy=False)

    @property
    def symbols(self) -> np.ndarray:
        if self.has_symbols():
            return self.__symbols
        raise ValueError("Data does not have valid symbols")

    @symbols.setter
    def symbols(self, symbols: np.ndarray) -> None:
        if self.__symbols.shape != symbols.shape:
            raise ValueError(
                f"Inconsistent shape: expected={self.__symbols.shape}, "
                + f"actual={self.symbols.shape}"
            )
        assert np.array_equal(symbols, np.sort(symbols)), symbols
        self.__symbols = symbols.astype(np.str_, copy=False)

    @property
    def columns(self) -> typing.List[str]:
        return list(self.__tensors.keys())

    @property
    def shape(self) -> typing.Tuple[int, int]:
        r"""Returns a 2-dimensional tuple representing the shape:
        (timestamp, symbol).  Aggregated dimensions will be 1.
        """
        return self.size()

    @property
    def device(self) -> torch.device:
        for t in self.__tensors.values():
            return t.device
        raise RuntimeError("No tensors are stored")

    @property
    def dtype(self) -> torch.dtype:
        for t in self.__tensors.values():
            return t.dtype
        raise RuntimeError("No tensors are stored")

    ############################################################################
    # Public methods
    ############################################################################

    @typing.overload
    def get(self, *columns: str) -> Data:
        pass

    @typing.overload
    def get(self, arg: typing.Iterable[str]) -> Data:
        pass

    @typing.overload
    def get(self, arg: typing.Callable[[str], bool]) -> Data:
        pass

    @typing.overload
    def get(self, *, pattern: typing.Optional[str] = None) -> Data:
        pass

    def get(
        self,
        arg: typing.Any = None,
        *args: typing.Any,
        pattern: typing.Optional[str] = None,
    ) -> Data:
        """Returns a subset of columns as a new Data object."""

        if arg is not None:
            args = (arg,) + args

        if pattern is not None:
            assert len(args) == 0
            # NOTE: This ensure the type of a variable to be captured for mypy.
            # mypy does not deduce the narrow type of a captured variable.
            pattern_str = pattern
            return self.get(lambda x: fnmatch.fnmatch(x, pattern_str))

        columns = []
        for arg in args:
            if isinstance(arg, str):
                columns.append(arg)
            elif callable(arg):
                assert len(args) == 1
                for column in self.columns:
                    if arg(column):
                        columns.append(column)
            else:
                assert len(args) == 1
                for x in arg:
                    columns.append(x)

        tensors = {}
        for column in columns:
            tensors[column] = self.__tensors[column]
        return self.from_tensors(tensors, self.__timestamps, self.__symbols)

    def set(self, key: str, value: typing.Union[torch.Tensor, Data]) -> None:
        """Sets a column with the given name and value.  If the name already
        exists, this replaces the column.  Otherwise, this appends the value as
        a new column."""
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, Data):
            assert np.array_equal(self.__timestamps, value.__timestamps)
            assert np.array_equal(self.__symbols, value.__symbols)
            tensor = value.tensor
        else:
            raise TypeError(f"Unsupported type: {value.__class__.__name__}")
        assert len(tensor.shape) >= 2
        assert tensor.shape[:2] == self.shape
        if len(self.__tensors) > 0:
            assert tensor.dtype == self.dtype
            assert tensor.device == self.device
        self.__tensors[key] = tensor

    def copy(self, deep: bool = False) -> Data:
        """Returns a copy of itself.

        By default, this is not a deep copy, so it prevents from changing a set
        of columns but does not prevent from updating their tensors.
        Set deep=True if you need a deep copy.

        Args:
            deep (bool): Make a deep copy if set to True.

        Returns:
            Data: A copy.
        """
        if deep:
            return self.from_tensors(
                {k: v.clone() for k, v in self.__tensors.items()},
                self.__timestamps.copy(),
                self.__symbols.copy(),
            )
        else:
            return self.from_tensors(
                self.__tensors.copy(), self.__timestamps, self.__symbols
            )

    @typing.overload
    def size(self, dim: None = None) -> typing.Tuple[int, int]:
        pass

    @typing.overload
    def size(self, dim: Axis) -> int:
        pass

    def size(
        self, dim: typing.Optional[Axis] = None
    ) -> typing.Union[typing.Tuple[int, int], int]:
        r"""Returns the size of the corresponding dimension if `dim` is given.
        If no dimension is specified, this returns a 2-dimensional tuple
        representing the shape: (timestamp, symbol).

        It returns 1 for an aggregated dimension (i.e.,
        `Data.sum("symbol").size("symbol")` should always return 1).
        """
        if dim is None:
            return self.size(0), self.size(1)
        dim_int = _parse_axis(dim)
        if dim_int == 0:
            return self.__timestamps.size
        elif dim_int == 1:
            return self.__symbols.size
        raise ValueError(f"Data.size got an unexpected dimension: {dim}")

    @typing.overload
    def to(self, dtype: torch.dtype) -> Data:
        pass

    @typing.overload
    def to(
        self, device: torch.device, dtype: typing.Optional[torch.dtype] = None
    ) -> Data:
        pass

    @typing.overload
    def to(self, tensor: torch.Tensor) -> Data:
        pass

    @typing.overload
    def to(self, data: Data) -> Data:
        pass

    def to(self, *args: typing.Any, **kwargs: typing.Any) -> Data:
        r"""Converts dtype and/or device of the tensors."""
        for arg in args:
            if isinstance(arg, Data):
                return self.to(next(iter(arg.raw_tensors.values())))
        return self.from_tensors(
            {k: v.to(*args, **kwargs) for k, v in self.raw_tensors.items()},
            self.__timestamps,
            self.__symbols,
        )

    def has_timestamps(self) -> bool:
        """Returns true iff the Data has timestamps."""
        return self.__timestamps.shape != (
            1,
        ) or not np.array_equal(  # type:ignore[comparison-overlap]
            self.__timestamps, self.__invalid_timestamp()[None]
        )

    def has_symbols(self) -> bool:
        """Returns true iff the Data has symbols."""
        return self.__symbols.shape != (
            1,
        ) or not np.array_equal(  # type:ignore[comparison-overlap]
            self.__symbols, self.__invalid_symbol()[None]
        )

    def equals(self, other: Data) -> bool:
        r"""Returns true if the Data equals to the given Data exactly."""
        if not np.array_equal(self.__timestamps, other.__timestamps):
            return False
        if not np.array_equal(self.__symbols, other.__symbols):
            return False
        if self.columns != other.columns:
            return False
        for k, v in self.__tensors.items():
            if not torch.allclose(
                v, other.__tensors[k], rtol=0, atol=0, equal_nan=True
            ):
                return False
        return True

    def allclose(
        self, other: Data, rtol: float = 1e-05, atol: float = 1e-08
    ) -> bool:
        r"""Returns true if the Data equals to the given Data exactly."""
        if not np.array_equal(self.__timestamps, other.__timestamps):
            return False
        if not np.array_equal(self.__symbols, other.__symbols):
            return False
        if self.columns != other.columns:
            return False
        for k, v in self.__tensors.items():
            if not torch.allclose(
                v, other.__tensors[k], rtol=rtol, atol=atol, equal_nan=True
            ):
                return False
        return True

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the Data into a DataFrame.  The returned DataFrame has
        secondary columns iff one or more tensors have extra dimensions (i.e.,
        3 or more dimensions)."""
        tensors = []
        columns = []
        for name, tensor in self.__tensors.items():
            # NOTE: Since the 1st and 2nd dimensions could be 0, so the other
            # dimensions cannot use -1 in reshape.
            tensors.append(tensor.reshape(-1, int(np.prod(tensor.shape[2:]))))
            for index in itertools.product(*map(range, tensor.shape[2:])):
                if index == ():
                    columns.append(name)
                else:
                    index_str = ",".join(map(str, index))
                    columns.append(f"{name}[{index_str}]")

        df = pd.DataFrame(
            torch.cat(tensors, dim=1).detach().cpu().numpy(),
            columns=pd.Index(columns),
        )

        # Insert timestamp and symbol columns.
        timestamps = np.broadcast_to(self.__timestamps[:, None], self.shape)
        df.insert(0, "timestamp", timestamps.reshape(-1))
        symbols = np.broadcast_to(self.__symbols[None], self.shape)
        df.insert(1, "symbol", symbols.reshape(-1))

        # Drop rows whose all values other than timestamp and symbol are NaN.
        df = df.dropna(thresh=3).reset_index(drop=True)

        # Drop unnecessary columns from timestamp/symbol columns.
        if not self.has_timestamps():
            # Drop timestamp column.
            df = df.drop(columns="timestamp")
        if not self.has_symbols():
            # Drop symbol column.
            df = df.drop(columns="symbol")

        return df

    def to_table(self) -> pd.DataFrame:
        """Converts the Data into a table as a DataFrame object.

        This tries to convert the data into a two-dimensional DataFrame.  If it
        is impossible, this raises a RuntimeError.  The latter dimension should
        be used for columns if the data has multiple columns because they are
        rarely homogeneous.  If both of timestamp/symbol have exactly one
        element, this uses timestamp as the former dimension because it is
        often used for indexing.
        """
        # If it has exactly one column, to_table should output a table indexed
        # by (timestamp, symbol).
        if len(self.columns) == 1:
            if self.has_timestamps():
                index = pd.Index(self.__timestamps, name="timestamp")
            else:
                index = None
            if self.has_symbols():
                columns = pd.Index(self.__symbols, name="symbol")
            else:
                # Inherit the column name if no symbols exist.
                columns = self.columns
            return pd.DataFrame(
                util.to_numpy(next(iter(self.__tensors.values()))),
                index=index,
                columns=columns,
            )

        # If it has more than one columns, to_table should use columns for the
        # latter dimension.  There are many options for the former dimension.
        # The possible patterns of dimensions are the followings:
        # (timestamp: 1, symbol: 1) => (timestamp, column),
        # (timestamp: N, symbol: 1) => (timestamp, column),
        # (timestamp: 1, symbol: X) => (timestamp, column),
        # (timestamp: N, symbol: X) => (timestamp, column),
        # (timestamp: X, symbol: 1) => (symbol, column),
        # (timestamp: 1, symbol: N) => (symbol, column),
        # (timestamp: X, symbol: N) => (symbol, column),
        # (timestamp: X, symbol: X) => (None, column),
        # (timestamp: N, symbol: N) => Invalid,
        # where
        # - X means the index has no elements,
        # - 1 means the index has exactly one element,
        # - N means the index has multiple elements.
        if len(self.__symbols) == 1 and self.has_timestamps():
            index = pd.Index(self.__timestamps, name="timestamp")
        elif not self.has_symbols() and not self.has_timestamps():
            index = None
        elif len(self.__timestamps) == 1:
            index = pd.Index(self.__symbols, name="symbol")
        else:
            raise RuntimeError(
                "Data.to_table requires 2D data, but it has "
                + f"{len(self.columns)} columns, {len(self.__symbols)} symbols, "
                + f"{len(self.__timestamps)} timestamps"
            )
        columns = pd.Index(self.columns, name="column")
        return pd.DataFrame(
            np.concatenate(
                [util.to_numpy(v) for v in self.__tensors.values()], axis=1
            ),
            index=index,
            columns=columns,
        )

    def to_series(self) -> pd.Series:
        """Converts the Data to a Series.  The Data must be a single time
        series.  Otherwise, this raises a ValueError.
        """

        if self.__symbols.shape[0] != 1:
            raise ValueError(
                "Data.to_series can be applied to Data with exactly one "
                + f"symbol, but its shape is {self.__symbols.shape}."
            )

        df = self.to_dataframe()

        # Use timestamp as an index if exists (if not aggregated).
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        # Use symbol as a name if exists (if not aggregated).
        name = ""
        if "symbol" in df.columns:
            name = str(self.__symbols[0])
            # Drop symbol column.
            # NOTE: DataFrame.drop does not work due to MultiIndex.
            df = pd.DataFrame(df.iloc[:, ~df.columns.get_loc("symbol")])

        if len(df.columns) != 1:
            raise ValueError(
                "Data.to_series can be applied to Data with exactly single "
                + f"column, but it has {len(df.columns)} columns."
            )
        return pd.Series(df.iloc[:, 0], name=name)

    @typing.overload
    def to_csv(self, path: None = None) -> str:
        pass

    @typing.overload
    def to_csv(self, path: str) -> None:
        pass

    def to_csv(self, path: typing.Optional[str] = None) -> typing.Optional[str]:
        result = self.to_dataframe().to_csv(path, index=False)
        return typing.cast(typing.Optional[str], result)

    def to_matrix_csv(
        self, path: typing.Optional[str] = None
    ) -> typing.Optional[str]:
        result = self.to_matrix().to_csv(path)
        return typing.cast(typing.Optional[str], result)

    def to_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.array, index=self.__timestamps, columns=self.__symbols
        )

    def timestamp_index(
        self, v: typing.Any, side: str = "equal"
    ) -> typing.Union[None, int, np.ndarray]:
        """Converts an index representing timestamp(s) into an integer
        index."""
        return _to_integer_index(
            self.__timestamps, lambda x: np.datetime64(x, "ns"), v, side
        )

    def symbol_index(
        self, v: typing.Any, side: str = "equal"
    ) -> typing.Union[None, int, np.ndarray]:
        """Converts an index representing symbol(s) into an integer index."""
        return _to_integer_index(self.__symbols, str, v, side)

    def like(self, other: "Data") -> "Data":
        """Reshape itself to the other's shape.  This should respect timestamps
        and symbols of each argument (i.e., they can different sets of
        timestamps and symbols).  Values whose the combination of a timestamp
        and a symbol does not exist in `other` will be discarded.  On the other
        hand, values whose the combiation does not exist are treated as invalid
        values (defined by qfeval.core.util.invalid_values_like).
        """
        timestamp_indexes, timestamp_mask = util.make_array_mapping(
            self.__timestamps, other.__timestamps
        )
        symbol_indexes, symbol_mask = util.make_array_mapping(
            self.__symbols, other.__symbols
        )
        tensors = {}
        for k, v in self.__tensors.items():
            v = v[
                torch.tensor(timestamp_indexes[:, None]),
                torch.tensor(symbol_indexes[None, :]),
            ]
            mask = timestamp_mask[:, None] | symbol_mask[None, :]
            mask = mask.reshape(
                mask.shape + (1,) * (len(v.shape) - len(mask.shape))
            )
            tensors[k] = torch.where(
                torch.tensor(mask, device=v.device),
                util.nans(like=v),
                v,
            )
        return self.from_tensors(tensors, other.__timestamps, other.__symbols)

    def rename(
        self,
        columns: typing.Union[
            str,
            typing.Iterable[str],
            typing.Dict[str, str],
        ],
    ) -> Data:
        r"""Rename column names."""

        # 1. Force `columns` to be a list or dictionary.
        if isinstance(columns, str):
            columns = [columns]

        # 2. Build a mapper to map an old name to a new name.
        mapper: typing.Dict[str, str] = {}
        if isinstance(columns, dict):
            mapper = {c: c for c in self.columns}
            for src, dest in columns.items():
                mapper[src] = dest
                if src not in self.columns:
                    raise KeyError(src)
        else:
            columns = list(columns)
            if len(columns) != len(self.columns):
                raise ValueError(
                    "Inconsistent number of columns: "
                    + f"actual={len(columns)}, expected={len(self.columns)}"
                )
            mapper = dict(zip(self.columns, columns))

        # 3. Build a new Data.
        tensors = {mapper[k]: v for k, v in self.raw_tensors.items()}
        return self.from_tensors(tensors, self.__timestamps, self.__symbols)

    # TODO(imos): Deprecate this.
    def with_column_name(self, name: str) -> Data:
        r"""Deprecated: use Data.rename instead."""
        return self.rename(name)

    def merge_columns(self, other: Data) -> Data:
        assert np.array_equal(other.__timestamps, self.__timestamps)
        assert np.array_equal(other.__symbols, self.__symbols)
        return self.from_tensors(
            {**self.__tensors, **other.__tensors},
            self.__timestamps,
            self.__symbols,
        )

    def merge(self, *others: Data) -> Data:
        r"""Merges the Data object and the given Data object(s) and returns the
        merged Data object.

        The returned Data object should be the union of the Data objects.  If
        some Data objects have values for the same combination of a
        timestamp/symbol and a column, the last non-NaN value should be
        selected.

        Parameters:
            - *others (Data): Data objects should be merged with `self`.
        """

        xs: typing.Tuple[Data, ...] = (self,) + others

        # Build timestamps/symbols for the result.
        timestamps = _merge_arrays(
            [x.__timestamps for x in xs if x.has_timestamps()],
            self.__timestamps,
        )
        symbols = _merge_arrays(
            [x.__symbols for x in xs if x.has_symbols()], self.__symbols
        )

        # Build parameters for columns.
        shapes = {}
        # A mapping from a column name to parameters.  A parameter should
        # consist of the followings:
        # - values (Tensor): a flattened tensor of a column in the Data to be
        #   merged.
        # - indexes (Tensor): indexes in the flattened result tensor for the
        #   flattened tensor to be merged (i.e., the shape should correspond to
        #   the shape of `values`).
        column_to_parameters: typing.Dict[
            str, typing.List[typing.Tuple[torch.Tensor, torch.Tensor]]
        ] = {}
        for x in xs:
            # Calculate base indexes.  `base_index[i, j]` represents `x[i, j]`
            # should be merged into `base_indexes[i, j]`-th batch of `result`.
            base_indexes = torch.tensor(
                np.searchsorted(timestamps, x.__timestamps)[:, None],
                device=self.device,
            ) * symbols.size + torch.tensor(
                np.searchsorted(symbols, x.__symbols)[None, :],
                device=self.device,
            )

            for column, tensor in x.__tensors.items():
                if column not in shapes:
                    shapes[column] = (
                        timestamps.size,
                        symbols.size,
                    ) + tensor.shape[2:]
                    column_to_parameters[column] = []
                if tensor.shape[2:] != shapes[column][2:]:
                    raise ValueError(
                        f"Inconsistent shape in column `{column}': "
                        + f"actual={tensor.shape[2:]}, "
                        + f"expected={shapes[column][2:]}"
                    )
                size = int(np.prod(shapes[column][2:]))
                tensor = tensor.flatten()
                # Determine which indexes of the result should be filled.
                indexes = (
                    base_indexes[:, :, None] * size
                    + torch.arange(size, device=self.device)[None, None, :]
                ).flatten()
                # Fill non-applicable (NaN) elements with the sentry index.
                # NOTE: A sentry is intentionally appended to merged_indexes.
                indexes[torch.isnan(tensor)] = int(np.prod(shapes[column]))
                column_to_parameters[column].append((tensor, indexes))

        # Build merged tensors.
        merged_tensors = {}
        for column, params in column_to_parameters.items():
            # Prepare a concatenated tensor.  It should contain all tensors of
            # the specified column of the Data objects to be merged.
            nan_tensor = torch.as_tensor(
                math.nan, dtype=self.dtype, device=self.device
            )
            concatenated_tensors = torch.cat(
                [nan_tensor[None]] + [t for t, _ in params]
            )
            # Prepare indexes in the concatenated tensor for the merged tensor.
            # NOTE: The default value is 0 because the first element of the
            # concatenated tensor is intentionally set NaN.
            size = int(np.prod(shapes[column]))
            merged_indexes = torch.zeros(
                size + 1, dtype=torch.long, device=self.device
            )
            count = 1
            for tensor, indexes in params:
                merged_indexes[indexes] = torch.arange(
                    count,
                    len(tensor) + count,
                    device=self.device,
                )
                count += len(tensor)
            merged_tensor = concatenated_tensors[merged_indexes]
            merged_tensors[column] = merged_tensor[:-1].reshape(shapes[column])

        return self.from_tensors(merged_tensors, timestamps, symbols)

    # NOTE: f's type intentionally uses typing.Any because PyTorch functions
    # are not typed and casting them appropriately is redandunt.
    def apply(
        self,
        f: typing.Callable[..., typing.Any],
        *args: typing.Any,
        skipna: bool = False,
    ) -> Data:
        r"""Applies internal tensor(s) to the given function.  Extra arguments
        are also passed to the given function.  If an extra argument is a Data
        object, it should have a single column or the same set of columns as
        `self` has.
        """
        if skipna:
            return self.apply(
                lambda *xs: functions.skipna(f, *xs, dim=0), *args
            )
        tensors = {}
        for k, v in self.__tensors.items():
            f_args = [v]
            for arg in args:
                if isinstance(arg, Data):
                    if __debug__:
                        if arg.has_symbols() and self.has_symbols():
                            assert np.array_equal(arg.__symbols, self.__symbols)
                        if arg.has_timestamps() and self.has_timestamps():
                            assert np.array_equal(
                                arg.__timestamps, self.__timestamps
                            )
                    if len(arg.columns) == 1:
                        arg = arg.raw_tensor
                    else:
                        arg = arg.get(k).raw_tensor
                f_args.append(arg)
            result = f(*f_args)
            assert result.shape[:2] == v.shape[:2], (
                "Data.apply got a tensor with an invalid shape: "
                f"expected=({v.shape[0]}, {v.shape[1]}, *), actual={result.shape}"
            )
            tensors[k] = result
        return self.from_tensors(tensors, self.__timestamps, self.__symbols)

    def subsequences(
        self, start: int, stop: int, indexes: typing.Any = None
    ) -> Data:
        """Returns subsequences of Data for the given indexes.  For each
        timestamp index `t`, it extracts a subsequence of timestamps
        `[t + start, t + stop)`.  If indexes is None, this returns
        subsequences for all viable timestamps."""
        assert start <= stop, "Invalid timestamp range: {} < {}".format(
            start, stop
        )
        if indexes is None:
            tensor_indexes = torch.arange(self.shape[0], device=self.device)
        elif isinstance(indexes, torch.Tensor):
            tensor_indexes = indexes
        else:
            # Support various indexes (e.g., timestamps, a slice) here.
            tensor_indexes = torch.arange(self.shape[0], device=self.device)[
                self.timestamp_index(indexes)  # type: ignore
            ]
        assert len(tensor_indexes.shape) == 1

        # Prepare indexes for timestamp and symbol respectively.  Their shapes
        # should be [timestamp, symbol, subsequence].
        timestamp_indexes = (
            tensor_indexes[:, None, None]
            + torch.arange(
                start, stop, device=self.device, dtype=tensor_indexes.dtype
            )[None, None, :]
        )
        # Prepare a mask of out-of-range timestamp indexes.
        mask = (timestamp_indexes < 0) | (timestamp_indexes >= self.shape[0])
        # Fill masked timestamp indexes with 0 to avoid an out-of-range error.
        timestamp_indexes[mask] = 0
        symbol_indexes = torch.arange(
            self.shape[1], device=self.device, dtype=tensor_indexes.dtype
        )[None, :, None]

        tensors = {}
        for k, v in self.__tensors.items():
            subsequences = v[timestamp_indexes, symbol_indexes]
            tensors[k] = torch.where(
                mask, util.nans(like=subsequences), subsequences
            )
        timestamps = self.__timestamps[tensor_indexes.cpu().numpy()]
        return self.from_tensors(tensors, timestamps, self.__symbols)

    @dataclass
    class RandSeqsResult:
        # Subsequences, whose shape is (batch_size, length).
        tensor: torch.Tensor
        # Timestamp indexes, whose shape is (batch_size).
        timestamps: torch.Tensor
        # Symbol indexes, whose shape is (batch_size).
        symbols: torch.Tensor

    def randseqs(
        self, batch_size: int, length: int, skipna: bool = True
    ) -> typing.Iterator[RandSeqsResult]:
        r"""Generates subsequences of (batch_size, length) randomly.

        This returns a generator of tensors instead of Data since it
        randomizes timestamps in addition to symbols.

        CAVEAT: The Data must have exactly one column.
        CAVEAT: The number of subsequences may be insufficient in some cases
        (e.g., reaching the end of a permutation, batch size is larger than
        the size of a permutation).
        """
        x = self.tensor
        t = x.size(0) - length + 1
        if t <= 0:
            return
        p = torch.randperm(t * x.size(1), device=self.device)
        if skipna:
            mask = x.isnan().reshape(x.shape[:2] + (-1,)).any(dim=-1)
            mask = (functions.msum(mask.flip(0), length, dim=0) > 0).flip(0)
            p = p[~mask.flatten(0, 1)[p]]
        f = x.flatten(0, 1)
        a = torch.arange(length, device=self.device)[None, :] * x.size(1)
        for start in range(0, len(p), batch_size):
            yield self.RandSeqsResult(
                tensor=f[p[start : min(start + batch_size, len(p)), None] + a],
                timestamps=p.div(x.size(1), rounding_mode="floor"),
                symbols=p.remainder(x.size(1)),
            )

    def plot(
        self,
        ax: typing.Optional[matplotlib.axes.Axes] = None,
        **kwargs: typing.Any,
    ) -> typing.List[matplotlib.axes.Axes]:
        """Plot OHLC(V) data or other generic data.

        This switches the backend based on the set of columns.  If the data has
        OHLC columns, this plots the data using candle sticks.
        """
        if plt is None:
            raise ImportError(
                "matplotlib is not installed. Please install it with `pip install qfeval-data[plot]`"
            )
        df = self.to_table()
        if len({"open", "high", "low", "close"}.difference(df.columns)) == 0:
            fig = plot.Figure(plt.figure() if ax is None else ax)
            self.candlestick(ax=fig.primary_axes)
            if "volume" in df.columns:
                self.volume.bar(ax=fig.append_axes())
            axs = fig.axes
        else:
            axs = plot.plot_dataframe(df, ax=ax, **kwargs)
        return axs

    def zeros(self) -> Data:
        """Returns zeros like the Data object."""
        return self.from_tensors(
            {k: torch.zeros_like(v) for k, v in self.__tensors.items()},
            self.__timestamps,
            self.__symbols,
        )

    def dropna(
        self,
        axis: Axis = 0,
        how: str = "any",
        thresh: typing.Optional[int] = None,
    ) -> Data:
        r"""Removes missing values."""

        # Parse axis.
        dim = _parse_axis(axis)

        # Count the number of NaNs.
        n_nans = torch.tensor(0, device=self.device)
        n_elements = 0
        for t in self.__tensors.values():
            t = t.isnan().transpose(0, dim)
            t = t.reshape(t.shape[0], -1)
            n_elements += t.shape[1]
            n_nans = n_nans + t.sum(dim=1)

        # Make a mask based on the number of NaNs.
        if thresh is None:
            if how == "any":
                should_keep = n_nans == 0
            elif how == "all":
                should_keep = n_nans != n_elements
            else:
                raise ValueError(f"how must be any or all, but: how={how}")
        else:
            should_keep = thresh <= n_elements - n_nans

        # Drop elements along with the specified axis.
        if dim == 0:
            return self[should_keep]
        elif dim == 1:
            return self[:, should_keep]
        raise ValueError(f"Axis must be timestamp or symbol: {axis}")

    def fillna(
        self,
        value: float = 0.0,
        *,
        method: typing.Optional[str] = None,
        axis: Axis = 0,
    ) -> Data:
        r"""Fills NaN values based on the specified method."""

        # Parse axis.
        dim = _parse_axis(axis)

        if method is None:
            return self.apply(lambda x: x.nan_to_num(value))
        if method == "ffill":
            return self.apply(lambda x: functions.ffill(x, dim=dim))
        if method == "bfill":
            return self.apply(lambda x: functions.bfill(x, dim=dim))
        raise ValueError(f"Unknown filling method: {method}")

    ############################################################################
    # Plot methods
    ############################################################################

    def line(
        self,
        ax: typing.Optional[matplotlib.axes.Axes] = None,
        even: bool = False,
        **kwargs: typing.Any,
    ) -> matplotlib.axes.Axes:
        r"""Plots y versus x as lines and/or markers, as well as
        `matplotlib.plot` does.
        """
        axes = self.prepare_axes(ax, even)
        x, y = self.__plot_args(self.__to_x(axes), self)
        axes.plot(x, y, **kwargs)
        return axes

    def bar(
        self,
        width: typing.Union[float, Data] = 0.8,
        bottom: typing.Union[float, Data] = 0.0,
        ax: typing.Optional[matplotlib.axes.Axes] = None,
        even: bool = False,
        **kwargs: typing.Any,
    ) -> matplotlib.axes.Axes:
        r"""Draws a bar plot, as well as `matplotlib.bar` does."""
        axes = self.prepare_axes(ax, even)
        x, h, w, b = self.__plot_args(self.__to_x(axes), self, width, bottom)
        axes.bar(x, height=h, width=w, bottom=b, **kwargs)
        return axes

    def vlines(
        self,
        ymax: typing.Union[float, Data] = 0.0,
        ax: typing.Optional[matplotlib.axes.Axes] = None,
        even: bool = False,
        **kwargs: typing.Any,
    ) -> matplotlib.axes.Axes:
        r"""Draws vertical lines at each x from `ymin` to `ymax` as well as
        `matplotlib.vlines` does.
        """
        axes = self.prepare_axes(ax, even)
        x, y1, y2 = self.__plot_args(self.__to_x(axes), self, ymax)
        axes.vlines(x, y1, y2, **kwargs)
        return axes

    def fill_between(
        self,
        y2: typing.Union[float, Data] = 0.0,
        ax: typing.Optional[matplotlib.axes.Axes] = None,
        even: bool = False,
        **kwargs: typing.Any,
    ) -> matplotlib.axes.Axes:
        r"""Fills the area between two horizontal curves, as well as
        `matplotlib.fill_between` does.
        """
        axes = self.prepare_axes(ax, even)
        x, ay1, ay2 = self.__plot_args(self.__to_x(axes), self, y2)
        axes.fill_between(x, ay1, ay2, **kwargs)
        return axes

    def candlestick(
        self,
        ax: typing.Optional[matplotlib.axes.Axes] = None,
        even: bool = False,
        *,
        color: typing.Optional[str] = None,
        upcolor: str = "#ee3333",
        downcolor: str = "#118822",
        neutralcolor: str = "#444444",
        bgcolor: str = "#ffffff",
        width: float = 0.6,
        linewidth: float = 0.5,
        zorder: int = 0,
        **kwargs: typing.Any,
    ) -> matplotlib.axes.Axes:
        r"""Plots candlesticks based on (open, high, low, close) columns."""
        axes = self.prepare_axes(ax, even)

        def draw_candlestick(mask: Data, color: str, fillcolor: str) -> None:
            if not mask.tensor.any():
                return
            # NOTE: Masked elements are not dropped, but filled with NaNs.
            self.low[mask].vlines(
                ax=axes,
                ymax=self.high[mask],
                linewidth=linewidth,
                zorder=zorder,
                color=color,
                edgecolor=color,
                **kwargs,
            )
            bottom = self.open.apply(lambda x, y: torch.min(x, y), self.close)
            (self.close - self.open).abs()[mask].bar(
                ax=axes,
                width=width,
                bottom=bottom[mask],
                linewidth=linewidth,
                color=fillcolor,
                edgecolor=color,
                zorder=zorder,
                **kwargs,
            )

        # 1. Draw candle sticks.  Use different colors for up/down/neutral.
        upcolor = color or upcolor
        downcolor = color or downcolor
        neutralcolor = color or neutralcolor
        # NOTE: Ascending candlesticks should color only outlines.
        draw_candlestick(self.open < self.close, upcolor, bgcolor)
        draw_candlestick(self.open > self.close, downcolor, downcolor)
        draw_candlestick(self.open.eq(self.close), neutralcolor, neutralcolor)

        # 2. Update the range of y-axis.
        # NOTE: This is necessary because bottom of bar adds no margins to
        # y-axis.
        high, low = (
            np.nanmax(self.high.tensor.cpu().numpy()),
            np.nanmin(self.low.tensor.cpu().numpy()),
        )
        ax_high, ax_low = axes.get_ylim()
        margin = (
            axes.margins()[1]  # type:ignore[index]
            / (1 - axes.margins()[1])  # type:ignore[index]
            / 2
        )
        axes.set_ylim(
            min(ax_low, low - (high - low) * margin),
            max(ax_high, high + (high - low) * margin),
        )

        return axes

    def prepare_axes(
        self,
        ax: typing.Optional[matplotlib.axes.Axes] = None,
        even: bool = False,
    ) -> matplotlib.axes.Axes:
        r"""Prepares an Axes object to plot on."""
        if plt is None:
            raise ImportError(
                "matplotlib is not installed. Please install it with `pip install qfeval-data[plot]`"
            )
        # 1. Prepare an Axes object.
        if ax is None:
            # Get the current axes provided by matplotlib.
            axes = plt.gca()
        else:
            axes = ax

        # 2. Initialize the Axes object if not.
        if not hasattr(axes, "__timestamps"):
            axes.__timestamps = self.timestamps  # type:ignore[attr-defined]
            axes.__even = even  # type:ignore[attr-defined]
            self.__set_xticks(axes)

        # 3. Returns the preprocessed Axes object.
        return axes

    def __to_x(self, ax: matplotlib.axes.Axes) -> np.ndarray:
        r"""Converts the timestamps into the x coordinate of the given axes."""

        # 1. If even is True, use the timescale as is.
        if ax.__even:  # type:ignore[attr-defined]
            unit, value = np.datetime_data(
                ax.__timestamps  # type:ignore[attr-defined]
            )
            ret: np.ndarray = self.timestamps.astype(
                f"datetime64[{value}{unit}]"
            )
            return ret

        # 2. Prepare two float arrays: ax_t and self_t, which represent x-axis
        # timestamps and data timestamps respectively.
        ax_timestamps = typing.cast(
            np.ndarray, ax.__timestamps  # type:ignore[attr-defined]
        )
        ax_t = ax_timestamps.astype("datetime64[us]").astype("f")
        self_t: np.ndarray = np.array(
            self.timestamps, dtype="datetime64[us]"
        ).astype("f")

        # 3. Figure out `start` and `stop` representing where to insert
        # between.
        ax_padded = np.concatenate((np.array([-1e30]), ax_t, np.array([1e30])))
        indices = np.searchsorted(ax_padded, self_t)
        start, stop = ax_padded[indices - 1], ax_padded[indices]

        # 3. Use evenly-scaled x for out of the window.
        even_x = (self_t - ax_t[0]) / (ax_t[-1] - ax_t[0]) * (len(ax_t) - 1)
        uneven_x = (self_t - start) / (stop - start) + indices - 2
        use_even = (self_t <= ax_t[0]) | (self_t >= ax_t[-1])
        return typing.cast(np.ndarray, np.where(use_even, even_x, uneven_x))

    def __plot_args(
        self,
        x: np.ndarray,
        *args: typing.Union[float, Data],
    ) -> typing.Tuple[typing.Union[float, np.ndarray], ...]:
        result: typing.List[typing.Union[float, np.ndarray]] = [x]
        for arg in args:
            if isinstance(arg, Data):
                result.append(arg.like(self).array)
            else:
                result.append(float(arg))
        mask = np.isnan(result[0])
        for r in result[1:]:
            mask = mask | np.isnan(r)
        return tuple(
            [r[~mask] if isinstance(r, np.ndarray) else r for r in result]
        )

    def __plot_arg(
        self, x: typing.Union[float, Data]
    ) -> typing.Union[float, np.ndarray]:
        r"""Converts an argument to a matplotlib-compatilble argument."""
        if isinstance(x, Data):
            return x.like(self).array
        return float(x)

    def __set_xticks(self, ax: matplotlib.axes.Axes) -> None:
        r"""Sets x ticks to the given axes based on the timestamps."""
        # 1. Figure out major/minor tick deltas.
        start = np.datetime64(self.timestamps[0])
        stop = np.datetime64(self.timestamps[-1])
        major_delta, minor_delta = self.__xtick_deltas(start, stop)

        # 2. Set major/minor ticks respectively and labels for major ticks.
        timestamps = self.timestamps.astype("datetime64[us]")
        for delta, is_major in [(major_delta, True), (minor_delta, False)]:
            ticks = self.__xticks(start, stop, delta).astype("datetime64[us]")
            tick_indices = np.searchsorted(timestamps, ticks, "right") - 1
            ax.set_xticks(tick_indices, minor=not is_major)
            if is_major:
                # Round ticks up to a multilple of a display unit.
                # NOTE: When setting monthly ticks to weekly candlesticks, the
                # first candlestick of a month would start with its previous
                # month.
                unit, _ = np.datetime_data(delta)
                label_ticks = util.ceil_time(
                    timestamps[tick_indices], np.timedelta64(1, unit)
                )
                labels = [t.astype(f"datetime64[{unit}]") for t in label_ticks]
                ax.set_xticklabels(labels, rotation=45, ha="right")

    @classmethod
    def __xtick_deltas(
        cls,
        start: np.datetime64,
        stop: np.datetime64,
        max_ticks: int = 8,
    ) -> typing.Tuple[np.timedelta64, np.timedelta64]:
        r"""Returns a pair of a major tick delta and a minor tick delta
        preferable for the given interval.
        """
        deltas = {
            "Y": [(50, 10), (20, 5), (10, 5), (5, 1), (2, 1), (1, 1)],
            "M": [(6, 1), (4, 1), (3, 1), (2, 1), (1, 1)],
            "D": [(14, 7), (7, 1), (4, 1), (2, 1), (1, 1)],
            "h": [(12, 3), (8, 2), (6, 1), (4, 1), (3, 1), (2, 1), (1, 1)],
            "m": [(30, 10), (20, 5), (10, 5), (5, 1), (2, 1), (1, 1)],
            "s": [(30, 10), (20, 5), (10, 5), (5, 1), (2, 1), (1, 1)],
        }
        last_deltas = (np.timedelta64(100, "Y"), np.timedelta64(100, "Y"))
        for unit, sizes in deltas.items():
            for major, minor in sizes:
                major_delta = np.timedelta64(major, unit)
                minor_delta = np.timedelta64(minor, unit)
                ticks = cls.__xticks(start, stop, major_delta)
                if len(ticks) > max_ticks:
                    return last_deltas
                last_deltas = (major_delta, minor_delta)
        return last_deltas

    @classmethod
    def __xticks(
        cls,
        start: np.datetime64,
        stop: np.datetime64,
        delta: np.timedelta64,
    ) -> np.ndarray:
        r"""Returns an array of timestamps from the start time to the end time
        in the given interval `delta`.

        The returned timestamps has the unit of `delta`.
        """
        start, stop = util.ceil_time(start, delta), util.floor_time(stop, delta)
        idelta: int = delta.astype(int)
        num_ticks = (stop - start).astype(int) // idelta + 1
        return typing.cast(np.ndarray, start + delta * np.arange(num_ticks))

    ############################################################################
    # Technical indicators
    ############################################################################

    def moving_average(self, window: int = 25, skipna: bool = True) -> Data:
        """Returns its moving average."""
        return self.apply(
            lambda x: functions.ma(x, window, dim=0), skipna=skipna
        )

    def plot_moving_average(
        self,
        window: int = 25,
        skipna: bool = True,
        ax: typing.Optional[matplotlib.axes.Axes] = None,
        **kwargs: typing.Any,
    ) -> matplotlib.axes.Axes:
        """Plots its moving average.

        See Data.moving_average for more details.
        """
        return self.moving_average(window, skipna).line(ax=ax, **kwargs)

    def bollinger_band(
        self, window: int = 20, sigma: float = 2.0, skipna: bool = True
    ) -> typing.Tuple[Data, Data, Data]:
        """Returns its bollinger band.

        Definition: https://www.investopedia.com/terms/b/bollingerbands.asp
        """
        middle = self.moving_average(window, skipna)
        width = self.apply(
            lambda x: functions.mstd(x, window, dim=0, ddof=0) * sigma,
            skipna=skipna,
        )
        return middle + width, middle, middle - width

    def plot_bollinger_band(
        self,
        window: int = 20,
        sigma: float = 2.0,
        skipna: bool = True,
        *,
        ax: typing.Optional[matplotlib.axes.Axes] = None,
        zorder: int = 0,
        linewidth: float = 0.2,
        fillcolor: str = "#eef8ff",
        edgecolor: str = "#4444aa",
        centercolor: str = "#ff6666",
        **kwargs: typing.Any,
    ) -> matplotlib.axes.Axes:
        """Plots its bollinger band.

        See Data.bollinger_band for more details.
        """
        upper, middle, lower = self.bollinger_band(window, sigma, skipna)
        kwargs["zorder"] = zorder
        ax = lower.fill_between(upper, ax=ax, color=fillcolor, **kwargs)
        kwargs["linewidth"] = linewidth
        ax = middle.line(ax=ax, color=centercolor, **kwargs)
        kwargs["color"] = edgecolor
        ax = upper.line(ax=ax, **kwargs)
        return lower.line(ax=ax, **kwargs)

    ############################################################################
    # Downsampling methods
    ############################################################################

    def downsample(
        self,
        delta: np.timedelta64,
        origin: typing.Optional[np.datetime64] = None,
        offset: typing.Optional[np.timedelta64] = None,
        aggregation_f: typing.Callable[..., typing.Any] = functions.nansum,
    ) -> Data:
        r"""Downsamples the data into the specified frequency data.

        Parameters:
        - `delta` is a unit of intervals between ticks.
        - `origin` is the origin time to determine the positions of ticks for
          the given `delta`.  If None, `qfeval.core.util.time_origin` is used
          to determine the origin.
        - `offset` is the shift of timestamps.  This also changes the labels of
          x-axis.  This is useful to adjust timezones.  For example,
          `2020-04-02T00:00:00+09:00` is on `2020-04-01` in UTC, but
          displaying it as `2020-04-02` is preferable in some datasets.
        - `aggregation_f` is the default function to aggregate values at
          multiple timestamps.
        """

        dest_unit = np.datetime_data(delta)[0]
        timestamps = util.floor_time(self.timestamps, delta, origin, offset)
        timestamps = timestamps.astype(f"datetime64[{dest_unit}]")
        timestamps, group_ids = np.unique(timestamps, return_inverse=True)
        group_ids = torch.tensor(group_ids, device=self.device)

        tensors = {}
        for k, v in self.__tensors.items():
            v = functions.groupby(v, group_ids, dim=0)
            if k == "open":
                # An opening price should represent the first valid price for
                # a time window.
                v = functions.bfill(v, dim=1)[:, 0]
            elif k == "high":
                v = functions.nanmax(v, dim=1).values
            elif k == "low":
                v = functions.nanmin(v, dim=1).values
            elif k == "close":
                # A closing price should represent the last valid price for
                # a time window.
                v = functions.ffill(v, dim=1)[:, -1]
            else:
                v = aggregation_f(v, dim=1)
            tensors[k] = v

        return self.from_tensors(tensors, timestamps, self.__symbols)

    def minutely(
        self,
        origin: typing.Optional[np.datetime64] = None,
        offset: typing.Optional[np.timedelta64] = None,
        aggregation_f: typing.Callable[..., typing.Any] = functions.nansum,
    ) -> Data:
        return self.downsample(
            np.timedelta64(1, "m"), origin, offset, aggregation_f=aggregation_f
        )

    def hourly(
        self,
        origin: typing.Optional[np.datetime64] = None,
        offset: typing.Optional[np.timedelta64] = None,
        aggregation_f: typing.Callable[..., typing.Any] = functions.nansum,
    ) -> Data:
        return self.downsample(
            np.timedelta64(1, "h"), origin, offset, aggregation_f=aggregation_f
        )

    def daily(
        self,
        origin: typing.Optional[np.datetime64] = None,
        offset: typing.Optional[np.timedelta64] = None,
        aggregation_f: typing.Callable[..., typing.Any] = functions.nansum,
    ) -> Data:
        return self.downsample(
            np.timedelta64(1, "D"), origin, offset, aggregation_f=aggregation_f
        )

    def weekly(
        self,
        origin: typing.Optional[np.datetime64] = None,
        offset: typing.Optional[np.timedelta64] = None,
        aggregation_f: typing.Callable[..., typing.Any] = functions.nansum,
    ) -> Data:
        # CAUTION: Do not use 1W because 1W supports only weeks from Monday.
        return self.downsample(
            np.timedelta64(7, "D"), origin, offset, aggregation_f=aggregation_f
        )

    def monthly(
        self,
        origin: typing.Optional[np.datetime64] = None,
        offset: typing.Optional[np.timedelta64] = None,
        aggregation_f: typing.Callable[..., typing.Any] = functions.nansum,
    ) -> Data:
        return self.downsample(
            np.timedelta64(1, "M"), origin, offset, aggregation_f=aggregation_f
        )

    def yearly(
        self,
        origin: typing.Optional[np.datetime64] = None,
        offset: typing.Optional[np.timedelta64] = None,
        aggregation_f: typing.Callable[..., typing.Any] = functions.nansum,
    ) -> Data:
        return self.downsample(
            np.timedelta64(1, "Y"), origin, offset, aggregation_f=aggregation_f
        )

    ############################################################################
    # Data processing methods
    ############################################################################

    abs = _define_unary_operator(torch.abs)
    sqrt = _define_unary_operator(torch.sqrt)

    def shift(self, shift: int = 1, skipna: bool = False) -> Data:
        """Shift timestamps by the given size `shift`.  `shift` can be
        positive, negative or zero."""
        return self.apply(
            lambda x: (
                functions.nanshift(x, shift, 0)
                if skipna
                else functions.shift(x, shift, 0)
            )
        )

    def group_shift(
        self, shift: int = 1, reference: typing.Optional[Data] = None
    ) -> Data:
        r"""Shift timestamps by the given size `shift`, while skipping
        the timestamps where one or more symbols have missing values.

        Parameters:
        - shift: The amount of shifts.
        - reference: If set, it will create locations to be skipped by
            extracting the patterns of missing values. `reference` must be
            a Data with the same set of columns as `self` has.
        """
        if reference is None:
            return self.apply(
                lambda x: functions.group_shift(
                    x,
                    shift=shift,
                    dim=0,
                    refdim=1,
                    agg_f="any",
                )
            )
        else:
            assert np.array_equal(reference.__timestamps, self.__timestamps)
            assert len(reference.columns) == len(self.columns)
            tensors = {}
            for k, v in reference.__tensors.items():
                mask = functions.reduce_nan_patterns(v, 0, 1, "any")
                x = self.get(k).raw_tensor
                shifted = functions.group_shift(x, shift, 0, mask=mask)
                tensors[k] = shifted
            return self.from_tensors(tensors, self.__timestamps, self.__symbols)

    def pct_change(self, periods: int = 1, skipna: bool = False) -> Data:
        return self / self.shift(periods, skipna=skipna) - 1

    def diff(self, periods: int = 1, skipna: bool = False) -> Data:
        return self - self.shift(periods, skipna=skipna)

    def cumsum(self, axis: Axis = 0, skipna: bool = True) -> Data:
        dim = _parse_axis(axis)
        return self.apply(
            lambda x: (
                functions.nancumsum(x, dim=dim)
                if skipna
                else torch.cumsum(x, dim=dim)
            )
        )

    def cumprod(self, axis: Axis = 0, skipna: bool = True) -> Data:
        dim = _parse_axis(axis)
        return self.apply(
            lambda x: (
                functions.nancumprod(x, dim=dim)
                if skipna
                else torch.cumprod(x, dim=dim)
            )
        )

    @dataclass
    class _Aggregator:
        items: typing.ItemsView[str, torch.Tensor]
        dim: typing.Tuple[int, ...]
        keepdim: bool
        timestamps: np.ndarray
        symbols: np.ndarray

    def aggregate(
        self,
        f: typing.Callable[..., typing.Any],
        axis: typing.Optional[Axis] = None,
    ) -> Data:
        r"""(Experimental) Aggregates data with the given aggregation function.

        The aggregation function must not squash the timestamp dimension and
        the symbol dimension due to the internal implementation.

        Parameters:
            - f: A function that aggregates data.
            - axis: One or more dimensions to aggregate.
        """
        ag = self.__aggregate(axis)
        tensors = {}
        for k, v in ag.items:
            tensors[k] = typing.cast(torch.Tensor, f(v))
        return self.from_tensors(tensors, ag.timestamps, ag.symbols)

    def sum(self, axis: typing.Optional[Axis] = None) -> Data:
        ag = self.__aggregate(axis, "sum")
        tensors = {}
        for k, v in ag.items:
            tensors[k] = functions.nansum(v, dim=ag.dim, keepdim=ag.keepdim)
        return self.from_tensors(tensors, ag.timestamps, ag.symbols)

    def mean(self, axis: typing.Optional[Axis] = None) -> Data:
        ag = self.__aggregate(axis, "mean")
        tensors = {}
        for k, v in ag.items:
            tensors[k] = functions.nanmean(v, dim=ag.dim, keepdim=ag.keepdim)
        return self.from_tensors(tensors, ag.timestamps, ag.symbols)

    def var(
        self,
        axis: typing.Optional[Axis] = None,
        ddof: int = 1,
        _name: str = "var",
    ) -> Data:
        assert 0 <= ddof and ddof <= 1
        ag = self.__aggregate(axis, _name)
        tensors = {}
        for k, v in ag.items:
            tensors[k] = functions.nanvar(
                v, dim=ag.dim, unbiased=ddof > 0, keepdim=ag.keepdim
            )
        return self.from_tensors(tensors, ag.timestamps, ag.symbols)

    def std(self, axis: typing.Optional[Axis] = None, ddof: int = 1) -> Data:
        return self.var(axis, ddof=ddof, _name="std").sqrt()

    def skew(self, axis: typing.Optional[Axis] = None, ddof: int = 1) -> Data:
        assert 0 <= ddof and ddof <= 1
        ag = self.__aggregate(axis, "skew")
        tensors = {}
        for k, v in ag.items:
            tensors[k] = functions.nanskew(
                v, dim=ag.dim, unbiased=ddof > 0, keepdim=ag.keepdim
            )
        return self.from_tensors(tensors, ag.timestamps, ag.symbols)

    def kurt(self, axis: typing.Optional[Axis] = None, ddof: int = 1) -> Data:
        assert 0 <= ddof and ddof <= 1
        ag = self.__aggregate(axis, "kurt")
        tensors = {}
        for k, v in ag.items:
            tensors[k] = functions.nankurtosis(
                v, dim=ag.dim, unbiased=ddof > 0, keepdim=ag.keepdim
            )
        return self.from_tensors(tensors, ag.timestamps, ag.symbols)

    def count(self, axis: typing.Optional[Axis] = None) -> Data:
        ag = self.__aggregate(axis, "count")
        tensors = {}
        for k, v in ag.items:
            tensors[k] = functions.nansum(
                ~v.isnan(), dim=ag.dim, keepdim=ag.keepdim
            )
        return self.from_tensors(tensors, ag.timestamps, ag.symbols)

    ############################################################################
    # Private methods
    ############################################################################

    def __invalid_timestamp(self) -> np.ndarray:
        r"""Returns an invalid value (0 in nanoseconds) as a timestamp.

        NOTE: 1753-08-29T22:43:41.128654848 is the origin in nanoseconds.
        NOTE: Previously this returns 0000-01-01, but recent Pandas requires
        dates to be convertiable to nanoseconds.
        """
        ret: np.ndarray = np.array("0", dtype="datetime64[ns]")
        return ret

    def __invalid_symbol(self) -> np.ndarray:
        """Returns an invalid value (an empty string) as a symbol."""
        ret: np.ndarray = np.array("")
        return ret

    def __aggregate(
        self,
        axis: typing.Optional[Axis] = None,
        name: str = "aggregated",
    ) -> _Aggregator:
        """Returns a tuple of parameters for torch aggregation functions.

        The returned tuple consists of three parameters: a tuple for the dim
        parameter of torch functions (e.g., torch.sum), (possibly aggregated)
        timestamps, and (possibly aggregated) symbols.
        """
        if isinstance(axis, str):
            return self.__aggregate(_parse_axis(axis))
        if axis not in (0, 1, 2) and axis is not None:
            raise ValueError(
                "Aggregation functions support only axis=(0|1|2|None): "
                + f"axis={axis}"
            )
        if axis == 2:
            flattened_columns = [
                t.reshape(t.shape[:2] + (int(np.prod(t.shape[2:])),))
                for t in self.raw_tensors.values()
            ]
            return self._Aggregator(
                items={name: torch.cat(flattened_columns, dim=2)}.items(),
                dim=(2,),
                keepdim=False,
                timestamps=self.__timestamps,
                symbols=self.__symbols,
            )
        d = (0, 1) if axis is None else (axis,)
        timestamps = (
            self.__timestamps
            if 0 not in d
            else self.__invalid_timestamp()[None]
        )
        symbols = (
            self.__symbols if 1 not in d else self.__invalid_symbol()[None]
        )
        return self._Aggregator(
            items=self.__tensors.items(),
            dim=d,
            keepdim=True,
            timestamps=timestamps,
            symbols=symbols,
        )

    def __index_slices(
        self,
    ) -> typing.Tuple[typing.Union[slice, int], typing.Union[slice, int]]:
        """Returns a pair of slices to reduce dimensions if the Data object
        drops timestamp/symbol columns.
        """
        return (
            slice(None) if self.has_timestamps() else 0,
            slice(None) if self.has_symbols() else 0,
        )


DataV2 = Data


def _parse_axis(axis: Axis) -> int:
    if isinstance(axis, str):
        if axis == "timestamp":
            return 0
        if axis == "symbol":
            return 1
        if axis == "column":
            return 2
        raise ValueError(f"Unknown axis name: {axis}")
    return int(axis)


def _to_integer_index(
    a: np.ndarray,
    cast: typing.Callable[[typing.Any], typing.Any],
    v: typing.Any,
    side: str,
) -> typing.Union[None, int, np.ndarray]:
    """Returns indices corresponding to `v`.  If `v` represents one or more
    indices, this returns indices can be used in NumPy indexing.  Otherwise,
    this finds indices where elements `v` should be inserted into the sorted
    array `a` with maintaining its order like `numpy.searchsorted`.  If
    `side` is `"equal"`, this checks if the given elements exist in the
    sorted array `a`.
    """
    if v is None:
        return None
    if isinstance(v, int) or isinstance(v, np.integer):
        return int(v)
    if isinstance(v, tuple) or isinstance(v, list):
        v = np.array(v)
    if isinstance(v, np.ndarray):
        if np.issubdtype(v.dtype, np.dtype(np.int64)):
            return v
        if np.issubdtype(v.dtype, np.dtype(np.bool_)):
            return v
        v = np.vectorize(cast)(v)
        left: np.ndarray = np.searchsorted(a, v, "left")
        right = np.searchsorted(a, v, "right")
        if np.any(np.equal(left, right)):
            raise KeyError(list(v[np.equal(left, right)]))
        return left
    if side == "equal":
        left_index = _to_integer_index(a, cast, v, "left")
        right_index = _to_integer_index(a, cast, v, "right")
        if left_index == right_index:
            raise KeyError(v)
        return left_index
    result = np.searchsorted(a, cast(v), side)  # type: ignore
    return int(result)


def _to_index(
    index: typing.Any,
    index_func: typing.Callable[
        [typing.Any, str], typing.Union[None, int, np.ndarray]
    ],
) -> typing.Tuple[typing.Union[None, slice, int, np.ndarray], bool]:
    result: typing.Union[None, slice, int, np.ndarray]
    collapse = False
    if isinstance(index, slice):
        result = slice(
            index_func(index.start, "left"),
            index_func(index.stop, "left"),
            index.step,
        )
    elif (
        isinstance(index, int)
        or isinstance(index, np.integer)
        or (isinstance(index, (np.ndarray, torch.Tensor)) and index.shape == ())  # type: ignore # NOQA
    ):
        if int(index) == -1:
            result = slice(-1, None)
        else:
            result = slice(int(index), int(index) + 1)
        collapse = True
    elif isinstance(index, np.ndarray) and index.dtype.kind == "b":
        result = index
    elif isinstance(index, torch.Tensor) and index.dtype == torch.bool:
        # NOTE: This converts a Tensor object to a numpy object to enable
        # indexing of timestamp/symbol axes like `self.symbols[index]` because
        # a tensor on a GPU cannot be used for indexing a NumPy array.  It can
        # be an option to return a tensor in the future because returning a
        # tensor would have the advantage of performance.
        result = util.to_numpy(index)
    elif (
        isinstance(index, np.ndarray)
        or isinstance(index, list)
        or isinstance(index, tuple)
    ):
        result = index_func(index, "equal")
    else:
        left, right = index_func(index, "left"), index_func(index, "right")
        if left == right:
            raise KeyError(index)
        result = slice(left, right)
        collapse = True
    return result, collapse


def _merge_arrays(
    xs: typing.Iterable[np.ndarray], default: np.ndarray
) -> np.ndarray:
    xs = list(xs)
    if len(xs) == 0:
        return default
    # Returns the first object if all arrays are the same.  This would speed up
    # array comparison for some situations because the comparison of arrays
    # pointing to the same object can be efficient.
    xs = list([xs[0]] + [x for x in xs[1:] if not np.array_equal(x, xs[0])])
    if len(xs) == 1:
        return xs[0]
    result = np.unique(np.concatenate(xs, axis=0), axis=0)
    if np.array_equal(result, xs[0]):
        return xs[0]
    return typing.cast(np.ndarray, result)
