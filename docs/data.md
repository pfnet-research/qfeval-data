# Data Class Reference

The `Data` class is the core component of qfeval-data. It manages numerical tensors indexed by timestamps and symbols, designed for efficient financial time series manipulation.

<!-- test:setup
import numpy as np
import pandas as pd
import torch
from qfeval_data import Data, Flattener

# Create sample OHLCV data for examples
def create_sample_data():
    timestamps = np.array(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        dtype="datetime64[D]",
    )
    symbols = np.array(["AAPL", "GOOG"])
    tensors = {
        "open": torch.tensor([[100.0, 200.0], [101.0, 201.0], [102.0, 202.0], [103.0, 203.0], [104.0, 204.0]]),
        "high": torch.tensor([[105.0, 205.0], [106.0, 206.0], [107.0, 207.0], [108.0, 208.0], [109.0, 209.0]]),
        "low": torch.tensor([[98.0, 198.0], [99.0, 199.0], [100.0, 200.0], [101.0, 201.0], [102.0, 202.0]]),
        "close": torch.tensor([[104.0, 204.0], [105.0, 205.0], [106.0, 206.0], [107.0, 207.0], [108.0, 208.0]]),
        "volume": torch.tensor([[1e6, 5e5], [1.1e6, 5.5e5], [1.2e6, 6e5], [1.3e6, 6.5e5], [1.4e6, 7e5]]),
    }
    return Data.from_tensors(tensors, timestamps, symbols)

data = create_sample_data()
-->

## Overview

<!-- test:skip -->
```python
from qfeval_data import Data
```

### Data Structure

- **Tensors**: Dictionary mapping column names (strings) to PyTorch tensors
- **Shape**: Each tensor has shape `(num_timestamps, num_symbols, *extra_dimensions)`
- **Timestamps**: `np.ndarray[datetime64]` - always sorted
- **Symbols**: `np.ndarray[str]` - always sorted

### Design Principles

1. **Lazy Slicing**: Slicing operations create views without copying data
2. **Sorted Indexes**: Timestamps and symbols are automatically sorted on construction
3. **Method Chaining**: Most methods return `Data` objects for fluent API
4. **GPU Support**: Full PyTorch tensor backend with device flexibility

---

## Construction Methods

### `Data.from_dataframe(df, dtype=None, device=None)`

Create a `Data` object from a pandas DataFrame.

**Parameters:**
- `df` (`pd.DataFrame`): DataFrame with required `timestamp` and `symbol` columns
- `dtype` (`torch.dtype`, optional): Data type for tensors
- `device` (`str` or `torch.device`, optional): Device for tensors

**Returns:** `Data`

**Example:**
```python
import pandas as pd
from qfeval_data import Data

df = pd.DataFrame({
    "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "symbol": ["AAPL", "GOOG", "AAPL", "GOOG"],
    "open": [150.0, 140.0, 152.0, 142.0],
    "close": [155.0, 145.0, 153.0, 143.0],
})
data = Data.from_dataframe(df)
```

**Multi-dimensional columns:**

Use bracket notation in column names for multi-dimensional data:
```python
df = pd.DataFrame({
    "timestamp": ["2024-01-01"],
    "symbol": ["AAPL"],
    "embedding[0]": [0.1],
    "embedding[1]": [0.2],
    "embedding[2]": [0.3],
})
data = Data.from_dataframe(df)
# data.embedding has shape (1, 1, 3)
```

---

### `Data.from_csv(input, dtype=None, device=None)`

Load a `Data` object from a CSV file.

**Parameters:**
- `input` (`str` or file-like): Path to CSV file or file object
- `dtype` (`torch.dtype`, optional): Data type for tensors
- `device` (`str` or `torch.device`, optional): Device for tensors

**Returns:** `Data`

**Example:**
<!-- test:skip -->
```python
data = Data.from_csv("prices.csv")
data = Data.from_csv("prices.csv.xz")  # Supports compressed files
```

**CSV Format:**
<!-- test:skip -->
```csv
timestamp,symbol,open,high,low,close,volume
2024-01-01,AAPL,150.0,156.0,149.0,155.0,1000000
2024-01-01,GOOG,140.0,146.0,139.0,145.0,800000
```

---

### `Data.from_tensors(tensors, timestamps, symbols)`

Create a `Data` object directly from tensors. This is the most primitive constructor.

**Parameters:**
- `tensors` (`Dict[str, torch.Tensor]`): Dictionary of column name to tensor
- `timestamps` (`np.ndarray`): 1D array of datetime64 values
- `symbols` (`np.ndarray`): 1D array of symbol strings

**Returns:** `Data`

**Example:**
```python
import torch
import numpy as np
from qfeval_data import Data

tensors = {
    "open": torch.tensor([[150.0, 140.0], [152.0, 142.0]]),
    "close": torch.tensor([[155.0, 145.0], [153.0, 143.0]]),
}
timestamps = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
symbols = np.array(["AAPL", "GOOG"])

data = Data.from_tensors(tensors, timestamps, symbols)
```

**Notes:**
- Timestamps and symbols are automatically sorted; tensors are reindexed accordingly
- All tensors must have shape `(len(timestamps), len(symbols), ...)`
- All tensors must be on the same device

---

### `Data.from_preset(name="pfn-topix500", dtype=None, device=None, paths=[])`

Load a preset data file from the system path.

**Parameters:**
- `name` (`str`): Preset name (searches for `data/{name}.csv` or `data/{name}.csv.xz`)
- `dtype` (`torch.dtype`, optional): Data type for tensors
- `device` (`str` or `torch.device`, optional): Device for tensors
- `paths` (`List[str]`): Additional paths to search

**Returns:** `Data`

**Raises:** `FileNotFoundError` if preset not found

---

## Properties

### Data Access Properties

| Property | Type | Description |
|----------|------|-------------|
| `tensors` | `Dict[str, Tensor]` | Tensors after slicing applied |
| `tensor` | `Tensor` | Single tensor (requires exactly 1 column) |
| `raw_tensors` | `Dict[str, Tensor]` | Direct tensor access without slicing |
| `raw_tensor` | `Tensor` | Single raw tensor |
| `arrays` | `Dict[str, np.ndarray]` | NumPy array versions of tensors |
| `array` | `np.ndarray` | Single array version |

### Metadata Properties

| Property | Type | Description |
|----------|------|-------------|
| `timestamps` | `np.ndarray` | Sorted datetime64 array |
| `symbols` | `np.ndarray` | Sorted string array |
| `columns` | `List[str]` | List of column names |
| `shape` | `Tuple[int, int]` | `(num_timestamps, num_symbols)` |
| `device` | `torch.device` | Tensor device |
| `dtype` | `torch.dtype` | Tensor data type |

---

## Indexing and Slicing

### `data[timestamp_idx, symbol_idx]`

Access data by timestamp and symbol indices. Supports multiple indexing styles:

**Integer indexing:**
```python
data[0, :]          # First timestamp, all symbols
data[:, 0]          # All timestamps, first symbol
data[0, 0]          # Single element
data[-1, :]         # Last timestamp
```

**Slice indexing:**
```python
data[:10, :]        # First 10 timestamps
data[5:15, :]       # Timestamps 5-14
data[:, :3]         # First 3 symbols
```

**Value-based indexing:**
```python
data["2024-01-01", :]           # By timestamp value
data["2024-01-01":"2024-01-31", :]  # Timestamp range
data[:, "AAPL"]                 # By symbol value
data[:, ["AAPL", "GOOG"]]       # Multiple symbols
```

**Boolean mask indexing:**
```python
mask = data.close > data.open   # Boolean Data
filtered = data[mask]           # Apply mask (non-matching become NaN)
```

---

## Column Access

### `data.get(*columns)` / `data.get(columns)` / `data.get(pattern=...)`

Extract a subset of columns.

**Signatures:**
```python
def get(self, *columns: str) -> Data: ...
def get(self, columns: Iterable[str]) -> Data: ...
def get(self, filter_func: Callable[[str], bool]) -> Data: ...
def get(self, *, pattern: str) -> Data: ...
```

**Examples:**
```python
# Single column
opens = data.get("open")

# Multiple columns
ohlc = data.get("open", "high", "low", "close")
ohlc = data.get(["open", "high", "low", "close"])

# Filter function
prices = data.get(lambda c: c in ["open", "close"])

# Glob pattern
prices = data.get(pattern="*price*")
```

### Attribute Access

Columns can be accessed as attributes:
```python
data.close      # Equivalent to data.get("close")
data.volume     # Equivalent to data.get("volume")
```

---

### `data.set(key, value)`

Add or update a column.

**Parameters:**
- `key` (`str`): Column name
- `value` (`torch.Tensor` or `Data`): Column values

**Example:**
```python
data.set("returns", data.close.pct_change().tensor)
data.set("spread", data.high - data.low)
```

---

### `data.rename(columns)`

Rename columns.

**Parameters:**
- `columns` (`str`, `List[str]`, or `Dict[str, str]`): New column names

**Returns:** `Data`

**Examples:**
<!-- test:skip -->
```python
# Rename single column (when Data has one column)
renamed = data.get("close").rename("price")

# Rename with list (must match column count)
renamed = data.rename(["o", "h", "l", "c"])

# Rename with dict (selective)
renamed = data.rename({"open": "o", "close": "c"})
```

---

## Arithmetic Operations

All arithmetic operations are element-wise on tensors:

### Binary Operators

| Operator | Description |
|----------|-------------|
| `+`, `-`, `*`, `/` | Basic arithmetic |
| `//` | Floor division |
| `%` | Modulo |
| `**` | Power |
| `@` | Matrix multiplication |
| `&`, `\|`, `^` | Bitwise operations |

### Comparison Operators

| Operator | Description |
|----------|-------------|
| `==`, `!=` | Equality (returns boolean Data) |
| `<`, `>`, `<=`, `>=` | Comparison (returns boolean Data) |

**Note:** Use `.eq()` and `.ne()` methods to avoid Python's truthiness evaluation.

### Unary Operators

| Operator | Description |
|----------|-------------|
| `-x` | Negation |
| `+x` | Positive |
| `abs(x)` | Absolute value |
| `~x` | Bitwise not |

**Examples:**
```python
returns = (data.close / data.open) - 1
spread = data.high - data.low
is_up = data.close > data.open
```

---

## Time Series Operations

### `data.shift(shift=1, skipna=False)`

Shift values along the timestamp axis.

**Parameters:**
- `shift` (`int`): Number of periods to shift (positive = forward, negative = backward)
- `skipna` (`bool`): If True, skip NaN values when shifting

**Returns:** `Data`

**Example:**
```python
previous = data.shift(1)      # Previous day's values
next_day = data.shift(-1)     # Next day's values
```

---

### `data.pct_change(periods=1, skipna=False)`

Calculate percentage change.

**Formula:** `(current / previous) - 1`

**Parameters:**
- `periods` (`int`): Periods to shift for comparison
- `skipna` (`bool`): Skip NaN values

**Returns:** `Data`

**Example:**
```python
daily_returns = data.close.pct_change()
weekly_returns = data.close.pct_change(periods=5)
```

---

### `data.diff(periods=1, skipna=False)`

Calculate difference between current and previous values.

**Formula:** `current - previous`

**Parameters:**
- `periods` (`int`): Periods to shift for comparison
- `skipna` (`bool`): Skip NaN values

**Returns:** `Data`

---

### `data.cumsum(axis=0, skipna=True)`

Cumulative sum along an axis.

**Parameters:**
- `axis` (`int` or `str`): Axis (0/"timestamp" or 1/"symbol")
- `skipna` (`bool`): Skip NaN values

**Returns:** `Data`

---

### `data.cumprod(axis=0, skipna=True)`

Cumulative product along an axis.

**Parameters:**
- `axis` (`int` or `str`): Axis (0/"timestamp" or 1/"symbol")
- `skipna` (`bool`): Skip NaN values

**Returns:** `Data`

---

### `data.group_shift(shift=1, reference=None)`

Shift values while skipping timestamps where any symbol has missing values.

**Parameters:**
- `shift` (`int`): Number of periods to shift
- `reference` (`Data`, optional): Data to determine skip pattern from

**Returns:** `Data`

---

## Aggregation Methods

All aggregation methods support the `axis` parameter:
- `axis=0` or `axis="timestamp"`: Aggregate across timestamps
- `axis=1` or `axis="symbol"`: Aggregate across symbols
- `axis=None`: Aggregate across both axes

### Statistical Aggregations

| Method | Description |
|--------|-------------|
| `sum(axis=None)` | Sum of values |
| `mean(axis=None)` | Arithmetic mean |
| `min(axis=None)` | Minimum value |
| `max(axis=None)` | Maximum value |
| `var(axis=None, ddof=1)` | Variance |
| `std(axis=None, ddof=1)` | Standard deviation |
| `skew(axis=None, ddof=1)` | Skewness |
| `kurt(axis=None, ddof=1)` | Kurtosis |
| `count(axis=None)` | Count of non-NaN values |

### Position Aggregations

| Method | Description |
|--------|-------------|
| `first(axis="timestamp", skipna=True)` | First value |
| `last(axis="timestamp", skipna=True)` | Last value |

**Examples:**
```python
# Average price across all timestamps
avg_price = data.close.mean(axis=0)

# Total volume per symbol
total_vol = data.volume.sum(axis=0)

# Overall statistics
stats = data.close.mean()  # Scalar (single value)
```

---

## Missing Value Handling

### `data.dropna(axis=0, how="any", thresh=None)`

Remove rows or columns with missing values.

**Parameters:**
- `axis` (`int` or `str`): Axis along which to drop (0=timestamps, 1=symbols)
- `how` (`str`): "any" (drop if any NaN) or "all" (drop if all NaN)
- `thresh` (`int`, optional): Minimum number of non-NaN values required

**Returns:** `Data`

**Example:**
```python
# Drop timestamps with any missing values
clean = data.dropna(axis=0, how="any")

# Drop symbols with all missing values
clean = data.dropna(axis=1, how="all")
```

---

### `data.fillna(value=0.0, method=None, axis=0)`

Fill missing values.

**Parameters:**
- `value` (`float`): Value to fill NaN with (when `method=None`)
- `method` (`str`, optional): Fill method - `"ffill"` (forward fill) or `"bfill"` (backward fill)
- `axis` (`int` or `str`): Axis for fill methods

**Returns:** `Data`

**Examples:**
```python
# Fill with zero
filled = data.fillna(0.0)

# Forward fill (use previous value)
filled = data.fillna(method="ffill")

# Backward fill (use next value)
filled = data.fillna(method="bfill")
```

---

## Financial Metrics

### `data.annualized_return()`

Calculate annualized return.

**Formula:** `(last / first) ^ (1 / years) - 1`

**Returns:** `Data` with single timestamp dimension collapsed

**Alias:** `ar()`

---

### `data.annualized_volatility()`

Calculate annualized volatility (standard deviation of returns scaled to yearly).

**Returns:** `Data` with single timestamp dimension collapsed

**Alias:** `avol()`

---

### `data.annualized_sharpe_ratio()`

Calculate annualized Sharpe ratio.

**Formula:** `annualized_return / annualized_volatility`

**Returns:** `Data` with single timestamp dimension collapsed

**Alias:** `asr()`

---

### `data.maximum_drawdown()`

Calculate maximum drawdown (largest peak-to-trough decline).

**Returns:** `Data` with single timestamp dimension collapsed

**Alias:** `mdd()`

---

### `data.metrics()`

Calculate all metrics at once.

**Returns:** `Data` with columns:
- `annualized_sharpe_ratio`
- `annualized_return`
- `annualized_volatility`
- `maximum_drawdown`

**Example:**
```python
metrics = data.close.metrics()
print(metrics.to_dataframe())
```

---

## Resampling Methods

Downsample data to lower frequency. OHLC columns are handled specially:
- `open`: First valid value in window
- `high`: Maximum value in window
- `low`: Minimum value in window
- `close`: Last valid value in window
- Other columns: Sum by default

### Methods

| Method | Frequency |
|--------|-----------|
| `minutely()` | 1 minute |
| `hourly()` | 1 hour |
| `daily()` | 1 day |
| `weekly()` | 7 days |
| `monthly()` | 1 month |
| `yearly()` | 1 year |

**Parameters (all methods):**
- `origin` (`np.datetime64`, optional): Origin time for bucketing
- `offset` (`np.timedelta64`, optional): Timezone offset adjustment
- `aggregation_f` (callable): Aggregation function for non-OHLC columns

**Example:**
<!-- test:skip -->
```python
# Convert tick data to daily OHLCV
daily = tick_data.daily()

# Weekly data with timezone offset
weekly = data.weekly(offset=np.timedelta64(9, "h"))
```

---

### `data.downsample(delta, origin=None, offset=None, aggregation_f=nansum)`

Generic downsampling to arbitrary frequency.

**Parameters:**
- `delta` (`np.timedelta64`): Time interval for bucketing
- `origin` (`np.datetime64`, optional): Origin time
- `offset` (`np.timedelta64`, optional): Timezone offset
- `aggregation_f` (callable): Aggregation function

**Example:**
<!-- test:skip -->
```python
# 15-minute bars
bars_15m = data.downsample(np.timedelta64(15, "m"))
```

---

## Technical Indicators

### `data.moving_average(window=25, skipna=True)`

Calculate simple moving average.

**Parameters:**
- `window` (`int`): Window size
- `skipna` (`bool`): Skip NaN values

**Returns:** `Data`

---

### `data.bollinger_band(window=20, sigma=2.0, skipna=True)`

Calculate Bollinger Bands.

**Parameters:**
- `window` (`int`): Window size for moving average
- `sigma` (`float`): Number of standard deviations for bands

**Returns:** `Tuple[Data, Data, Data]` - (upper, middle, lower)

**Example:**
```python
upper, middle, lower = data.close.bollinger_band(window=20, sigma=2.0)
```

---

## Visualization Methods

All visualization methods require matplotlib (`pip install qfeval-data[plot]`).

### `data.plot(ax=None, **kwargs)`

Auto-detect plot type based on columns. Uses candlestick for OHLC data, line plot otherwise.

**Parameters:**
- `ax` (`matplotlib.axes.Axes`, optional): Axes to plot on

**Returns:** `List[matplotlib.axes.Axes]`

---

### `data.line(ax=None, even=False, **kwargs)`

Line plot.

**Parameters:**
- `ax` (`matplotlib.axes.Axes`, optional): Axes to plot on
- `even` (`bool`): Use even x-axis spacing (ignore time gaps)
- `**kwargs`: Passed to `matplotlib.plot()`

---

### `data.bar(width=0.8, bottom=0.0, ax=None, **kwargs)`

Bar plot.

**Parameters:**
- `width` (`float` or `Data`): Bar width
- `bottom` (`float` or `Data`): Bar bottom position
- `ax` (`matplotlib.axes.Axes`, optional): Axes to plot on

---

### `data.candlestick(ax=None, **kwargs)`

OHLC candlestick chart. Requires `open`, `high`, `low`, `close` columns.

**Parameters:**
- `ax` (`matplotlib.axes.Axes`, optional): Axes to plot on
- `upcolor` (`str`): Color for up candles (default: "#ee3333")
- `downcolor` (`str`): Color for down candles (default: "#118822")
- `neutralcolor` (`str`): Color for neutral candles (default: "#444444")
- `width` (`float`): Candle body width (default: 0.6)
- `linewidth` (`float`): Wick line width (default: 0.5)

**Example:**
<!-- test:skip -->
```python
import matplotlib.pyplot as plt
from qfeval_data import Data

data = Data.from_csv("prices.csv")
data.plot()
plt.show()
```

---

### `data.vlines(ymax=0.0, ax=None, **kwargs)`

Vertical lines plot.

---

### `data.fill_between(y2=0.0, ax=None, **kwargs)`

Fill area between curves.

---

### `data.plot_moving_average(window=25, ax=None, **kwargs)`

Plot moving average line.

---

### `data.plot_bollinger_band(window=20, sigma=2.0, ax=None, **kwargs)`

Plot Bollinger Bands with fill.

---

## Conversion Methods

### `data.to_dataframe()`

Convert to pandas DataFrame in long format.

**Returns:** `pd.DataFrame` with columns: `timestamp`, `symbol`, and all data columns

**Example:**
```python
df = data.to_dataframe()
#    timestamp symbol   open  close
# 0 2024-01-01   AAPL  150.0  155.0
# 1 2024-01-01   GOOG  140.0  145.0
```

---

### `data.to_table()`

Convert to pandas DataFrame in wide format (2D table).

**Returns:** `pd.DataFrame`

**Notes:**
- Single column: timestamps as index, symbols as columns
- Multiple columns: requires single timestamp or single symbol

---

### `data.to_series()`

Convert to pandas Series. Requires single column and single symbol.

**Returns:** `pd.Series`

---

### `data.to_csv(path=None)`

Export to CSV format.

**Parameters:**
- `path` (`str`, optional): File path. If None, returns CSV string.

**Returns:** `str` (if path is None) or `None`

---

### `data.to_matrix()`

Convert to DataFrame with timestamps as index and symbols as columns.

**Returns:** `pd.DataFrame`

---

### `data.to_matrix_csv(path=None)`

Export matrix format as CSV.

---

## Utility Methods

### `data.copy(deep=False)`

Create a copy.

**Parameters:**
- `deep` (`bool`): If True, copy tensors; otherwise, share tensor references

**Returns:** `Data`

---

### `data.to(dtype_or_device)`

Convert dtype and/or device.

**Signatures:**
```python
def to(self, dtype: torch.dtype) -> Data: ...
def to(self, device: torch.device) -> Data: ...
def to(self, tensor: torch.Tensor) -> Data: ...
def to(self, data: Data) -> Data: ...
```

**Examples:**
<!-- test:skip -->
```python
data_gpu = data.to("cuda")
data_f64 = data.to(torch.float64)
data_like = data.to(other_data)  # Match dtype/device
```

---

### `data.like(other)`

Reshape to match another Data's timestamps and symbols.

**Parameters:**
- `other` (`Data`): Reference Data for shape

**Returns:** `Data`

**Notes:**
- Missing timestamp/symbol combinations filled with NaN
- Extra combinations discarded

---

### `data.merge(*others)`

Merge multiple Data objects (union of timestamps/symbols).

**Parameters:**
- `*others` (`Data`): Data objects to merge

**Returns:** `Data`

**Notes:**
- For overlapping cells, last non-NaN value wins
- Columns with same name must have compatible shapes

---

### `data.merge_columns(other)`

Merge columns from another Data (same timestamps/symbols required).

**Parameters:**
- `other` (`Data`): Data with columns to add

**Returns:** `Data`

---

### `data.apply(f, *args, skipna=False)`

Apply function to tensors.

**Parameters:**
- `f` (callable): Function taking tensor(s) and returning tensor
- `*args`: Additional arguments (Data or values)
- `skipna` (`bool`): Skip NaN values

**Returns:** `Data`

**Example:**
<!-- test:skip -->
```python
# Apply custom function
result = data.apply(lambda x: torch.log(x + 1))

# With additional argument
result = data.apply(lambda x, y: x * y, other_data)
```

---

### `data.subsequences(start, stop, indexes=None)`

Extract time subsequences.

**Parameters:**
- `start` (`int`): Start offset from each timestamp
- `stop` (`int`): Stop offset from each timestamp
- `indexes`: Specific timestamps to extract from

**Returns:** `Data` with extra dimension for subsequence

---

### `data.zeros()`

Create Data with same shape filled with zeros.

**Returns:** `Data`

---

## Comparison Methods

### `data.equals(other)`

Check exact equality (including NaN positions).

**Returns:** `bool`

---

### `data.allclose(other, rtol=1e-5, atol=1e-8)`

Check approximate equality.

**Parameters:**
- `rtol` (`float`): Relative tolerance
- `atol` (`float`): Absolute tolerance

**Returns:** `bool`

---

## Index Conversion

### `data.timestamp_index(v, side="equal")`

Convert timestamp value(s) to integer indices.

**Returns:** `int` or `np.ndarray`

---

### `data.symbol_index(v, side="equal")`

Convert symbol value(s) to integer indices.

**Returns:** `int` or `np.ndarray`

---

### `data.has_timestamps()`

Check if Data has valid timestamps (not aggregated).

**Returns:** `bool`

---

### `data.has_symbols()`

Check if Data has valid symbols (not aggregated).

**Returns:** `bool`

---

### `data.size(dim=None)`

Get size of dimension(s).

**Parameters:**
- `dim` (`int` or `str`, optional): Specific dimension

**Returns:** `Tuple[int, int]` (if dim is None) or `int`

---

## Serialization

The `Data` class supports Python's pickle protocol:

<!-- test:skip -->
```python
import pickle

# Save
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

# Load
with open("data.pkl", "rb") as f:
    data = pickle.load(f)
```
