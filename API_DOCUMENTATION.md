# qfeval-data API Documentation

## Overview

qfeval-data is a Python library developed by Preferred Networks' Financial Solutions team for efficiently handling financial time series data. It provides specialized data structures and functions for quantitative finance applications, built on top of PyTorch and pandas.

## Installation

```bash
pip install qfeval_data

# For plotting functionality
pip install qfeval_data[plot]
```

## Core Classes

### Data

The `Data` class is the central component of qfeval-data, managing numerical tensors indexed by timestamps and symbols. It's designed for efficient handling of financial time series data like OHLC (Open, High, Low, Close) data.

#### Constructor and Factory Methods

##### `Data(data: Data)`
Initializes a new Data object from an existing Data object.

**Parameters:**
- `data`: An existing Data object to copy from

**Note:** Use factory methods like `from_tensors`, `from_dataframe`, or `from_csv` for creating new Data objects.

##### `Data.from_tensors(tensors: Dict[str, torch.Tensor], timestamps: np.ndarray, symbols: np.ndarray) -> Data`
Creates a Data object from PyTorch tensors.

**Parameters:**
- `tensors`: Dictionary mapping column names to PyTorch tensors
- `timestamps`: Array of timestamps (np.datetime64)
- `symbols`: Array of symbol names (strings)

**Example:**
```python
import torch
import numpy as np
from qfeval_data import Data

# Create sample data
timestamps = np.array(['2023-01-01', '2023-01-02'], dtype='datetime64')
symbols = np.array(['AAPL', 'GOOGL'])
tensors = {
    'price': torch.tensor([[100.0, 200.0], [105.0, 195.0]]),
    'volume': torch.tensor([[1000, 2000], [1100, 1900]])
}

data = Data.from_tensors(tensors, timestamps, symbols)
```

##### `Data.from_dataframe(df: pd.DataFrame, dtype=None, device=None) -> Data`
Creates a Data object from a pandas DataFrame.

**Parameters:**
- `df`: DataFrame with 'timestamp' and 'symbol' columns
- `dtype`: PyTorch data type (optional)
- `device`: PyTorch device (optional, supports "auto", "cpu", "cuda")

**Example:**
```python
import pandas as pd
from qfeval_data import Data

df = pd.DataFrame({
    'timestamp': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
    'symbol': ['AAPL', 'GOOGL', 'AAPL', 'GOOGL'],
    'open': [100.0, 200.0, 105.0, 195.0],
    'close': [104.0, 198.0, 108.0, 192.0],
    'volume': [1000, 2000, 1100, 1900]
})

data = Data.from_dataframe(df)
```

##### `Data.from_csv(input: Union[str, io.IOBase], dtype=None, device=None) -> Data`
Creates a Data object from a CSV file.

**Parameters:**
- `input`: File path or file-like object
- `dtype`: PyTorch data type (optional)
- `device`: PyTorch device (optional)

**Example:**
```python
from qfeval_data import Data

# From file path
data = Data.from_csv('financial_data.csv')

# With specific device
data = Data.from_csv('financial_data.csv', device='cuda')
```

##### `Data.from_preset(name: str = "pfn-topix500", dtype=None, device=None, paths: List[str] = []) -> Data`
Loads preset financial datasets.

**Parameters:**
- `name`: Preset dataset name
- `dtype`: PyTorch data type (optional)
- `device`: PyTorch device (optional)
- `paths`: Additional search paths for datasets

#### Properties

##### `timestamps: np.ndarray`
Array of timestamps in the data.

##### `symbols: np.ndarray`
Array of symbol names in the data.

##### `columns: List[str]`
List of column names in the data.

##### `shape: Tuple[int, int]`
Shape of the data as (timestamps, symbols).

##### `device: torch.device`
PyTorch device where tensors are stored.

##### `dtype: torch.dtype`
Data type of the tensors.

##### `tensors: Dict[str, torch.Tensor]`
Dictionary of tensors with applied indexing.

##### `raw_tensors: Dict[str, torch.Tensor]`
Dictionary of raw tensors without indexing.

##### `tensor: torch.Tensor`
Single tensor (only available when data has exactly one column).

##### `raw_tensor: torch.Tensor`
Single raw tensor (only available when data has exactly one column).

##### `arrays: Dict[str, np.ndarray]`
Dictionary of NumPy arrays converted from tensors.

##### `array: np.ndarray`
Single NumPy array (only available when data has exactly one column).

#### Data Access and Manipulation

##### `get(*columns: str) -> Data`
##### `get(columns: Iterable[str]) -> Data`
##### `get(predicate: Callable[[str], bool]) -> Data`
##### `get(pattern: str = None) -> Data`
Returns a subset of columns.

**Examples:**
```python
# Get specific columns
price_data = data.get('open', 'close')

# Get columns by pattern
ohlc_data = data.get(pattern='*')

# Get columns by predicate
volume_data = data.get(lambda x: 'volume' in x)
```

##### `set(key: str, value: Union[torch.Tensor, Data]) -> None`
Sets or updates a column.

**Example:**
```python
# Add a new column
data.set('returns', data.close.pct_change())
```

##### `__getitem__(key) -> Data`
Slices the data by timestamp and/or symbol indices.

**Examples:**
```python
# Get first 10 timestamps
recent_data = data[:10]

# Get specific symbol
aapl_data = data[:, 'AAPL']

# Get date range
jan_data = data['2023-01-01':'2023-01-31']

# Boolean indexing
high_volume = data[data.volume > 1000]
```

##### `copy(deep: bool = False) -> Data`
Creates a copy of the data.

**Parameters:**
- `deep`: If True, creates a deep copy

#### Data Conversion

##### `to_dataframe() -> pd.DataFrame`
Converts to pandas DataFrame.

##### `to_table() -> pd.DataFrame`
Converts to a 2D table format.

##### `to_series() -> pd.Series`
Converts to pandas Series (for single symbol data).

##### `to_csv(path: Optional[str] = None) -> Optional[str]`
Exports to CSV format.

##### `to_matrix() -> pd.DataFrame`
Converts to matrix format with timestamps as index and symbols as columns.

#### Mathematical Operations

The Data class supports all standard mathematical operations:

```python
# Arithmetic operations
result = data1 + data2
result = data * 2
result = data / data.mean()

# Comparison operations
mask = data.close > data.open
expensive_stocks = data[data.close > 100]

# Logical operations
condition = (data.volume > 1000) & (data.close > data.open)
```

#### Aggregation Methods

##### `sum(axis: Optional[Axis] = None) -> Data`
##### `mean(axis: Optional[Axis] = None) -> Data`
##### `var(axis: Optional[Axis] = None, ddof: int = 1) -> Data`
##### `std(axis: Optional[Axis] = None, ddof: int = 1) -> Data`
##### `count(axis: Optional[Axis] = None) -> Data`
##### `skew(axis: Optional[Axis] = None, ddof: int = 1) -> Data`
##### `kurt(axis: Optional[Axis] = None, ddof: int = 1) -> Data`

Aggregation functions along timestamp, symbol, or column axes.

**Parameters:**
- `axis`: Aggregation axis ("timestamp", "symbol", "column", 0, 1, 2, or None)
- `ddof`: Delta degrees of freedom for variance calculations

**Examples:**
```python
# Daily averages across symbols
daily_avg = data.mean(axis='symbol')

# Symbol statistics across time
symbol_stats = data.std(axis='timestamp')

# Overall statistics
total_volume = data.volume.sum()
```

#### Time Series Operations

##### `shift(shift: int = 1, skipna: bool = False) -> Data`
Shifts data along the timestamp axis.

**Example:**
```python
# Get previous day's prices
prev_prices = data.close.shift(1)
returns = (data.close - prev_prices) / prev_prices
```

##### `pct_change(periods: int = 1, skipna: bool = False) -> Data`
Calculates percentage change.

##### `diff(periods: int = 1, skipna: bool = False) -> Data`
Calculates differences.

##### `cumsum(axis: Axis = 0, skipna: bool = True) -> Data`
##### `cumprod(axis: Axis = 0, skipna: bool = True) -> Data`
Cumulative operations.

#### Data Cleaning

##### `dropna(axis: Axis = 0, how: str = "any", thresh: Optional[int] = None) -> Data`
Removes missing values.

**Parameters:**
- `axis`: Axis along which to drop ("timestamp" or "symbol")
- `how`: "any" or "all"
- `thresh`: Minimum number of non-NA values required

##### `fillna(value: float = 0.0, method: Optional[str] = None, axis: Axis = 0) -> Data`
Fills missing values.

**Parameters:**
- `value`: Fill value
- `method`: "ffill" (forward fill) or "bfill" (backward fill)
- `axis`: Axis along which to fill

**Examples:**
```python
# Remove rows with any missing values
clean_data = data.dropna()

# Forward fill missing values
filled_data = data.fillna(method='ffill')
```

#### Downsampling Methods

##### `downsample(delta: np.timedelta64, origin: Optional[np.datetime64] = None, offset: Optional[np.timedelta64] = None, aggregation_f: Callable = functions.nansum) -> Data`
Downsamples data to specified frequency.

##### `minutely()`, `hourly()`, `daily()`, `weekly()`, `monthly()`, `yearly()`
Convenience methods for common downsampling frequencies.

**Example:**
```python
# Convert to daily data
daily_data = data.daily()

# Custom downsampling
weekly_data = data.downsample(np.timedelta64(7, 'D'))
```

#### Technical Analysis

##### `moving_average(window: int = 25, skipna: bool = True) -> Data`
Calculates moving average.

##### `bollinger_band(window: int = 20, sigma: float = 2.0, skipna: bool = True) -> Tuple[Data, Data, Data]`
Calculates Bollinger Bands, returning (upper, middle, lower).

**Examples:**
```python
# 20-day moving average
ma20 = data.close.moving_average(20)

# Bollinger Bands
upper, middle, lower = data.close.bollinger_band(20, 2.0)
```

#### Plotting Methods

##### `plot(ax: Optional[matplotlib.axes.Axes] = None, **kwargs) -> List[matplotlib.axes.Axes]`
Automatically plots appropriate chart type based on data columns.

##### `line(ax: Optional[matplotlib.axes.Axes] = None, even: bool = False, **kwargs) -> matplotlib.axes.Axes`
Line plot.

##### `bar(width: Union[float, Data] = 0.8, bottom: Union[float, Data] = 0.0, ax: Optional[matplotlib.axes.Axes] = None, even: bool = False, **kwargs) -> matplotlib.axes.Axes`
Bar plot.

##### `candlestick(ax: Optional[matplotlib.axes.Axes] = None, even: bool = False, **kwargs) -> matplotlib.axes.Axes`
Candlestick chart for OHLC data.

##### `plot_moving_average(window: int = 25, skipna: bool = True, ax: Optional[matplotlib.axes.Axes] = None, **kwargs) -> matplotlib.axes.Axes`
Plots moving average.

##### `plot_bollinger_band(window: int = 20, sigma: float = 2.0, skipna: bool = True, **kwargs) -> matplotlib.axes.Axes`
Plots Bollinger Bands.

**Examples:**
```python
import matplotlib.pyplot as plt

# Automatic plotting
data.plot()

# Candlestick chart
data.candlestick()

# Line plot with moving average
ax = data.close.line()
data.close.plot_moving_average(20, ax=ax)

plt.show()
```

#### Advanced Methods

##### `apply(f: Callable, *args, skipna: bool = False) -> Data`
Applies a function to tensors.

##### `like(other: Data) -> Data`
Reshapes data to match another Data object's shape.

##### `merge(*others: Data) -> Data`
Merges multiple Data objects.

##### `subsequences(start: int, stop: int, indexes=None) -> Data`
Extracts subsequences for each timestamp.

##### `randseqs(batch_size: int, length: int, skipna: bool = True) -> Iterator[RandSeqsResult]`
Generates random subsequences for machine learning.

### Flattener

The `Flattener` class helps convert between Data objects and PyTorch tensors for machine learning applications.

#### Constructor

##### `Flattener(*data: Data)`
Creates a flattener based on reference Data objects.

**Example:**
```python
from qfeval_data import Flattener

flattener = Flattener(data)
```

#### Methods

##### `flatten(data: Data) -> torch.Tensor`
Flattens Data to a 1D tensor suitable for ML models.

##### `unflatten(tensor: torch.Tensor, name: str = "") -> Data`
Converts a 1D tensor back to Data format.

##### `timestamp_indexes() -> torch.Tensor`
##### `symbol_indexes() -> torch.Tensor`
Returns index arrays for timestamps and symbols.

**Example:**
```python
# Prepare data for ML model
flattener = Flattener(train_data)
X = flattener.flatten(train_data)
y = flattener.flatten(target_data)

# Train model
model = MyModel()
predictions = model(X)

# Convert back to Data format
pred_data = flattener.unflatten(predictions, 'predictions')
```

## Utility Functions

### qfeval_data.util

#### `to_numpy(tensor: torch.Tensor) -> np.ndarray`
Converts PyTorch tensor to NumPy array, handling GPU tensors and gradients.

#### `torch_device(device) -> torch.device`
Resolves device specification, supporting "auto" for automatic GPU selection.

#### `ceil_time(t: Union[np.datetime64, np.ndarray], d: np.timedelta64, origin: Optional[np.datetime64] = None, offset: Optional[np.timedelta64] = None) -> Union[np.datetime64, np.ndarray]`
#### `floor_time(t: Union[np.datetime64, np.ndarray], d: np.timedelta64, origin: Optional[np.datetime64] = None, offset: Optional[np.timedelta64] = None) -> Union[np.datetime64, np.ndarray]`
Time rounding functions for temporal operations.

#### `time_origin(d: np.timedelta64) -> np.datetime64`
Returns appropriate time origin for given time delta.

#### `gc() -> None`
Performs garbage collection and clears GPU memory.

**Examples:**
```python
from qfeval_data import util
import torch

# Convert tensor to numpy
tensor = torch.tensor([1, 2, 3])
array = util.to_numpy(tensor)

# Device resolution
device = util.torch_device('auto')  # Selects GPU if available

# Time operations
import numpy as np
time = np.datetime64('2023-01-15T10:30:00')
day_start = util.floor_time(time, np.timedelta64(1, 'D'))
```

## Plotting

### qfeval_data.plot

#### `Figure`
Enhanced matplotlib figure wrapper for financial data visualization.

##### `Figure(figure: Union[None, matplotlib.axes.Axes, matplotlib.figure.Figure] = None)`

##### Properties and Methods:
- `figure: matplotlib.figure.Figure`
- `axes: List[matplotlib.axes.Axes]`
- `primary_axes: matplotlib.axes.Axes`
- `show() -> None`
- `append_axes(scale: float = 0.4) -> matplotlib.axes.Axes`

#### `plot_dataframe(df: pd.DataFrame, **kwargs) -> List[matplotlib.axes.Axes]`
Plots pandas DataFrame with financial data styling.

**Example:**
```python
from qfeval_data import plot
import matplotlib.pyplot as plt

# Create multi-panel plot
fig = plot.Figure()
data.close.line(ax=fig.primary_axes)
data.volume.bar(ax=fig.append_axes())
fig.show()
```

## Complete Usage Examples

### Basic Financial Data Analysis

```python
import pandas as pd
import numpy as np
from qfeval_data import Data

# Load data
df = pd.read_csv('stock_data.csv')
data = Data.from_dataframe(df)

# Basic analysis
print(f"Data shape: {data.shape}")
print(f"Symbols: {data.symbols}")
print(f"Date range: {data.timestamps[0]} to {data.timestamps[-1]}")

# Calculate returns
returns = data.close.pct_change()
daily_returns = returns.mean(axis='symbol')

# Technical indicators
ma20 = data.close.moving_average(20)
upper, middle, lower = data.close.bollinger_band()

# Plotting
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

data.close.line(ax=ax1, label='Close Price')
ma20.line(ax=ax1, label='20-day MA')
ax1.legend()

data.volume.bar(ax=ax2)
plt.show()
```

### Machine Learning Pipeline

```python
from qfeval_data import Data, Flattener
import torch
import torch.nn as nn

# Prepare data
train_data = Data.from_csv('train_data.csv')
test_data = Data.from_csv('test_data.csv')

# Create features
features = train_data.get('open', 'high', 'low', 'close', 'volume')
features = features.fillna(method='ffill')

# Add technical indicators
features.set('ma5', features.close.moving_average(5))
features.set('ma20', features.close.moving_average(20))
features.set('returns', features.close.pct_change())

# Prepare for ML
flattener = Flattener(features)
X_train = flattener.flatten(features)
y_train = flattener.flatten(train_data.close.shift(-1))  # Next day price

# Define model
class PricePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.linear(x)

# Train model
model = PricePredictor(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

for epoch in range(100):
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Make predictions
X_test = flattener.flatten(test_data)
predictions = model(X_test)
pred_data = flattener.unflatten(predictions, 'predicted_price')
```

### Advanced Time Series Analysis

```python
# Multi-timeframe analysis
daily_data = data.daily()
weekly_data = data.weekly()
monthly_data = data.monthly()

# Cross-sectional analysis
sector_performance = data.mean(axis='timestamp')
time_series_stats = data.std(axis='symbol')

# Risk analysis
returns = data.close.pct_change()
volatility = returns.std(axis='timestamp') * np.sqrt(252)  # Annualized
sharpe_ratio = returns.mean(axis='timestamp') / returns.std(axis='timestamp')

# Portfolio analysis
weights = Data.from_tensors(
    {'weight': torch.ones(1, len(data.symbols)) / len(data.symbols)},
    data.timestamps[:1],
    data.symbols
)

portfolio_returns = (returns * weights).sum(axis='symbol')
portfolio_value = (1 + portfolio_returns).cumprod()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Price chart with Bollinger Bands
data.close.plot_bollinger_band(ax=axes[0,0])
axes[0,0].set_title('Price with Bollinger Bands')

# Volume
data.volume.bar(ax=axes[0,1])
axes[0,1].set_title('Volume')

# Returns distribution
returns.plot(ax=axes[1,0])
axes[1,0].set_title('Returns')

# Portfolio performance
portfolio_value.line(ax=axes[1,1])
axes[1,1].set_title('Portfolio Value')

plt.tight_layout()
plt.show()
```

## Best Practices

1. **Memory Management**: Use `util.gc()` for large datasets to free GPU memory
2. **Device Management**: Use `device='auto'` for automatic GPU selection
3. **Data Validation**: Always check data shapes and handle missing values
4. **Performance**: Use `skipna=True` for faster computations when appropriate
5. **Visualization**: Use the built-in plotting methods for consistent styling

## Error Handling

Common errors and solutions:

- **Shape Mismatch**: Ensure timestamps and symbols align when merging data
- **Device Mismatch**: Keep all tensors on the same device
- **Missing Values**: Use `fillna()` or `dropna()` before calculations
- **Memory Issues**: Use `util.gc()` and consider smaller batch sizes

## Integration with Other Libraries

qfeval-data integrates seamlessly with:
- **PyTorch**: Direct tensor operations and GPU support
- **pandas**: Easy conversion to/from DataFrames
- **NumPy**: Array operations and datetime handling
- **matplotlib**: Built-in plotting capabilities
- **scikit-learn**: Via flattener for ML pipelines