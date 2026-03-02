# AGENTS.md

Guidelines for AI agents working with the qfeval-data repository.

## Repository Overview

**qfeval-data** is a Python library for handling financial time series data. It provides the `Data` class—a specialized data structure built on PyTorch tensors for efficient manipulation of timestamped, symbol-indexed financial data (OHLCV).

For detailed specifications, see the `docs/` directory:
- `docs/README.md` - Documentation index (Japanese: `docs/README.ja.md`)
- `docs/data.md` - Complete Data class API reference (Japanese: `docs/data.ja.md`)
- `docs/flattener.md` - Flattener class reference (Japanese: `docs/flattener.ja.md`)
- `docs/util.md` - Utility functions reference (Japanese: `docs/util.ja.md`)
- `docs/examples.md` - Practical examples and recipes (Japanese: `docs/examples.ja.md`)

## Codebase Structure

```
qfeval-data/
├── qfeval_data/           # Main package
│   ├── __init__.py        # Exports: Data, Flattener, __version__
│   ├── data.py            # Core Data class
│   ├── flattener.py       # Tensor flattening utilities
│   ├── util.py            # Helper functions
│   ├── plot.py            # Visualization (requires matplotlib)
│   └── version.py         # Version string
├── tests/                 # pytest test suite
└── pyproject.toml         # Project configuration
```

## Development Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=qfeval_data

# Linting and formatting
black qfeval_data tests
isort qfeval_data tests
flake8 qfeval_data tests

# Type checking
mypy qfeval_data
```

## Code Style

- Formatter: `black` (line-length: 80)
- Import sorting: `isort`
- Linting: `flake8`
- Type checking: `mypy` (strict mode)

## Key Design Patterns

1. **Lazy slicing**: Data slicing creates views without copying tensors
2. **Sorted indexes**: Timestamps and symbols are always sorted internally
3. **Method chaining**: Most methods return `Data` for fluent API
4. **PyTorch backend**: Full GPU support via tensor operations

---

## Using qfeval_data.Data (PyPI Package)

This section is for agents that consume qfeval-data as a dependency.

### Installation

```bash
pip install qfeval-data

# With plotting support
pip install qfeval-data[plot]
```

### Core Concepts

The `Data` class wraps a dictionary of PyTorch tensors indexed by:
- **timestamps**: `np.ndarray[datetime64]` (sorted)
- **symbols**: `np.ndarray[str]` (sorted)
- **columns**: Named tensors with shape `(num_timestamps, num_symbols, *extra_dims)`

### Creating Data Objects

```python
from qfeval_data import Data
import pandas as pd
import numpy as np
import torch

# From pandas DataFrame (requires "timestamp" and "symbol" columns)
df = pd.DataFrame({
    "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "symbol": ["AAPL", "GOOG", "AAPL", "GOOG"],
    "open": [100.0, 200.0, 101.0, 201.0],
    "close": [105.0, 205.0, 106.0, 206.0],
})
data = Data.from_dataframe(df)

# From CSV file
data = Data.from_csv("prices.csv")

# From tensors directly
tensors = {
    "open": torch.tensor([[100.0, 200.0], [101.0, 201.0]]),
    "close": torch.tensor([[105.0, 205.0], [106.0, 206.0]]),
}
timestamps = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
symbols = np.array(["AAPL", "GOOG"])
data = Data.from_tensors(tensors, timestamps, symbols)

# From preset data files (searches sys.path for data/{name}.csv)
data = Data.from_preset("pfn-topix500")
```

### Accessing Data

```python
# Column access
opens = data.get("open")           # Single column
ohlc = data.get(["open", "high", "low", "close"])  # Multiple columns
data.open                          # Attribute access shortcut

# Slicing (lazy - no data copy)
subset = data[:10, :]              # First 10 timestamps
subset = data["2024-01-01", :]     # By timestamp value
subset = data[:, "AAPL"]           # Single symbol
subset = data[:, ["AAPL", "GOOG"]] # Multiple symbols

# Properties
data.timestamps   # np.ndarray of timestamps
data.symbols      # np.ndarray of symbols
data.columns      # List of column names
data.shape        # (num_timestamps, num_symbols)
data.tensors      # Dict[str, Tensor] after slicing
data.tensor       # Single tensor (when only one column)
```

### Arithmetic Operations

All arithmetic is element-wise on tensors:

```python
returns = (data.close / data.open) - 1
spread = data.high - data.low
mask = data.close > data.open  # Boolean Data
```

### Time Series Operations

```python
data.shift(1)              # Shift forward by 1 timestamp
data.pct_change()          # Percent change
data.diff()                # Difference
data.cumsum()              # Cumulative sum
data.moving_average(20)    # 20-period moving average
```

### Aggregation (axis: 0=timestamp, 1=symbol, None=both)

```python
data.mean(axis=0)     # Mean across timestamps
data.sum(axis=1)      # Sum across symbols
data.std()            # Std dev across all
data.min(axis=0)
data.max(axis=0)
data.count()          # Count non-NaN
```

### Missing Value Handling

```python
data.dropna(axis=0, how="any")   # Drop timestamps with any NaN
data.fillna(0.0)                 # Fill NaN with value
data.fillna(method="ffill")      # Forward fill
```

### Financial Metrics

```python
data.annualized_return()
data.annualized_volatility()
data.annualized_sharpe_ratio()
data.maximum_drawdown()
data.metrics()  # All metrics combined
```

### Resampling

```python
data.daily()
data.weekly()
data.monthly()
data.yearly()
```

### Conversion

```python
df = data.to_dataframe()     # pandas DataFrame (long format)
csv = data.to_csv()          # CSV string
data.to_csv("output.csv")    # Write to file
```

### Device and Dtype

```python
data.to("cuda")              # Move to GPU
data.to(torch.float64)       # Change dtype
data.device                  # Current device
data.dtype                   # Current dtype
```

### Method Chaining Example

```python
result = (
    Data.from_csv("prices.csv")
    .get(["open", "close"])
    .dropna()
    .pct_change()
    .fillna(0.0)
    .mean(axis=1)
)
```

### Flattener Utility

Convert between `Data` (timestamp/symbol indexed) and flat `Tensor` (batch indexed):

```python
from qfeval_data import Flattener

flattener = Flattener(data)
flat_tensor = flattener.flatten(data)      # Data -> Tensor
restored = flattener.unflatten(flat_tensor, "prices")  # Tensor -> Data
```

### Further Documentation

For complete API documentation and more examples, see:
- `docs/data.md` - Full Data class reference with all methods and parameters
- `docs/flattener.md` - Flattener class details
- `docs/examples.md` - Practical recipes for common tasks
