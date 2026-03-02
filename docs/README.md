# qfeval-data Documentation

[[日本語](README.ja.md)]

**qfeval-data** is a Python library for efficient manipulation of financial time series data. It provides the `Data` class—a specialized data structure built on PyTorch tensors for working with timestamped, symbol-indexed financial data.

## Key Features

- **PyTorch Backend**: Full GPU acceleration support via PyTorch tensors
- **Lazy Slicing**: Efficient data access without unnecessary copying
- **Financial Focus**: Built-in support for OHLCV data, metrics, and technical indicators
- **Flexible I/O**: Load from CSV, DataFrame, or construct from tensors
- **Visualization**: Integrated candlestick charts and plotting via matplotlib

## Installation

```bash
pip install qfeval-data

# With plotting support
pip install qfeval-data[plot]
```

## Quick Start

```python
from qfeval_data import Data
import pandas as pd

# Load from CSV
data = Data.from_csv("prices.csv")

# Or from DataFrame
df = pd.DataFrame({
    "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "symbol": ["AAPL", "GOOG", "AAPL", "GOOG"],
    "close": [150.0, 140.0, 152.0, 142.0],
})
data = Data.from_dataframe(df)

# Access and manipulate
returns = data.pct_change()
avg_return = returns.mean(axis=0)

# Calculate metrics
metrics = data.metrics()
print(metrics.to_dataframe())
```

## Documentation Contents

- [Data Class Reference](data.md) - Complete API reference for the `Data` class
- [Flattener Reference](flattener.md) - Converting between `Data` and flat tensors
- [Utility Functions](util.md) - Helper functions for arrays and time operations
- [Examples](examples.md) - Common usage patterns and recipes

## Requirements

- Python >= 3.9
- PyTorch
- NumPy
- pandas
- qfeval-functions

## License

See the [LICENSE](../LICENSE) file for details.
