# Examples and Recipes

This document provides practical examples and common usage patterns for qfeval-data.

<!-- test:setup
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from qfeval_data import Data, Flattener

# Set up test data directory
_test_data_dir = Path(__file__).parent.parent / "tests" / "data" if "__file__" in dir() else Path("tests/data")
if not _test_data_dir.exists():
    _test_data_dir = Path("/Users/imos/git/qfeval-data/tests/data")
os.chdir(_test_data_dir)

# Create sample OHLCV data for examples
def create_sample_data():
    timestamps = np.array(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
         "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"],
        dtype="datetime64[D]",
    )
    symbols = np.array(["AAPL", "GOOG", "MSFT"])
    np.random.seed(42)
    n_ts, n_sym = len(timestamps), len(symbols)
    tensors = {
        "open": torch.tensor(np.random.randn(n_ts, n_sym) * 5 + 150, dtype=torch.float32),
        "high": torch.tensor(np.random.randn(n_ts, n_sym) * 5 + 155, dtype=torch.float32),
        "low": torch.tensor(np.random.randn(n_ts, n_sym) * 5 + 145, dtype=torch.float32),
        "close": torch.tensor(np.random.randn(n_ts, n_sym) * 5 + 152, dtype=torch.float32),
        "volume": torch.tensor(np.random.randn(n_ts, n_sym) * 1e5 + 1e6, dtype=torch.float32),
    }
    return Data.from_tensors(tensors, timestamps, symbols)

data = create_sample_data()
tick_data = data  # alias for examples
daily = data  # alias for examples

# Pre-create ML variables for later examples
import torch.nn as nn
import torch.optim as optim

feature = data.close
target = data.close.pct_change()
flattener = Flattener(feature, target)
X = flattener.flatten(feature).unsqueeze(-1)
y = flattener.flatten(target)

# Pre-trained simple model for examples
class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.layers(x).squeeze(-1)

model = Model(1)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()
for _ in range(5):
    optimizer.zero_grad()
    criterion(model(X), y).backward()
    optimizer.step()

# Pre-load data for visualization examples
prices = Data.from_csv("prices.csv")
aapl = prices[:, "AAPL"]
-->

## Table of Contents

1. [Loading Data](#loading-data)
2. [Basic Operations](#basic-operations)
3. [Time Series Analysis](#time-series-analysis)
4. [Portfolio Analysis](#portfolio-analysis)
5. [Data Transformation](#data-transformation)
6. [Visualization](#visualization)
7. [Machine Learning Integration](#machine-learning-integration)
8. [Working with Multiple Symbols](#working-with-multiple-symbols)

---

## Loading Data

### From CSV File

```python
from qfeval_data import Data

# Basic loading
data = Data.from_csv("prices.csv")

# With specific dtype
data = Data.from_csv("prices.csv", dtype=torch.float32)
```

**Expected CSV format:**
<!-- test:skip -->
```csv
timestamp,symbol,open,high,low,close,volume
2024-01-02,AAPL,185.5,186.2,184.1,185.8,50000000
2024-01-02,GOOG,140.0,141.5,139.5,141.0,20000000
2024-01-03,AAPL,186.0,187.5,185.0,186.5,48000000
2024-01-03,GOOG,141.0,142.0,140.0,141.5,19000000
```

### From pandas DataFrame

```python
import pandas as pd
from qfeval_data import Data

# Create sample data
df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=10, freq="D").repeat(2),
    "symbol": ["AAPL", "GOOG"] * 10,
    "close": [150 + i * 0.5 + (0 if i % 2 == 0 else 10) for i in range(20)],
})

data = Data.from_dataframe(df)
print(data.shape)  # (10, 2)
```

### From Raw Tensors

```python
import torch
import numpy as np
from qfeval_data import Data

# Create tensors
timestamps = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[D]")
symbols = np.array(["AAPL", "GOOG", "MSFT"])
prices = torch.randn(3, 3) * 10 + 100  # 3 timestamps x 3 symbols

data = Data.from_tensors({"close": prices}, timestamps, symbols)
```

### Multi-dimensional Data

```python
# Embedding vectors per timestamp/symbol
df = pd.DataFrame({
    "timestamp": ["2024-01-01", "2024-01-01"],
    "symbol": ["AAPL", "GOOG"],
    "embedding[0]": [0.1, 0.2],
    "embedding[1]": [0.3, 0.4],
    "embedding[2]": [0.5, 0.6],
})
data = Data.from_dataframe(df)
print(data.embedding.tensor.shape)  # (1, 2, 3)
```

---

## Basic Operations

### Accessing Data

```python
from qfeval_data import Data

# data is created in setup

# Get specific columns
closes = data.close                    # Attribute access
closes = data.get("close")             # Method access
ohlc = data.get("open", "high", "low", "close")

# Slice by time
first_week = data[:5, :]               # First 5 timestamps
jan_data = data["2024-01-01":"2024-01-12", :]

# Slice by symbol
apple = data[:, "AAPL"]                # Single symbol
tech = data[:, ["AAPL", "GOOG", "MSFT"]]  # Multiple symbols

# Combined slicing
apple_jan = data["2024-01-01":"2024-01-12", "AAPL"]
```

### Arithmetic

```python
# Returns
returns = data.close.pct_change()

# Log returns
log_returns = (data.close / data.close.shift(1)).apply(torch.log)

# Spread
spread = data.high - data.low

# Custom calculations
typical_price = (data.high + data.low + data.close) / 3
```

### Filtering

```python
# Boolean filtering
up_days = data[data.close > data.open]  # Non-matching become NaN

# Drop missing values
clean = data.dropna()

# Fill missing values
filled = data.fillna(method="ffill")
```

---

## Time Series Analysis

### Rolling Calculations

```python
# Moving average (window size <= data length)
ma_5 = data.close.moving_average(window=5)

# Bollinger Bands
upper, middle, lower = data.close.bollinger_band(window=5, sigma=2.0)
```

### Lagged Features

```python
# Previous values
prev_close = data.close.shift(1)
prev_5_close = data.close.shift(5)

# Future values (for targets)
next_close = data.close.shift(-1)
next_return = data.close.shift(-1).pct_change()
```

### Resampling

```python
# Daily data from tick data (no-op if already daily)
daily = tick_data.daily()

# Weekly OHLCV
weekly = daily.weekly()

# Monthly with timezone offset
monthly = daily.monthly(offset=np.timedelta64(9, "h"))

# Custom interval (2-day bars for daily data)
bars_2d = data.downsample(np.timedelta64(2, "D"))
```

---

## Portfolio Analysis

### Single Stock Metrics

```python
# Get metrics for a single stock
apple = data[:, "AAPL"]
metrics = apple.close.metrics()
print(metrics.to_dataframe())
```

### Cross-sectional Analysis

```python
# Compare metrics across all stocks
all_metrics = data.close.metrics()

# Find best Sharpe ratio
sharpe = all_metrics.get("annualized_sharpe_ratio")
best_idx = sharpe.tensor.argmax()
best_symbol = data.symbols[best_idx]
print(f"Best Sharpe: {best_symbol}")
```

### Portfolio Returns

```python
import torch

# Equal-weighted portfolio
weights = torch.ones(data.shape[1]) / data.shape[1]
portfolio_returns = (data.close.pct_change() * weights).sum(axis=1)

# Custom weights
weights = torch.tensor([0.4, 0.3, 0.3])  # AAPL, GOOG, MSFT
portfolio_returns = (data.close.pct_change() * weights).sum(axis=1)

# Portfolio cumulative return
cumulative = (1 + portfolio_returns).cumprod()
```

### Correlation Analysis

```python
# Calculate returns
returns = data.close.pct_change()

# Convert to numpy for correlation
returns_array = returns.dropna().array
import numpy as np
corr_matrix = np.corrcoef(returns_array.T)
print(pd.DataFrame(corr_matrix, index=data.symbols, columns=data.symbols))
```

---

## Data Transformation

### Normalization

```python
# Z-score normalization across time
mean = data.close.mean(axis=0)
std = data.close.std(axis=0)
normalized = (data.close - mean) / std

# Min-max normalization
min_val = data.close.min(axis=0)
max_val = data.close.max(axis=0)
scaled = (data.close - min_val) / (max_val - min_val)
```

### Creating Features

```python
def create_features(data):
    """Create common technical features."""
    features = []

    # Returns
    features.append(data.close.pct_change().rename("return_1d"))
    features.append(data.close.pct_change(3).rename("return_3d"))

    # Moving averages (window size <= data length)
    ma_3 = data.close.moving_average(3)
    ma_5 = data.close.moving_average(5)
    features.append((data.close / ma_3 - 1).rename("close_ma3_ratio"))
    features.append((data.close / ma_5 - 1).rename("close_ma5_ratio"))
    features.append((ma_3 / ma_5 - 1).rename("ma3_ma5_ratio"))

    # Volume ratio
    if "volume" in data.columns:
        vol_ma = data.volume.moving_average(5)
        features.append((data.volume / vol_ma).rename("volume_ratio"))

    # Merge all features
    result = features[0]
    for f in features[1:]:
        result = result.merge_columns(f)
    return result

features = create_features(data)
```

### Merging Data Sources

```python
# Merge multiple data sources
prices = Data.from_csv("prices.csv")
fundamentals = Data.from_csv("fundamentals.csv")

# Same timestamps/symbols - merge columns
combined = prices.merge_columns(fundamentals)

# Different timestamps/symbols - union merge
combined = prices.merge(fundamentals)
```

---

## Visualization

### Basic Plots

```python
import matplotlib.pyplot as plt
from qfeval_data import Data

prices = Data.from_csv("prices.csv")
aapl = prices[:, "AAPL"]

# Candlestick for OHLC data
aapl.candlestick()
plt.title("AAPL")
plt.close()
```

### Candlestick Chart

```python
# Explicit candlestick (using aapl from previous example)
aapl.candlestick()
plt.title("AAPL Candlestick")
plt.close()

# Custom colors
aapl.candlestick(
    upcolor="#00ff00",
    downcolor="#ff0000",
    width=0.8
)
plt.close()
```

### Line Plots

```python
# Single series
aapl.close.line()
plt.title("AAPL Close Price")
plt.close()

# Multiple series
fig, ax = plt.subplots()
aapl.close.line(ax=ax, label="Close")
aapl.close.moving_average(5).line(ax=ax, label="MA5")
plt.legend()
plt.close()
```

### Technical Indicators

```python
# Moving average overlay
fig, ax = plt.subplots()
aapl.candlestick(ax=ax)
aapl.close.plot_moving_average(window=5, ax=ax, color="blue")
plt.close()

# Bollinger Bands
fig, ax = plt.subplots()
aapl.candlestick(ax=ax)
aapl.close.plot_bollinger_band(window=5, ax=ax)
plt.close()
```

### Multiple Subplots

```python
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Price with Bollinger Bands
aapl.candlestick(ax=axes[0])
aapl.close.plot_bollinger_band(window=5, ax=axes[0])
axes[0].set_title("Price")

# Volume
aapl.volume.bar(ax=axes[1])
axes[1].set_title("Volume")

# Returns
aapl.close.pct_change().line(ax=axes[2])
axes[2].set_title("Daily Returns")

plt.tight_layout()
plt.close()
```

---

## Machine Learning Integration

### Preparing Data for PyTorch

```python
import torch
from qfeval_data import Data, Flattener

# data is created in setup

# Create feature and target (single column for Flattener)
feature = data.close
target = data.close.pct_change()  # Daily return

# Create flattener for alignment
flattener = Flattener(feature, target)

# Convert to tensors
X = flattener.flatten(feature).unsqueeze(-1)  # shape: (B, 1)
y = flattener.flatten(target)     # shape: (B,)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

### Training Loop

```python
import torch.nn as nn
import torch.optim as optim

# Simple model
class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)

model = Model(X.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training (short loop for example)
for epoch in range(10):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
```

### Making Predictions

```python
# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(X)

# Convert back to Data format
pred_data = flattener.unflatten(predictions, "prediction")

# View predictions shape
print(f"Predictions shape: {pred_data.shape}")
```

### Time Series Split

```python
# Split by time (using dates in the sample data)
split_date = "2024-01-08"
train_data = data[:split_date, :]
test_data = data[split_date:, :]

print(f"Train: {train_data.shape}, Test: {test_data.shape}")
```

---

## Working with Multiple Symbols

### Cross-sectional Operations

```python
# Rank within each timestamp
def rank_cross_section(d):
    """Rank values across symbols for each timestamp."""
    return d.apply(
        lambda x: x.argsort(dim=1).argsort(dim=1).float() / (x.shape[1] - 1)
    )

ranked = rank_cross_section(data.close.pct_change())
```

### Sector Analysis

```python
# Assuming you have sector mapping
sector_map = {"AAPL": "Tech", "GOOG": "Tech", "MSFT": "Tech"}
sectors = [sector_map.get(s, "Other") for s in data.symbols]

# Group by sector
tech_symbols = [s for s, sec in zip(data.symbols, sectors) if sec == "Tech"]
tech_data = data[:, tech_symbols]

# Sector average
tech_avg = tech_data.close.mean(axis=1).rename("tech_avg")
```

### Universe Filtering

```python
# Filter by liquidity (threshold adjusted for sample data)
avg_volume = data.volume.mean(axis=0)
liquid_mask = avg_volume.tensor > 900000
liquid_symbols = data.symbols[liquid_mask.cpu().numpy()]
liquid_data = data[:, liquid_symbols.tolist()]

# Filter by price
avg_price = data.close.mean(axis=0)
valid_mask = (avg_price.tensor > 5) & (avg_price.tensor < 1000)
valid_symbols = data.symbols[valid_mask.cpu().numpy()]
```

### Pair Trading

```python
# Calculate spread between two stocks
spread = data[:, "AAPL"].close - data[:, "GOOG"].close

# Normalize spread
spread_mean = spread.mean(axis=0)
spread_std = spread.std(axis=0)
zscore = (spread - spread_mean) / spread_std

# Generate signals
long_signal = zscore < -2    # Buy AAPL, sell GOOG
short_signal = zscore > 2    # Sell AAPL, buy GOOG
```
