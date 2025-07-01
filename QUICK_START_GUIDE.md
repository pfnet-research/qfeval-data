# qfeval-data Quick Start Guide

## What is qfeval-data?

qfeval-data is a high-performance Python library for financial time series analysis, built on PyTorch. It provides:

- **Efficient Data Structures**: Handle large financial datasets with GPU acceleration
- **Time Series Operations**: Built-in functions for returns, moving averages, and technical indicators
- **Flexible Indexing**: Slice data by timestamps, symbols, or conditions
- **Visualization**: Create professional financial charts with minimal code
- **ML Integration**: Seamless integration with PyTorch for machine learning

## 5-Minute Quick Start

### Installation

```bash
pip install qfeval_data
pip install qfeval_data[plot]  # For plotting features
```

### Basic Usage

```python
import pandas as pd
import numpy as np
from qfeval_data import Data

# 1. Load your financial data
df = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
    'symbol': ['AAPL'] * 100,
    'open': np.random.randn(100).cumsum() + 100,
    'high': np.random.randn(100).cumsum() + 105,
    'low': np.random.randn(100).cumsum() + 95,
    'close': np.random.randn(100).cumsum() + 102,
    'volume': np.random.randint(1000, 10000, 100)
})

# 2. Convert to qfeval-data format
data = Data.from_dataframe(df)

# 3. Basic analysis
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns}")

# 4. Calculate returns
returns = data.close.pct_change()
print(f"Average daily return: {returns.mean().array:.4f}")

# 5. Technical indicators
ma20 = data.close.moving_average(20)
upper, middle, lower = data.close.bollinger_band(20, 2.0)

# 6. Plot results
data.plot()  # Automatic OHLC chart with volume
```

## Key Concepts

### Data Structure

```python
# Data is organized as (timestamps × symbols × features)
print(data.shape)  # (100, 1) - 100 days, 1 symbol
print(data.columns)  # ['open', 'high', 'low', 'close', 'volume']
```

### Indexing and Slicing

```python
# Time-based slicing
recent = data[-30:]  # Last 30 days
jan_data = data['2023-01-01':'2023-01-31']

# Symbol-based slicing (for multi-symbol data)
aapl_only = data[:, 'AAPL']

# Conditional filtering
high_volume = data[data.volume > 5000]
```

### Column Access

```python
# Access columns as attributes
prices = data.close
volumes = data.volume

# Or use get() method
ohlc = data.get('open', 'high', 'low', 'close')
```

## Common Workflows

### 1. Load and Explore Data

```python
# From CSV
data = Data.from_csv('stock_data.csv')

# Basic info
print(f"Date range: {data.timestamps[0]} to {data.timestamps[-1]}")
print(f"Symbols: {list(data.symbols)}")
print(f"Missing values: {data.count()}")

# Summary statistics
print(data.describe())  # If you add this method, or use:
print(f"Mean close: {data.close.mean()}")
print(f"Volatility: {data.close.std()}")
```

### 2. Technical Analysis

```python
# Price indicators
sma_20 = data.close.moving_average(20)
sma_50 = data.close.moving_average(50)

# Bollinger Bands
upper, middle, lower = data.close.bollinger_band(20, 2.0)

# Returns and volatility
returns = data.close.pct_change()
rolling_vol = returns.std() * np.sqrt(252)  # Annualized volatility

# Add indicators to data
data.set('sma_20', sma_20)
data.set('returns', returns)
```

### 3. Multi-Symbol Analysis

```python
# Assuming multi-symbol data
portfolio_data = Data.from_csv('portfolio_data.csv')

# Cross-sectional analysis
avg_returns = portfolio_data.close.pct_change().mean(axis='timestamp')
correlations = portfolio_data.close.pct_change().apply(np.corrcoef)

# Portfolio metrics
equal_weights = torch.ones(1, len(portfolio_data.symbols)) / len(portfolio_data.symbols)
portfolio_returns = (portfolio_data.close.pct_change() * equal_weights).sum(axis='symbol')
```

### 4. Time Series Resampling

```python
# Convert high-frequency to lower frequency
daily_data = data.daily()      # If data is intraday
weekly_data = data.weekly()
monthly_data = data.monthly()

# Custom resampling
data_2d = data.downsample(np.timedelta64(2, 'D'))  # Every 2 days
```

### 5. Machine Learning Preparation

```python
from qfeval_data import Flattener

# Prepare features
features = data.get('open', 'high', 'low', 'close', 'volume')
features.set('returns', features.close.pct_change())
features.set('ma_20', features.close.moving_average(20))

# Handle missing values
features = features.fillna(method='ffill')

# Flatten for ML
flattener = Flattener(features)
X = flattener.flatten(features)
y = flattener.flatten(features.close.shift(-1))  # Next day target

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
```

### 6. Visualization

```python
import matplotlib.pyplot as plt

# Automatic plotting based on data type
data.plot()

# Custom plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Price with moving averages
data.close.line(ax=ax1, label='Close')
data.close.moving_average(20).line(ax=ax1, label='MA20')
ax1.legend()

# Volume
data.volume.bar(ax=ax2)

# Bollinger Bands
data.close.plot_bollinger_band()

plt.show()
```

## Performance Tips

### GPU Acceleration

```python
# Use GPU for large datasets
data_gpu = Data.from_csv('large_dataset.csv', device='cuda')

# Or use automatic device selection
data_auto = Data.from_csv('large_dataset.csv', device='auto')

# Check device
print(f"Data on device: {data.device}")
```

### Memory Management

```python
from qfeval_data import util

# Free GPU memory when needed
util.gc()

# Use appropriate data types
data_fp16 = data.to(torch.float16)  # Half precision for memory savings
```

### Efficient Operations

```python
# Use skipna=True for faster computations when appropriate
fast_ma = data.close.moving_average(20, skipna=True)

# Batch operations instead of loops
returns_batch = data.get('open', 'high', 'low', 'close').pct_change()
```

## Common Patterns

### Pattern 1: Daily Analysis Pipeline

```python
def daily_analysis(symbol):
    # Load data
    data = Data.from_csv(f'{symbol}_data.csv')
    
    # Calculate indicators
    data.set('sma_20', data.close.moving_average(20))
    data.set('sma_50', data.close.moving_average(50))
    data.set('returns', data.close.pct_change())
    
    # Generate signals
    signals = (data.sma_20 > data.sma_50) & (data.returns > 0)
    
    # Plot
    ax = data.close.line(label='Price')
    data.sma_20.line(ax=ax, label='SMA20')
    data.sma_50.line(ax=ax, label='SMA50')
    
    return data, signals
```

### Pattern 2: Portfolio Backtesting

```python
def backtest_strategy(data, signals):
    # Calculate strategy returns
    strategy_returns = data.close.pct_change() * signals.shift(1)
    
    # Performance metrics
    total_return = (1 + strategy_returns).cumprod()
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    
    # Plot performance
    total_return.line(label='Strategy')
    (1 + data.close.pct_change()).cumprod().line(label='Buy & Hold')
    
    return {
        'total_return': total_return.array[-1],
        'sharpe_ratio': sharpe_ratio.array,
        'max_drawdown': calculate_max_drawdown(total_return)
    }
```

### Pattern 3: Multi-Timeframe Analysis

```python
def multi_timeframe_analysis(data):
    # Different timeframes
    daily = data.daily()
    weekly = data.weekly()
    monthly = data.monthly()
    
    # Align timeframes
    weekly_aligned = weekly.like(daily)
    monthly_aligned = monthly.like(daily)
    
    # Combined signals
    daily_signal = daily.close > daily.close.moving_average(20)
    weekly_signal = weekly_aligned.close > weekly_aligned.close.moving_average(4)
    
    combined_signal = daily_signal & weekly_signal
    
    return combined_signal
```

## Next Steps

1. **Read the Full API Documentation**: See `API_DOCUMENTATION.md` for comprehensive details
2. **Explore Examples**: Check the `tests/` directory for more usage patterns
3. **Join the Community**: Contribute to the project or ask questions
4. **Optimize Performance**: Learn about GPU acceleration and memory optimization
5. **Build Applications**: Create trading strategies, risk models, or analytical tools

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you have the required dependencies
   ```bash
   pip install torch pandas numpy matplotlib
   ```

2. **GPU Issues**: Check CUDA availability
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

3. **Memory Issues**: Use smaller batch sizes or lower precision
   ```python
   data = data.to(torch.float16)  # Use half precision
   util.gc()  # Free memory
   ```

4. **Date Parsing**: Ensure timestamp column is properly formatted
   ```python
   df['timestamp'] = pd.to_datetime(df['timestamp'])
   ```

### Getting Help

- **Documentation**: Full API reference in `API_DOCUMENTATION.md`
- **Examples**: Look at test files for usage patterns
- **Issues**: Report bugs or request features on GitHub
- **Performance**: Use profiling tools to identify bottlenecks

Happy analyzing! 🚀📈