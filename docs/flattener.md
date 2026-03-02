# Flattener Class Reference

The `Flattener` class assists conversion between `Data` objects (with timestamp/symbol indices) and flat `torch.Tensor` objects (with a single batch index).

<!-- test:setup
import numpy as np
import pandas as pd
import torch
from qfeval_data import Data, Flattener

# Create sample data for examples
def create_sample_data():
    timestamps = np.array(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        dtype="datetime64[D]",
    )
    symbols = np.array(["AAPL", "GOOG"])
    tensors = {
        "open": torch.tensor([[100.0, 200.0], [101.0, 201.0], [102.0, 202.0], [103.0, 203.0]]),
        "high": torch.tensor([[105.0, 205.0], [106.0, 206.0], [107.0, 207.0], [108.0, 208.0]]),
        "low": torch.tensor([[98.0, 198.0], [99.0, 199.0], [100.0, 200.0], [101.0, 201.0]]),
        "close": torch.tensor([[104.0, 204.0], [105.0, 205.0], [106.0, 206.0], [107.0, 207.0]]),
        "volume": torch.tensor([[1e6, 5e5], [1.1e6, 5.5e5], [1.2e6, 6e5], [1.3e6, 6.5e5]]),
    }
    return Data.from_tensors(tensors, timestamps, symbols)

data = create_sample_data()
prices = data
features = data.get(["open", "high", "low", "close"])
-->

## Overview

<!-- test:skip -->
```python
from qfeval_data import Flattener
```

The Flattener is useful when you need to:
- Convert financial data to batch format for machine learning models
- Work with flat tensor representations
- Convert model outputs back to timestamp/symbol indexed format

## Constructor

### `Flattener(*data)`

Create a Flattener from one or more Data objects.

**Parameters:**
- `*data` (`Data`): One or more Data objects that define the flattening mask

**Behavior:**
- Creates a mask of valid (non-NaN) timestamp/symbol pairs
- All input Data objects must have the same timestamps and symbols
- A pair is considered valid if it has no NaN values across all input Data

**Example:**
<!-- test:skip -->
```python
from qfeval_data import Data, Flattener

data = Data.from_csv("prices.csv")
flattener = Flattener(data)
```

**With multiple Data objects:**
```python
# Flattener will only include pairs valid in BOTH datasets
flattener = Flattener(prices, features)
```

---

## Methods

### `flattener.flatten(data)`

Convert a Data object to a flat tensor.

**Parameters:**
- `data` (`Data`): Data object to flatten (must have same timestamps/symbols as constructor input)

**Returns:** `torch.Tensor` with shape `(batch_size, *extra_dims)`

**Shape transformation:**
- Input Data shape: `(T, S, *extra_dims)` where T=timestamps, S=symbols
- Output tensor shape: `(B, *extra_dims)` where B=number of valid pairs

**Example:**
<!-- test:skip -->
```python
data = Data.from_csv("prices.csv")
flattener = Flattener(data)

# Flatten to batch tensor
flat_tensor = flattener.flatten(data.close)
print(flat_tensor.shape)  # (B,) where B = number of valid timestamp/symbol pairs
```

**Notes:**
- Only valid (non-NaN) pairs are included in the output
- The order of elements follows row-major order (timestamp varies slowest)

---

### `flattener.unflatten(tensor, name="")`

Convert a flat tensor back to a Data object.

**Parameters:**
- `tensor` (`torch.Tensor`): Flat tensor with shape `(batch_size, *extra_dims)`
- `name` (`str`): Column name for the returned Data object

**Returns:** `Data` with shape matching the original timestamps/symbols

**Shape transformation:**
- Input tensor shape: `(B, *extra_dims)`
- Output Data shape: `(T, S, *extra_dims)`

**Example:**
<!-- test:skip -->
```python
# After processing...
output_tensor = model(flat_tensor)  # shape: (B,)

# Convert back to Data
predictions = flattener.unflatten(output_tensor, name="prediction")
print(predictions.shape)  # (T, S)
```

**Notes:**
- Invalid pairs (those that were masked during flattening) are filled with NaN
- Tensor batch size must match the number of valid pairs from construction

---

### `flattener.timestamp_indexes()`

Get the timestamp index for each element in the flattened representation.

**Returns:** `torch.Tensor` with shape `(batch_size,)`

**Example:**
<!-- test:skip -->
```python
ts_idx = flattener.timestamp_indexes()
# ts_idx[i] = timestamp index of the i-th element in flattened tensor
```

---

### `flattener.symbol_indexes()`

Get the symbol index for each element in the flattened representation.

**Returns:** `torch.Tensor` with shape `(batch_size,)`

**Example:**
<!-- test:skip -->
```python
sym_idx = flattener.symbol_indexes()
# sym_idx[i] = symbol index of the i-th element in flattened tensor
```

---

## Complete Example

<!-- test:skip -->
```python
import torch
from qfeval_data import Data, Flattener

# Load data
data = Data.from_csv("prices.csv")
print(f"Original shape: {data.shape}")  # e.g., (252, 100)

# Create flattener
flattener = Flattener(data)

# Flatten closing prices
prices = flattener.flatten(data.close)
print(f"Flattened shape: {prices.shape}")  # e.g., (25000,)

# Do some processing
log_prices = torch.log(prices)

# Unflatten back to Data
result = flattener.unflatten(log_prices, "log_price")
print(f"Result shape: {result.shape}")  # (252, 100)

# Get index mapping
ts_idx = flattener.timestamp_indexes()
sym_idx = flattener.symbol_indexes()
print(f"First element: timestamp={ts_idx[0].item()}, symbol={sym_idx[0].item()}")
```

---

## Machine Learning Workflow

<!-- test:skip -->
```python
import torch
import torch.nn as nn
from qfeval_data import Data, Flattener

# Load and prepare data
data = Data.from_csv("prices.csv")
features = data.get(["open", "high", "low", "close", "volume"])
target = data.close.shift(-1).pct_change()  # Next day return

# Create flattener from both (ensures alignment)
flattener = Flattener(features, target)

# Flatten for training
X = flattener.flatten(features)  # shape: (B, 5)
y = flattener.flatten(target)    # shape: (B,)

# Train model
model = nn.Linear(5, 1)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    pred = model(X).squeeze()
    loss = ((pred - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Make predictions and unflatten
with torch.no_grad():
    predictions = model(X).squeeze()
    pred_data = flattener.unflatten(predictions, "prediction")

# Now pred_data has the same timestamp/symbol structure as original data
print(pred_data.to_dataframe())
```

---

## Notes

1. **Memory Efficiency**: Flattening creates a contiguous tensor by copying valid elements, not a view
2. **NaN Handling**: Only non-NaN elements are included in the flattened representation
3. **Device Consistency**: The flattener operates on the same device as the input Data
4. **Batch Dimension**: The flattened tensor combines timestamp and symbol dimensions into a single batch dimension
