# Utility Functions Reference

The `qfeval_data.util` module provides helper functions for array operations, time calculations, and other utilities.

## Overview

```python
from qfeval_data import util
```

---

## Array Operations

### `util.to_numpy(tensor)`

Convert a PyTorch tensor to a NumPy array.

**Parameters:**
- `tensor` (`torch.Tensor`): PyTorch tensor (can be on GPU or have gradients)

**Returns:** `np.ndarray`

**Example:**
```python
import torch
from qfeval_data import util

tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
array = util.to_numpy(tensor)
print(type(array))  # <class 'numpy.ndarray'>
```

**Notes:**
- Automatically detaches from computation graph
- Automatically moves from GPU to CPU if necessary

---

### `util.nans(shape=None, like=None)`

Create a tensor filled with NaN values.

**Parameters:**
- `shape` (`Tuple[int, ...]`, optional): Shape of the output tensor
- `like` (`torch.Tensor`): Reference tensor for dtype and device

**Returns:** `torch.Tensor`

**Example:**
```python
import torch
from qfeval_data import util

ref = torch.tensor([1.0, 2.0], device="cuda")
nans = util.nans((3, 4), like=ref)
print(nans.shape)   # torch.Size([3, 4])
print(nans.device)  # cuda:0
```

**Notes:**
- `like` parameter is required
- If `shape` is None, uses the shape of `like`

---

### `util.make_array_mapping(ref, like)`

Create index mapping between two sorted arrays.

**Parameters:**
- `ref` (`np.ndarray`): Reference array (sorted)
- `like` (`np.ndarray`): Array to map from (sorted)

**Returns:** `Tuple[np.ndarray, np.ndarray]`
- First array: indices where `ref[indices[i]] == like[i]`
- Second array: boolean mask where True indicates the mapping is invalid

**Example:**
```python
import numpy as np
from qfeval_data import util

ref = np.array(["A", "B", "C", "D"])
like = np.array(["B", "D", "E"])

indexes, mask = util.make_array_mapping(ref, like)
print(indexes)  # [1, 3, 0]  (index for E is 0, but masked)
print(mask)     # [False, False, True]  (E not in ref)
```

**Use case:** Aligning data from different sources with different symbols.

---

### `util.are_broadcastable_shapes(*shapes)`

Check if shapes are broadcastable for NumPy-like operations.

**Parameters:**
- `*shapes` (`Tuple[int, ...]` or `torch.Size`): Shapes to check

**Returns:** `bool`

**Example:**
```python
from qfeval_data import util

print(util.are_broadcastable_shapes((3, 4), (4,)))     # True
print(util.are_broadcastable_shapes((3, 4), (3, 1)))   # True
print(util.are_broadcastable_shapes((3, 4), (2, 4)))   # False
```

---

## Time Functions

### `util.floor_time(t, d, origin=None, offset=None)`

Floor datetime to a time interval.

**Parameters:**
- `t` (`np.datetime64` or `np.ndarray`): Timestamp(s) to floor
- `d` (`np.timedelta64`): Time interval
- `origin` (`np.datetime64`, optional): Origin for interval calculation
- `offset` (`np.timedelta64`, optional): Offset to apply before flooring

**Returns:** `np.datetime64` or `np.ndarray`

**Example:**
```python
import numpy as np
from qfeval_data import util

t = np.datetime64("2024-01-15T14:35:00")
d = np.timedelta64(1, "h")

floored = util.floor_time(t, d)
print(floored)  # 2024-01-15T14:00:00
```

**With offset (timezone adjustment):**
```python
# Floor to day boundary with 9-hour offset (JST timezone)
t = np.datetime64("2024-01-15T08:00:00")  # UTC
offset = np.timedelta64(9, "h")
floored = util.floor_time(t, np.timedelta64(1, "D"), offset=offset)
# Result considers JST day boundary
```

---

### `util.ceil_time(t, d, origin=None, offset=None)`

Ceil datetime to a time interval.

**Parameters:**
- `t` (`np.datetime64` or `np.ndarray`): Timestamp(s) to ceil
- `d` (`np.timedelta64`): Time interval
- `origin` (`np.datetime64`, optional): Origin for interval calculation
- `offset` (`np.timedelta64`, optional): Offset to apply before ceiling

**Returns:** `np.datetime64` or `np.ndarray`

**Example:**
```python
import numpy as np
from qfeval_data import util

t = np.datetime64("2024-01-15T14:35:00")
d = np.timedelta64(1, "h")

ceiled = util.ceil_time(t, d)
print(ceiled)  # 2024-01-15T15:00:00

# If already on boundary, returns same value
t2 = np.datetime64("2024-01-15T14:00:00")
print(util.ceil_time(t2, d))  # 2024-01-15T14:00:00
```

---

### `util.time_origin(d)`

Get the default time origin for a given interval.

**Parameters:**
- `d` (`np.timedelta64`): Time interval

**Returns:** `np.datetime64`

**Behavior:**
- For monthly/yearly intervals: returns `1000-01-01`
- For other intervals: returns `1893-01-01` (a Sunday, predating Dow Jones)

**Example:**
```python
import numpy as np
from qfeval_data import util

print(util.time_origin(np.timedelta64(1, "D")))  # 1893-01-01
print(util.time_origin(np.timedelta64(1, "M")))  # 1000-01-01
```

**Notes:**
- `1893-01-01` is chosen because:
  - It's a Sunday (useful for week calculations)
  - Predates Dow Jones Industrial Average (1896)
  - Allows sufficient range for nanosecond precision
- Weekly intervals use 7-day periods starting from Sunday

---

## Other Utilities

### `util.sha1(x)`

Compute SHA1 hash of various data types.

**Parameters:**
- `x` (`bytes`, `str`, `np.ndarray`, or `torch.Tensor`): Data to hash

**Returns:** `str` (hexadecimal hash)

**Example:**
```python
import numpy as np
import torch
from qfeval_data import util

print(util.sha1("hello"))  # aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d
print(util.sha1(np.array([1, 2, 3])))  # hash includes shape info
print(util.sha1(torch.tensor([1.0, 2.0])))  # works with tensors
```

**Notes:**
- For arrays/tensors, the hash includes shape information
- Tensors are automatically converted to NumPy before hashing

---

### `util.gc()`

Run garbage collection and clear GPU memory.

**Example:**
```python
from qfeval_data import util

# Free memory after processing
del large_data
util.gc()
```

**Behavior:**
- Runs Python garbage collection (generation 2)
- Clears CUDA cache if GPUs are available

---

### `util.torch_device(device)`

Parse device specification to `torch.device`.

**Parameters:**
- `device` (`str`, `torch.device`, or `None`): Device specification

**Returns:** `torch.device`

**Special values:**
- `None`: Returns CPU device
- `"auto"`: Returns CUDA if available, otherwise CPU
- `"cpu"`, `"cuda"`, `"cuda:0"`, etc.: Standard PyTorch device strings

**Example:**
```python
from qfeval_data import util

print(util.torch_device(None))     # cpu
print(util.torch_device("auto"))   # cuda or cpu
print(util.torch_device("cuda:0")) # cuda:0
```

---

## Type Variables

The module defines type variables for generic typing:

```python
# Generic type
T = typing.TypeVar("T")

# Array-like type (torch.Tensor or np.ndarray)
Array = typing.TypeVar("Array", torch.Tensor, np.ndarray)
```
