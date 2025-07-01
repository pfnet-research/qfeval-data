# qfeval-data Method Reference

## Data Class Methods

### Factory Methods (Class Methods)
```python
Data.from_tensors(tensors: Dict[str, torch.Tensor], timestamps: np.ndarray, symbols: np.ndarray) -> Data
Data.from_dataframe(df: pd.DataFrame, dtype=None, device=None) -> Data
Data.from_csv(input: Union[str, io.IOBase], dtype=None, device=None) -> Data
Data.from_preset(name: str = "pfn-topix500", dtype=None, device=None, paths: List[str] = []) -> Data
```

### Properties
```python
data.timestamps: np.ndarray          # Array of timestamps
data.symbols: np.ndarray             # Array of symbol names
data.columns: List[str]              # List of column names
data.shape: Tuple[int, int]          # Shape as (timestamps, symbols)
data.device: torch.device            # PyTorch device
data.dtype: torch.dtype              # Data type
data.tensors: Dict[str, torch.Tensor]     # Tensors with indexing
data.raw_tensors: Dict[str, torch.Tensor] # Raw tensors
data.tensor: torch.Tensor            # Single tensor (one column only)
data.raw_tensor: torch.Tensor        # Single raw tensor (one column only)
data.arrays: Dict[str, np.ndarray]   # NumPy arrays
data.array: np.ndarray               # Single array (one column only)
```

### Data Access and Manipulation
```python
data.get(*columns: str) -> Data                    # Get specific columns
data.get(columns: Iterable[str]) -> Data           # Get columns from iterable
data.get(predicate: Callable[[str], bool]) -> Data # Get columns by predicate
data.get(pattern: str) -> Data                     # Get columns by pattern
data.set(key: str, value: Union[torch.Tensor, Data]) -> None  # Set column
data[key] -> Data                                  # Index/slice data
data.copy(deep: bool = False) -> Data              # Copy data
```

### Size and Shape
```python
data.size() -> Tuple[int, int]                     # Get shape
data.size(dim: Axis) -> int                        # Get size along axis
data.has_timestamps() -> bool                      # Check if has valid timestamps
data.has_symbols() -> bool                         # Check if has valid symbols
```

### Type and Device Conversion
```python
data.to(dtype: torch.dtype) -> Data                # Convert data type
data.to(device: torch.device) -> Data              # Move to device
data.to(tensor: torch.Tensor) -> Data              # Match tensor's dtype/device
data.to(data: Data) -> Data                        # Match other Data's dtype/device
```

### Data Conversion
```python
data.to_dataframe() -> pd.DataFrame                # Convert to DataFrame
data.to_table() -> pd.DataFrame                    # Convert to 2D table
data.to_series() -> pd.Series                      # Convert to Series (single symbol)
data.to_csv(path: Optional[str] = None) -> Optional[str]  # Export to CSV
data.to_matrix() -> pd.DataFrame                   # Convert to matrix format
data.to_matrix_csv(path: Optional[str] = None) -> Optional[str]  # Export matrix to CSV
```

### Mathematical Operations
```python
# Arithmetic
data + other, data - other, data * other, data / other
data // other, data % other, data ** other
data @ other  # Matrix multiplication

# Comparison
data == other, data != other, data > other, data < other
data >= other, data <= other

# Logical
data & other, data | other, data ^ other, ~data

# Unary
+data, -data, abs(data)
```

### Aggregation Methods
```python
data.sum(axis: Optional[Axis] = None) -> Data      # Sum
data.mean(axis: Optional[Axis] = None) -> Data     # Mean
data.var(axis: Optional[Axis] = None, ddof: int = 1) -> Data   # Variance
data.std(axis: Optional[Axis] = None, ddof: int = 1) -> Data   # Standard deviation
data.count(axis: Optional[Axis] = None) -> Data    # Count non-NaN values
data.skew(axis: Optional[Axis] = None, ddof: int = 1) -> Data  # Skewness
data.kurt(axis: Optional[Axis] = None, ddof: int = 1) -> Data  # Kurtosis
```

### Time Series Operations
```python
data.shift(shift: int = 1, skipna: bool = False) -> Data           # Shift values
data.group_shift(shift: int = 1, reference: Optional[Data] = None) -> Data  # Group-aware shift
data.pct_change(periods: int = 1, skipna: bool = False) -> Data    # Percentage change
data.diff(periods: int = 1, skipna: bool = False) -> Data          # Difference
data.cumsum(axis: Axis = 0, skipna: bool = True) -> Data           # Cumulative sum
data.cumprod(axis: Axis = 0, skipna: bool = True) -> Data          # Cumulative product
```

### Data Cleaning
```python
data.dropna(axis: Axis = 0, how: str = "any", thresh: Optional[int] = None) -> Data
data.fillna(value: float = 0.0, method: Optional[str] = None, axis: Axis = 0) -> Data
```

### Downsampling
```python
data.downsample(delta: np.timedelta64, origin: Optional[np.datetime64] = None, 
                offset: Optional[np.timedelta64] = None, 
                aggregation_f: Callable = functions.nansum) -> Data
data.minutely(origin=None, offset=None, aggregation_f=functions.nansum) -> Data
data.hourly(origin=None, offset=None, aggregation_f=functions.nansum) -> Data
data.daily(origin=None, offset=None, aggregation_f=functions.nansum) -> Data
data.weekly(origin=None, offset=None, aggregation_f=functions.nansum) -> Data
data.monthly(origin=None, offset=None, aggregation_f=functions.nansum) -> Data
data.yearly(origin=None, offset=None, aggregation_f=functions.nansum) -> Data
```

### Technical Analysis
```python
data.moving_average(window: int = 25, skipna: bool = True) -> Data
data.bollinger_band(window: int = 20, sigma: float = 2.0, skipna: bool = True) -> Tuple[Data, Data, Data]
```

### Plotting Methods
```python
data.plot(ax: Optional[matplotlib.axes.Axes] = None, **kwargs) -> List[matplotlib.axes.Axes]
data.line(ax=None, even: bool = False, **kwargs) -> matplotlib.axes.Axes
data.bar(width=0.8, bottom=0.0, ax=None, even: bool = False, **kwargs) -> matplotlib.axes.Axes
data.vlines(ymax=0.0, ax=None, even: bool = False, **kwargs) -> matplotlib.axes.Axes
data.fill_between(y2=0.0, ax=None, even: bool = False, **kwargs) -> matplotlib.axes.Axes
data.candlestick(ax=None, even: bool = False, **kwargs) -> matplotlib.axes.Axes
data.plot_moving_average(window: int = 25, skipna: bool = True, ax=None, **kwargs) -> matplotlib.axes.Axes
data.plot_bollinger_band(window: int = 20, sigma: float = 2.0, skipna: bool = True, **kwargs) -> matplotlib.axes.Axes
data.prepare_axes(ax=None, even: bool = False) -> matplotlib.axes.Axes
```

### Advanced Methods
```python
data.apply(f: Callable, *args, skipna: bool = False) -> Data
data.like(other: Data) -> Data                      # Reshape to match other
data.merge(*others: Data) -> Data                   # Merge multiple Data objects
data.merge_columns(other: Data) -> Data             # Merge columns
data.rename(columns: Union[str, Iterable[str], Dict[str, str]]) -> Data
data.subsequences(start: int, stop: int, indexes=None) -> Data
data.randseqs(batch_size: int, length: int, skipna: bool = True) -> Iterator[RandSeqsResult]
data.zeros() -> Data                                 # Create zeros with same structure
```

### Comparison and Equality
```python
data.equals(other: Data) -> bool                    # Exact equality
data.allclose(other: Data, rtol: float = 1e-05, atol: float = 1e-08) -> bool  # Approximate equality
```

### Index Conversion
```python
data.timestamp_index(v: Any, side: str = "equal") -> Union[None, int, np.ndarray]
data.symbol_index(v: Any, side: str = "equal") -> Union[None, int, np.ndarray]
```

## Flattener Class Methods

### Constructor
```python
Flattener(*data: Data)
```

### Methods
```python
flattener.flatten(data: Data) -> torch.Tensor       # Flatten to 1D tensor
flattener.unflatten(tensor: torch.Tensor, name: str = "") -> Data  # Unflatten to Data
flattener.timestamp_indexes() -> torch.Tensor       # Get timestamp indices
flattener.symbol_indexes() -> torch.Tensor          # Get symbol indices
```

## Utility Functions (qfeval_data.util)

### Array Operations
```python
util.to_numpy(tensor: torch.Tensor) -> np.ndarray   # Convert tensor to numpy
util.nans(shape=None, like=None) -> torch.Tensor    # Create NaN tensor
util.make_array_mapping(ref: np.ndarray, like: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
util.are_broadcastable_shapes(x: Tuple[int, ...], *ys: Tuple[int, ...]) -> bool
```

### Time Operations
```python
util.ceil_time(t: Union[np.datetime64, np.ndarray], d: np.timedelta64, 
               origin=None, offset=None) -> Union[np.datetime64, np.ndarray]
util.floor_time(t: Union[np.datetime64, np.ndarray], d: np.timedelta64, 
                origin=None, offset=None) -> Union[np.datetime64, np.ndarray]
util.time_origin(d: np.timedelta64) -> np.datetime64
```

### System Utilities
```python
util.sha1(x: Union[bytes, str, np.ndarray, torch.Tensor]) -> str  # Hash function
util.gc() -> None                                    # Garbage collection + GPU cleanup
util.torch_device(device: Any) -> torch.device      # Device resolution (supports "auto")
```

## Plot Module (qfeval_data.plot)

### Figure Class
```python
plot.Figure(figure=None)                            # Enhanced figure wrapper

# Properties
figure.figure: matplotlib.figure.Figure             # Matplotlib figure
figure.axes: List[matplotlib.axes.Axes]             # List of axes
figure.primary_axes: matplotlib.axes.Axes           # Primary axes

# Methods
figure.show() -> None                                # Show figure
figure.append_axes(scale: float = 0.4) -> matplotlib.axes.Axes  # Add subplot
```

### Functions
```python
plot.plot_dataframe(df: pd.DataFrame, **kwargs) -> List[matplotlib.axes.Axes]
```

## Constants and Types

### Axis Type
```python
Axis = Union[int, Literal["timestamp"], Literal["symbol"], Literal["column"]]
# Valid values: 0, 1, 2, "timestamp", "symbol", "column"
```

### RandSeqsResult
```python
@dataclass
class RandSeqsResult:
    tensor: torch.Tensor      # Shape: (batch_size, length)
    timestamps: torch.Tensor  # Shape: (batch_size,)
    symbols: torch.Tensor     # Shape: (batch_size,)
```

## Method Chaining Examples

```python
# Common method chains
result = (data
    .get('close')
    .fillna(method='ffill')
    .pct_change()
    .moving_average(20)
    .std())

# Analysis pipeline
signals = (data
    .close
    .moving_average(20)
    .gt(data.close.moving_average(50))
    .and_(data.volume > data.volume.mean()))

# Plotting chain
(data
    .close
    .moving_average(20)
    .plot_bollinger_band()
    .set_title('Price with Bollinger Bands'))
```

## Special Methods

### Magic Methods Supported
```python
__repr__()          # String representation
__getitem__()       # Indexing: data[key]
__getattr__()       # Attribute access: data.column_name
__float__()         # float(data) for single values
__add__(), __sub__(), __mul__(), __truediv__()  # Arithmetic
__eq__(), __ne__(), __gt__(), __lt__(), __ge__(), __le__()  # Comparison
__and__(), __or__(), __xor__(), __invert__()  # Logical
__neg__(), __pos__(), __abs__()  # Unary operations
```

### Serialization
```python
__getstate__()      # Pickle support
__setstate__()      # Pickle support
```