# Flattener クラスリファレンス

`Flattener` クラスは、`Data` オブジェクト（タイムスタンプ/シンボルインデックス付き）とフラットな `torch.Tensor` オブジェクト（単一のバッチインデックス付き）間の変換を支援します。

<!-- test:setup
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from qfeval_data import Data, Flattener

# Set up test data directory
_test_data_dir = Path(__file__).parent.parent / "tests" / "data" if "__file__" in dir() else Path("tests/data")
if not _test_data_dir.exists():
    _test_data_dir = Path("/Users/imos/git/qfeval-data/tests/data")
os.chdir(_test_data_dir)

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
flattener = Flattener(data)
flat_tensor = flattener.flatten(data.close)
-->

## 概要

```python
from qfeval_data import Flattener
```

Flattener は以下のような場合に便利です:
- 金融データを機械学習モデル用のバッチ形式に変換
- フラットなテンソル表現での作業
- モデル出力をタイムスタンプ/シンボルインデックス形式に戻す変換

## コンストラクタ

### `Flattener(*data)`

1つ以上の Data オブジェクトから Flattener を作成します。

**パラメータ:**
- `*data` (`Data`): フラット化マスクを定義する1つ以上の Data オブジェクト

**動作:**
- 有効な（非 NaN の）タイムスタンプ/シンボルペアのマスクを作成
- すべての入力 Data オブジェクトは同じタイムスタンプとシンボルを持つ必要あり
- すべての入力 Data において NaN 値がないペアのみが有効と見なされる

**例:**
```python
from qfeval_data import Data, Flattener

# data はセットアップで作成済み
flattener = Flattener(data)
```

**複数の Data オブジェクトの場合:**
```python
# Flattener は両方のデータセットで有効なペアのみを含む
flattener = Flattener(prices, features)
```

---

## メソッド

### `flattener.flatten(data)`

Data オブジェクトをフラットなテンソルに変換します。

**パラメータ:**
- `data` (`Data`): フラット化する Data オブジェクト（コンストラクタ入力と同じタイムスタンプ/シンボルを持つ必要あり）

**戻り値:** 形状 `(batch_size, *extra_dims)` の `torch.Tensor`

**形状変換:**
- 入力 Data 形状: `(T, S, *extra_dims)` ここで T=タイムスタンプ数、S=シンボル数
- 出力テンソル形状: `(B, *extra_dims)` ここで B=有効ペア数

**例:**
```python
# data と flattener はセットアップで作成済み
flat_tensor = flattener.flatten(data.close)
print(flat_tensor.shape)  # (B,) ここで B = 有効なタイムスタンプ/シンボルペアの数
```

**注意:**
- 有効な（非 NaN の）ペアのみが出力に含まれる
- 要素の順序は行優先（タイムスタンプが最も遅く変化）

---

### `flattener.unflatten(tensor, name="")`

フラットなテンソルを Data オブジェクトに戻します。

**パラメータ:**
- `tensor` (`torch.Tensor`): 形状 `(batch_size, *extra_dims)` のフラットテンソル
- `name` (`str`): 返される Data オブジェクトのカラム名

**戻り値:** 元のタイムスタンプ/シンボルに一致する形状の `Data`

**形状変換:**
- 入力テンソル形状: `(B, *extra_dims)`
- 出力 Data 形状: `(T, S, *extra_dims)`

**例:**
```python
# 処理後（flat_tensor を処理してシミュレート）
output_tensor = flat_tensor * 2  # 形状: (B,)

# Data に戻す
predictions = flattener.unflatten(output_tensor, name="prediction")
print(predictions.shape)  # (T, S)
```

**注意:**
- 無効なペア（フラット化時にマスクされたもの）は NaN で補完
- テンソルのバッチサイズは構築時の有効ペア数と一致する必要あり

---

### `flattener.timestamp_indexes()`

フラット化された表現の各要素のタイムスタンプインデックスを取得します。

**戻り値:** 形状 `(batch_size,)` の `torch.Tensor`

**例:**
```python
ts_idx = flattener.timestamp_indexes()
# ts_idx[i] = フラット化されたテンソルの i 番目の要素のタイムスタンプインデックス
```

---

### `flattener.symbol_indexes()`

フラット化された表現の各要素のシンボルインデックスを取得します。

**戻り値:** 形状 `(batch_size,)` の `torch.Tensor`

**例:**
```python
sym_idx = flattener.symbol_indexes()
# sym_idx[i] = フラット化されたテンソルの i 番目の要素のシンボルインデックス
```

---

## 完全な例

```python
import torch
from qfeval_data import Data, Flattener

# data はセットアップで作成済み
print(f"元の形状: {data.shape}")  # (4, 2)

# Flattener 作成
flattener = Flattener(data)

# 終値をフラット化
prices = flattener.flatten(data.close)
print(f"フラット化後の形状: {prices.shape}")  # (8,)

# 何らかの処理
log_prices = torch.log(prices)

# Data に戻す
result = flattener.unflatten(log_prices, "log_price")
print(f"結果の形状: {result.shape}")  # (4, 2)

# インデックスマッピングを取得
ts_idx = flattener.timestamp_indexes()
sym_idx = flattener.symbol_indexes()
print(f"最初の要素: timestamp={ts_idx[0].item()}, symbol={sym_idx[0].item()}")
```

---

## 機械学習ワークフロー

```python
import torch
import torch.nn as nn
from qfeval_data import Data, Flattener

# data はセットアップで作成済み
feature = data.close  # シンプルのため単一特徴量
target = data.close.pct_change()  # 日次リターン

# 両方から Flattener を作成（アライメントを保証）
flattener = Flattener(feature, target)

# 訓練用にフラット化
X = flattener.flatten(feature).unsqueeze(-1)  # 形状: (B, 1)
y = flattener.flatten(target)    # 形状: (B,)

# モデル訓練
model = nn.Linear(1, 1)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):  # 例のため短い訓練
    pred = model(X).squeeze()
    loss = ((pred - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 予測を作成してアンフラット化
with torch.no_grad():
    predictions = model(X).squeeze()
    pred_data = flattener.unflatten(predictions, "prediction")

# pred_data は元データと同じタイムスタンプ/シンボル構造を持つ
print(pred_data.shape)
```

---

## 注意事項

1. **メモリ効率**: フラット化は有効な要素をコピーして連続テンソルを作成します（ビューではない）
2. **NaN 処理**: 非 NaN の要素のみがフラット化された表現に含まれる
3. **デバイス一貫性**: Flattener は入力 Data と同じデバイスで動作
4. **バッチ次元**: フラット化されたテンソルはタイムスタンプとシンボルの次元を単一のバッチ次元に結合
