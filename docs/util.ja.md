# ユーティリティ関数リファレンス

`qfeval_data.util` モジュールは、配列操作、時間計算、その他のユーティリティのためのヘルパー関数を提供します。

<!-- test:setup
import numpy as np
import torch
from qfeval_data import util
-->

## 概要

```python
from qfeval_data import util
```

---

## 配列操作

### `util.to_numpy(tensor)`

PyTorch テンソルを NumPy 配列に変換します。

**パラメータ:**
- `tensor` (`torch.Tensor`): PyTorch テンソル（GPU 上でも勾配があっても可）

**戻り値:** `np.ndarray`

**例:**
```python
import torch
from qfeval_data import util

tensor = torch.tensor([1.0, 2.0, 3.0])
array = util.to_numpy(tensor)
print(type(array))  # <class 'numpy.ndarray'>
```

**注意:**
- 計算グラフから自動的にデタッチ
- 必要に応じて GPU から CPU に自動的に移動

---

### `util.nans(shape=None, like=None)`

NaN 値で埋められたテンソルを作成します。

**パラメータ:**
- `shape` (`Tuple[int, ...]`, 省略可): 出力テンソルの形状
- `like` (`torch.Tensor`): dtype と device の参照テンソル

**戻り値:** `torch.Tensor`

**例:**
```python
import torch
from qfeval_data import util

ref = torch.tensor([1.0, 2.0])
nans = util.nans((3, 4), like=ref)
print(nans.shape)   # torch.Size([3, 4])
print(nans.device)  # cpu
```

**注意:**
- `like` パラメータは必須
- `shape` が None の場合、`like` の形状を使用

---

### `util.make_array_mapping(ref, like)`

2つのソート済み配列間のインデックスマッピングを作成します。

**パラメータ:**
- `ref` (`np.ndarray`): 参照配列（ソート済み）
- `like` (`np.ndarray`): マッピング元の配列（ソート済み）

**戻り値:** `Tuple[np.ndarray, np.ndarray]`
- 最初の配列: `ref[indices[i]] == like[i]` となるインデックス
- 2番目の配列: マッピングが無効な場合に True となるブールマスク

**例:**
```python
import numpy as np
from qfeval_data import util

ref = np.array(["A", "B", "C", "D"])
like = np.array(["B", "D", "E"])

indexes, mask = util.make_array_mapping(ref, like)
print(indexes)  # [1, 3, 0]  (E のインデックスは 0 だがマスクされる)
print(mask)     # [False, False, True]  (E は ref にない)
```

**ユースケース:** 異なるシンボルを持つ異なるソースからのデータを揃える。

---

### `util.are_broadcastable_shapes(*shapes)`

形状が NumPy ライクな演算でブロードキャスト可能かチェックします。

**パラメータ:**
- `*shapes` (`Tuple[int, ...]` または `torch.Size`): チェックする形状

**戻り値:** `bool`

**例:**
```python
from qfeval_data import util

print(util.are_broadcastable_shapes((3, 4), (4,)))     # True
print(util.are_broadcastable_shapes((3, 4), (3, 1)))   # True
print(util.are_broadcastable_shapes((3, 4), (2, 4)))   # False
```

---

## 時間関数

### `util.floor_time(t, d, origin=None, offset=None)`

日時を時間間隔で切り捨てます。

**パラメータ:**
- `t` (`np.datetime64` または `np.ndarray`): 切り捨てるタイムスタンプ
- `d` (`np.timedelta64`): 時間間隔
- `origin` (`np.datetime64`, 省略可): 間隔計算の起点
- `offset` (`np.timedelta64`, 省略可): 切り捨て前に適用するオフセット

**戻り値:** `np.datetime64` または `np.ndarray`

**例:**
```python
import numpy as np
from qfeval_data import util

t = np.datetime64("2024-01-15T14:35:00")
d = np.timedelta64(1, "h")

floored = util.floor_time(t, d)
print(floored)  # 2024-01-15T14:00:00
```

**オフセット付き（タイムゾーン調整）:**
```python
# 9時間オフセットで日境界に切り捨て（JST タイムゾーン）
t = np.datetime64("2024-01-15T08:00:00")  # UTC
offset = np.timedelta64(9, "h")
floored = util.floor_time(t, np.timedelta64(1, "D"), offset=offset)
# 結果は JST の日境界を考慮
```

---

### `util.ceil_time(t, d, origin=None, offset=None)`

日時を時間間隔で切り上げます。

**パラメータ:**
- `t` (`np.datetime64` または `np.ndarray`): 切り上げるタイムスタンプ
- `d` (`np.timedelta64`): 時間間隔
- `origin` (`np.datetime64`, 省略可): 間隔計算の起点
- `offset` (`np.timedelta64`, 省略可): 切り上げ前に適用するオフセット

**戻り値:** `np.datetime64` または `np.ndarray`

**例:**
```python
import numpy as np
from qfeval_data import util

t = np.datetime64("2024-01-15T14:35:00")
d = np.timedelta64(1, "h")

ceiled = util.ceil_time(t, d)
print(ceiled)  # 2024-01-15T15:00:00

# すでに境界上にある場合は同じ値を返す
t2 = np.datetime64("2024-01-15T14:00:00")
print(util.ceil_time(t2, d))  # 2024-01-15T14:00:00
```

---

### `util.time_origin(d)`

指定された間隔のデフォルト時間起点を取得します。

**パラメータ:**
- `d` (`np.timedelta64`): 時間間隔

**戻り値:** `np.datetime64`

**動作:**
- 月/年間隔の場合: `1000-01-01` を返す
- その他の間隔の場合: `1893-01-01` を返す（日曜日、ダウ・ジョーンズより前）

**例:**
```python
import numpy as np
from qfeval_data import util

print(util.time_origin(np.timedelta64(1, "D")))  # 1893-01-01
print(util.time_origin(np.timedelta64(1, "M")))  # 1000-01-01
```

**注意:**
- `1893-01-01` が選ばれた理由:
  - 日曜日である（週計算に有用）
  - ダウ・ジョーンズ工業株価平均（1896年）より前
  - ナノ秒精度で十分な範囲を確保
- 週間隔は日曜日から始まる7日間を使用

---

## その他のユーティリティ

### `util.sha1(x)`

様々なデータ型の SHA1 ハッシュを計算します。

**パラメータ:**
- `x` (`bytes`, `str`, `np.ndarray`, または `torch.Tensor`): ハッシュするデータ

**戻り値:** `str`（16進数ハッシュ）

**例:**
```python
import numpy as np
import torch
from qfeval_data import util

print(util.sha1("hello"))  # aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d
print(util.sha1(np.array([1, 2, 3])))  # ハッシュには形状情報が含まれる
print(util.sha1(torch.tensor([1.0, 2.0])))  # テンソルでも動作
```

**注意:**
- 配列/テンソルの場合、ハッシュには形状情報が含まれる
- テンソルはハッシュ前に自動的に NumPy に変換

---

### `util.gc()`

ガベージコレクションを実行し、GPU メモリをクリアします。

**例:**
```python
from qfeval_data import util

# 処理後にメモリを解放
util.gc()
```

**動作:**
- Python ガベージコレクション（世代2）を実行
- GPU が利用可能な場合、CUDA キャッシュをクリア

---

### `util.torch_device(device)`

デバイス指定を `torch.device` にパースします。

**パラメータ:**
- `device` (`str`, `torch.device`, または `None`): デバイス指定

**戻り値:** `torch.device`

**特別な値:**
- `None`: CPU デバイスを返す
- `"auto"`: CUDA が利用可能なら CUDA、そうでなければ CPU を返す
- `"cpu"`, `"cuda"`, `"cuda:0"` など: 標準 PyTorch デバイス文字列

**例:**
```python
from qfeval_data import util

print(util.torch_device(None))     # cpu
print(util.torch_device("auto"))   # cuda または cpu
print(util.torch_device("cuda:0")) # cuda:0
```

---

## 型変数

モジュールはジェネリック型付けのための型変数を定義しています:

```python
import typing

# ジェネリック型
T = typing.TypeVar("T")

# 配列ライク型（torch.Tensor または np.ndarray）
Array = typing.TypeVar("Array", torch.Tensor, np.ndarray)
```
