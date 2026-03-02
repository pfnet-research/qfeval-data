# Data クラスリファレンス

`Data` クラスは qfeval-data の中核コンポーネントです。タイムスタンプとシンボルでインデックス付けされた数値テンソルを管理し、効率的な金融時系列データの操作のために設計されています。

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
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        dtype="datetime64[D]",
    )
    symbols = np.array(["AAPL", "GOOG"])
    tensors = {
        "open": torch.tensor([[100.0, 200.0], [101.0, 201.0], [102.0, 202.0], [103.0, 203.0], [104.0, 204.0]]),
        "high": torch.tensor([[105.0, 205.0], [106.0, 206.0], [107.0, 207.0], [108.0, 208.0], [109.0, 209.0]]),
        "low": torch.tensor([[98.0, 198.0], [99.0, 199.0], [100.0, 200.0], [101.0, 201.0], [102.0, 202.0]]),
        "close": torch.tensor([[104.0, 204.0], [105.0, 205.0], [106.0, 206.0], [107.0, 207.0], [108.0, 208.0]]),
        "volume": torch.tensor([[1e6, 5e5], [1.1e6, 5.5e5], [1.2e6, 6e5], [1.3e6, 6.5e5], [1.4e6, 7e5]]),
    }
    return Data.from_tensors(tensors, timestamps, symbols)

data = create_sample_data()
tick_data = data  # alias for examples
-->

## 概要

```python
from qfeval_data import Data
```

### データ構造

- **テンソル**: カラム名（文字列）から PyTorch テンソルへの辞書
- **形状**: 各テンソルは `(num_timestamps, num_symbols, *extra_dimensions)` の形状を持つ
- **タイムスタンプ**: `np.ndarray[datetime64]` - 常にソート済み
- **シンボル**: `np.ndarray[str]` - 常にソート済み

### 設計原則

1. **遅延スライシング**: スライス操作はデータをコピーせずビューを作成
2. **ソート済みインデックス**: タイムスタンプとシンボルは構築時に自動的にソート
3. **メソッドチェーン**: ほとんどのメソッドは `Data` オブジェクトを返し、流暢な API を実現
4. **GPU サポート**: PyTorch テンソルバックエンドによる完全なデバイス柔軟性

---

## 構築メソッド

### `Data.from_dataframe(df, dtype=None, device=None)`

pandas DataFrame から `Data` オブジェクトを作成します。

**パラメータ:**
- `df` (`pd.DataFrame`): `timestamp` と `symbol` カラムが必須の DataFrame
- `dtype` (`torch.dtype`, 省略可): テンソルのデータ型
- `device` (`str` または `torch.device`, 省略可): テンソルのデバイス

**戻り値:** `Data`

**例:**
```python
import pandas as pd
from qfeval_data import Data

df = pd.DataFrame({
    "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "symbol": ["AAPL", "GOOG", "AAPL", "GOOG"],
    "open": [150.0, 140.0, 152.0, 142.0],
    "close": [155.0, 145.0, 153.0, 143.0],
})
data = Data.from_dataframe(df)
```

**多次元カラム:**

多次元データにはカラム名にブラケット記法を使用:
```python
df = pd.DataFrame({
    "timestamp": ["2024-01-01"],
    "symbol": ["AAPL"],
    "embedding[0]": [0.1],
    "embedding[1]": [0.2],
    "embedding[2]": [0.3],
})
data = Data.from_dataframe(df)
# data.embedding の形状は (1, 1, 3)
```

---

### `Data.from_csv(input, dtype=None, device=None)`

CSV ファイルから `Data` オブジェクトを読み込みます。

**パラメータ:**
- `input` (`str` またはファイルライクオブジェクト): CSV ファイルへのパスまたはファイルオブジェクト
- `dtype` (`torch.dtype`, 省略可): テンソルのデータ型
- `device` (`str` または `torch.device`, 省略可): テンソルのデバイス

**戻り値:** `Data`

**例:**
```python
data = Data.from_csv("prices.csv")
# data = Data.from_csv("prices.csv.xz")  # 圧縮ファイルもサポート
```

**CSV フォーマット:**
<!-- test:skip -->
```csv
timestamp,symbol,open,high,low,close,volume
2024-01-01,AAPL,150.0,156.0,149.0,155.0,1000000
2024-01-01,GOOG,140.0,146.0,139.0,145.0,800000
```

---

### `Data.from_tensors(tensors, timestamps, symbols)`

テンソルから直接 `Data` オブジェクトを作成します。最もプリミティブなコンストラクタです。

**パラメータ:**
- `tensors` (`Dict[str, torch.Tensor]`): カラム名からテンソルへの辞書
- `timestamps` (`np.ndarray`): datetime64 値の1次元配列
- `symbols` (`np.ndarray`): シンボル文字列の1次元配列

**戻り値:** `Data`

**例:**
```python
import torch
import numpy as np
from qfeval_data import Data

tensors = {
    "open": torch.tensor([[150.0, 140.0], [152.0, 142.0]]),
    "close": torch.tensor([[155.0, 145.0], [153.0, 143.0]]),
}
timestamps = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
symbols = np.array(["AAPL", "GOOG"])

data = Data.from_tensors(tensors, timestamps, symbols)
```

**注意:**
- タイムスタンプとシンボルは自動的にソートされ、テンソルもそれに応じて再インデックスされます
- すべてのテンソルは `(len(timestamps), len(symbols), ...)` の形状を持つ必要があります
- すべてのテンソルは同じデバイス上にある必要があります

---

### `Data.from_preset(name="pfn-topix500", dtype=None, device=None, paths=[])`

システムパスからプリセットデータファイルを読み込みます。

**パラメータ:**
- `name` (`str`): プリセット名（`data/{name}.csv` または `data/{name}.csv.xz` を検索）
- `dtype` (`torch.dtype`, 省略可): テンソルのデータ型
- `device` (`str` または `torch.device`, 省略可): テンソルのデバイス
- `paths` (`List[str]`): 追加の検索パス

**戻り値:** `Data`

**例外:** プリセットが見つからない場合 `FileNotFoundError`

---

## プロパティ

### データアクセスプロパティ

| プロパティ | 型 | 説明 |
|----------|------|-------------|
| `tensors` | `Dict[str, Tensor]` | スライシング適用後のテンソル |
| `tensor` | `Tensor` | 単一テンソル（カラムが1つの場合のみ） |
| `raw_tensors` | `Dict[str, Tensor]` | スライシングなしの直接テンソルアクセス |
| `raw_tensor` | `Tensor` | 単一の生テンソル |
| `arrays` | `Dict[str, np.ndarray]` | テンソルの NumPy 配列版 |
| `array` | `np.ndarray` | 単一配列版 |

### メタデータプロパティ

| プロパティ | 型 | 説明 |
|----------|------|-------------|
| `timestamps` | `np.ndarray` | ソート済み datetime64 配列 |
| `symbols` | `np.ndarray` | ソート済み文字列配列 |
| `columns` | `List[str]` | カラム名のリスト |
| `shape` | `Tuple[int, int]` | `(num_timestamps, num_symbols)` |
| `device` | `torch.device` | テンソルのデバイス |
| `dtype` | `torch.dtype` | テンソルのデータ型 |

---

## インデックスとスライシング

### `data[timestamp_idx, symbol_idx]`

タイムスタンプとシンボルインデックスでデータにアクセス。複数のインデックススタイルをサポート:

**整数インデックス:**
```python
data[0, :]          # 最初のタイムスタンプ、全シンボル
data[:, 0]          # 全タイムスタンプ、最初のシンボル
data[0, 0]          # 単一要素
data[-1, :]         # 最後のタイムスタンプ
```

**スライスインデックス:**
```python
data[:10, :]        # 最初の10タイムスタンプ
data[5:15, :]       # タイムスタンプ 5-14
data[:, :3]         # 最初の3シンボル
```

**値ベースインデックス:**
```python
data["2024-01-01", :]           # タイムスタンプ値で指定
data["2024-01-01":"2024-01-31", :]  # タイムスタンプ範囲
data[:, "AAPL"]                 # シンボル値で指定
data[:, ["AAPL", "GOOG"]]       # 複数シンボル
```

**ブールマスクインデックス:**
```python
mask = data.close > data.open   # ブール Data
filtered = data[mask]           # マスク適用（マッチしない箇所は NaN に）
```

---

## カラムアクセス

### `data.get(*columns)` / `data.get(columns)` / `data.get(pattern=...)`

カラムのサブセットを抽出します。

**シグネチャ:**
```python
def get(self, *columns: str) -> Data: ...
def get(self, columns: Iterable[str]) -> Data: ...
def get(self, filter_func: Callable[[str], bool]) -> Data: ...
def get(self, *, pattern: str) -> Data: ...
```

**例:**
```python
# 単一カラム
opens = data.get("open")

# 複数カラム
ohlc = data.get("open", "high", "low", "close")
ohlc = data.get(["open", "high", "low", "close"])

# フィルタ関数
prices = data.get(lambda c: c in ["open", "close"])

# Glob パターン
prices = data.get(pattern="*price*")
```

### 属性アクセス

カラムは属性としてアクセス可能:
```python
data.close      # data.get("close") と同等
data.volume     # data.get("volume") と同等
```

---

### `data.set(key, value)`

カラムを追加または更新します。

**パラメータ:**
- `key` (`str`): カラム名
- `value` (`torch.Tensor` または `Data`): カラム値

**例:**
```python
data.set("returns", data.close.pct_change().tensor)
data.set("spread", data.high - data.low)
```

---

### `data.rename(columns)`

カラム名を変更します。

**パラメータ:**
- `columns` (`str`, `List[str]`, または `Dict[str, str]`): 新しいカラム名

**戻り値:** `Data`

**例:**
```python
# 単一カラムの名前変更（Data が1カラムの場合）
renamed = data.get("close").rename("price")

# リストで名前変更（カラム数と一致する必要あり）
renamed = data.rename(["o", "h", "l", "c", "v"])

# 辞書で選択的に名前変更
renamed = data.rename({"open": "o", "close": "c"})
```

---

## 算術演算

すべての算術演算はテンソルに対して要素単位で行われます:

### 二項演算子

| 演算子 | 説明 |
|----------|-------------|
| `+`, `-`, `*`, `/` | 基本算術 |
| `//` | 切り捨て除算 |
| `%` | 剰余 |
| `**` | べき乗 |
| `@` | 行列乗算 |
| `&`, `\|`, `^` | ビット演算 |

### 比較演算子

| 演算子 | 説明 |
|----------|-------------|
| `==`, `!=` | 等価（ブール Data を返す） |
| `<`, `>`, `<=`, `>=` | 比較（ブール Data を返す） |

**注意:** Python の真偽値評価を避けるため `.eq()` と `.ne()` メソッドを使用してください。

### 単項演算子

| 演算子 | 説明 |
|----------|-------------|
| `-x` | 符号反転 |
| `+x` | 正 |
| `abs(x)` | 絶対値 |
| `~x` | ビット否定 |

**例:**
```python
returns = (data.close / data.open) - 1
spread = data.high - data.low
is_up = data.close > data.open
```

---

## 時系列演算

### `data.shift(shift=1, skipna=False)`

タイムスタンプ軸に沿って値をシフトします。

**パラメータ:**
- `shift` (`int`): シフトする期間数（正=前方、負=後方）
- `skipna` (`bool`): True の場合、シフト時に NaN 値をスキップ

**戻り値:** `Data`

**例:**
```python
previous = data.shift(1)      # 前日の値
next_day = data.shift(-1)     # 翌日の値
```

---

### `data.pct_change(periods=1, skipna=False)`

変化率を計算します。

**計算式:** `(current / previous) - 1`

**パラメータ:**
- `periods` (`int`): 比較のためのシフト期間
- `skipna` (`bool`): NaN 値をスキップ

**戻り値:** `Data`

**例:**
```python
daily_returns = data.close.pct_change()
weekly_returns = data.close.pct_change(periods=5)
```

---

### `data.diff(periods=1, skipna=False)`

現在値と前回値の差を計算します。

**計算式:** `current - previous`

**パラメータ:**
- `periods` (`int`): 比較のためのシフト期間
- `skipna` (`bool`): NaN 値をスキップ

**戻り値:** `Data`

---

### `data.cumsum(axis=0, skipna=True)`

軸に沿った累積和。

**パラメータ:**
- `axis` (`int` または `str`): 軸（0/"timestamp" または 1/"symbol"）
- `skipna` (`bool`): NaN 値をスキップ

**戻り値:** `Data`

---

### `data.cumprod(axis=0, skipna=True)`

軸に沿った累積積。

**パラメータ:**
- `axis` (`int` または `str`): 軸（0/"timestamp" または 1/"symbol"）
- `skipna` (`bool`): NaN 値をスキップ

**戻り値:** `Data`

---

## 集約メソッド

すべての集約メソッドは `axis` パラメータをサポート:
- `axis=0` または `axis="timestamp"`: タイムスタンプ方向に集約
- `axis=1` または `axis="symbol"`: シンボル方向に集約
- `axis=None`: 両軸で集約

### 統計集約

| メソッド | 説明 |
|--------|-------------|
| `sum(axis=None)` | 合計 |
| `mean(axis=None)` | 算術平均 |
| `min(axis=None)` | 最小値 |
| `max(axis=None)` | 最大値 |
| `var(axis=None, ddof=1)` | 分散 |
| `std(axis=None, ddof=1)` | 標準偏差 |
| `skew(axis=None, ddof=1)` | 歪度 |
| `kurt(axis=None, ddof=1)` | 尖度 |
| `count(axis=None)` | 非 NaN 値のカウント |

### 位置集約

| メソッド | 説明 |
|--------|-------------|
| `first(axis="timestamp", skipna=True)` | 最初の値 |
| `last(axis="timestamp", skipna=True)` | 最後の値 |

**例:**
```python
# 全タイムスタンプの平均価格
avg_price = data.close.mean(axis=0)

# シンボルごとの合計出来高
total_vol = data.volume.sum(axis=0)

# 全体統計
stats = data.close.mean()  # スカラー（単一値）
```

---

## 欠損値処理

### `data.dropna(axis=0, how="any", thresh=None)`

欠損値を含む行またはカラムを削除します。

**パラメータ:**
- `axis` (`int` または `str`): 削除する軸（0=タイムスタンプ、1=シンボル）
- `how` (`str`): "any"（いずれかが NaN なら削除）または "all"（すべてが NaN なら削除）
- `thresh` (`int`, 省略可): 必要な非 NaN 値の最小数

**戻り値:** `Data`

**例:**
```python
# 欠損値があるタイムスタンプを削除
clean = data.dropna(axis=0, how="any")

# すべてが欠損のシンボルを削除
clean = data.dropna(axis=1, how="all")
```

---

### `data.fillna(value=0.0, method=None, axis=0)`

欠損値を補完します。

**パラメータ:**
- `value` (`float`): NaN を置き換える値（`method=None` の場合）
- `method` (`str`, 省略可): 補完方法 - `"ffill"`（前方補完）または `"bfill"`（後方補完）
- `axis` (`int` または `str`): 補完メソッドの軸

**戻り値:** `Data`

**例:**
```python
# ゼロで補完
filled = data.fillna(0.0)

# 前方補完（前の値を使用）
filled = data.fillna(method="ffill")

# 後方補完（次の値を使用）
filled = data.fillna(method="bfill")
```

---

## 金融メトリクス

### `data.annualized_return()`

年率リターンを計算します。

**計算式:** `(last / first) ^ (1 / years) - 1`

**戻り値:** タイムスタンプ次元が折りたたまれた `Data`

**エイリアス:** `ar()`

---

### `data.annualized_volatility()`

年率ボラティリティ（年率換算されたリターンの標準偏差）を計算します。

**戻り値:** タイムスタンプ次元が折りたたまれた `Data`

**エイリアス:** `avol()`

---

### `data.annualized_sharpe_ratio()`

年率シャープレシオを計算します。

**計算式:** `annualized_return / annualized_volatility`

**戻り値:** タイムスタンプ次元が折りたたまれた `Data`

**エイリアス:** `asr()`

---

### `data.maximum_drawdown()`

最大ドローダウン（最大のピークからトラフへの下落）を計算します。

**戻り値:** タイムスタンプ次元が折りたたまれた `Data`

**エイリアス:** `mdd()`

---

### `data.metrics()`

すべてのメトリクスを一度に計算します。

**戻り値:** 以下のカラムを持つ `Data`:
- `annualized_sharpe_ratio`
- `annualized_return`
- `annualized_volatility`
- `maximum_drawdown`

**例:**
```python
metrics = data.close.metrics()
print(metrics.to_dataframe())
```

---

## リサンプリングメソッド

データをより低い頻度にダウンサンプリングします。OHLC カラムは特別に処理されます:
- `open`: ウィンドウ内の最初の有効値
- `high`: ウィンドウ内の最大値
- `low`: ウィンドウ内の最小値
- `close`: ウィンドウ内の最後の有効値
- その他のカラム: デフォルトで合計

### メソッド

| メソッド | 頻度 |
|--------|-----------|
| `minutely()` | 1分 |
| `hourly()` | 1時間 |
| `daily()` | 1日 |
| `weekly()` | 7日 |
| `monthly()` | 1ヶ月 |
| `yearly()` | 1年 |

**パラメータ（すべてのメソッド共通）:**
- `origin` (`np.datetime64`, 省略可): バケット化の起点時刻
- `offset` (`np.timedelta64`, 省略可): タイムゾーンオフセット調整
- `aggregation_f` (callable): 非 OHLC カラムの集約関数

**例:**
```python
# ティックデータを日次 OHLCV に変換
daily = tick_data.daily()

# タイムゾーンオフセット付き週次データ
weekly = data.weekly(offset=np.timedelta64(9, "h"))
```

---

### `data.downsample(delta, origin=None, offset=None, aggregation_f=nansum)`

任意の頻度への汎用ダウンサンプリング。

**パラメータ:**
- `delta` (`np.timedelta64`): バケット化の時間間隔
- `origin` (`np.datetime64`, 省略可): 起点時刻
- `offset` (`np.timedelta64`, 省略可): タイムゾーンオフセット
- `aggregation_f` (callable): 集約関数

**例:**
```python
# 15分足
bars_15m = data.downsample(np.timedelta64(15, "m"))
```

---

## テクニカル指標

### `data.moving_average(window=25, skipna=True)`

単純移動平均を計算します。

**パラメータ:**
- `window` (`int`): ウィンドウサイズ
- `skipna` (`bool`): NaN 値をスキップ

**戻り値:** `Data`

---

### `data.bollinger_band(window=20, sigma=2.0, skipna=True)`

ボリンジャーバンドを計算します。

**パラメータ:**
- `window` (`int`): 移動平均のウィンドウサイズ
- `sigma` (`float`): バンドの標準偏差数

**戻り値:** `Tuple[Data, Data, Data]` - (上限、中央、下限)

**例:**
```python
upper, middle, lower = data.close.bollinger_band(window=20, sigma=2.0)
```

---

## 可視化メソッド

すべての可視化メソッドには matplotlib が必要です（`pip install qfeval-data[plot]`）。

### `data.plot(ax=None, **kwargs)`

カラムに基づいてプロットタイプを自動検出。OHLC データにはローソク足、それ以外は折れ線グラフを使用。

**パラメータ:**
- `ax` (`matplotlib.axes.Axes`, 省略可): プロット先の Axes

**戻り値:** `List[matplotlib.axes.Axes]`

---

### `data.line(ax=None, even=False, **kwargs)`

折れ線グラフ。

**パラメータ:**
- `ax` (`matplotlib.axes.Axes`, 省略可): プロット先の Axes
- `even` (`bool`): 等間隔の x 軸を使用（時間ギャップを無視）
- `**kwargs`: `matplotlib.plot()` に渡される

---

### `data.bar(width=0.8, bottom=0.0, ax=None, **kwargs)`

棒グラフ。

**パラメータ:**
- `width` (`float` または `Data`): 棒の幅
- `bottom` (`float` または `Data`): 棒の底の位置
- `ax` (`matplotlib.axes.Axes`, 省略可): プロット先の Axes

---

### `data.candlestick(ax=None, **kwargs)`

OHLC ローソク足チャート。`open`, `high`, `low`, `close` カラムが必要です。

**パラメータ:**
- `ax` (`matplotlib.axes.Axes`, 省略可): プロット先の Axes
- `upcolor` (`str`): 陽線の色（デフォルト: "#ee3333"）
- `downcolor` (`str`): 陰線の色（デフォルト: "#118822"）
- `neutralcolor` (`str`): 中立の色（デフォルト: "#444444"）
- `width` (`float`): ローソク実体の幅（デフォルト: 0.6）
- `linewidth` (`float`): 髭の線幅（デフォルト: 0.5）

**例:**
```python
import matplotlib.pyplot as plt
from qfeval_data import Data

data = Data.from_csv("prices.csv")
data[:, "AAPL"].candlestick()  # 単一シンボルをプロット
plt.close()
```

---

## 変換メソッド

### `data.to_dataframe()`

ロング形式の pandas DataFrame に変換します。

**戻り値:** `timestamp`, `symbol`, および全データカラムを持つ `pd.DataFrame`

**例:**
```python
df = data.to_dataframe()
#    timestamp symbol   open  close
# 0 2024-01-01   AAPL  150.0  155.0
# 1 2024-01-01   GOOG  140.0  145.0
```

---

### `data.to_table()`

ワイド形式（2次元テーブル）の pandas DataFrame に変換します。

**戻り値:** `pd.DataFrame`

**注意:**
- 単一カラム: タイムスタンプがインデックス、シンボルがカラム
- 複数カラム: 単一タイムスタンプまたは単一シンボルが必要

---

### `data.to_series()`

pandas Series に変換します。単一カラムかつ単一シンボルが必要です。

**戻り値:** `pd.Series`

---

### `data.to_csv(path=None)`

CSV 形式でエクスポートします。

**パラメータ:**
- `path` (`str`, 省略可): ファイルパス。None の場合、CSV 文字列を返します。

**戻り値:** `str`（path が None の場合）または `None`

---

## ユーティリティメソッド

### `data.copy(deep=False)`

コピーを作成します。

**パラメータ:**
- `deep` (`bool`): True の場合、テンソルをコピー；そうでなければテンソル参照を共有

**戻り値:** `Data`

---

### `data.to(dtype_or_device)`

dtype および/または device を変換します。

**シグネチャ:**
```python
def to(self, dtype: torch.dtype) -> Data: ...
def to(self, device: torch.device) -> Data: ...
def to(self, tensor: torch.Tensor) -> Data: ...
def to(self, data: Data) -> Data: ...
```

**例:**
```python
data_f64 = data.to(torch.float64)
other_data = data.get("close")
data_like = data.to(other_data)  # dtype/device を合わせる
```

---

### `data.like(other)`

別の Data のタイムスタンプとシンボルに合わせてリシェイプします。

**パラメータ:**
- `other` (`Data`): 形状の参照となる Data

**戻り値:** `Data`

**注意:**
- 欠落したタイムスタンプ/シンボルの組み合わせは NaN で補完
- 余分な組み合わせは破棄

---

### `data.merge(*others)`

複数の Data オブジェクトをマージ（タイムスタンプ/シンボルの和集合）。

**パラメータ:**
- `*others` (`Data`): マージする Data オブジェクト

**戻り値:** `Data`

**注意:**
- 重複するセルでは、最後の非 NaN 値が優先
- 同名のカラムは互換性のある形状を持つ必要あり

---

### `data.apply(f, *args, skipna=False)`

テンソルに関数を適用します。

**パラメータ:**
- `f` (callable): テンソルを受け取りテンソルを返す関数
- `*args`: 追加の引数（Data または値）
- `skipna` (`bool`): NaN 値をスキップ

**戻り値:** `Data`

**例:**
```python
# カスタム関数を適用
result = data.close.apply(lambda x: torch.log(x + 1))

# 追加引数付き
other_data = data.close
result = data.close.apply(lambda x, y: x * y, other_data)
```

---

## 比較メソッド

### `data.equals(other)`

完全な等価性をチェック（NaN の位置を含む）。

**戻り値:** `bool`

---

### `data.allclose(other, rtol=1e-5, atol=1e-8)`

近似等価性をチェック。

**パラメータ:**
- `rtol` (`float`): 相対許容誤差
- `atol` (`float`): 絶対許容誤差

**戻り値:** `bool`

---

## シリアライゼーション

`Data` クラスは Python の pickle プロトコルをサポート:

```python
import pickle
import tempfile
import os

# tempfile を使用して保存と読み込み
with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
    pickle.dump(data, f)
    temp_path = f.name

with open(temp_path, "rb") as f:
    loaded_data = pickle.load(f)

os.unlink(temp_path)  # クリーンアップ
```
