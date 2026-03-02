# 使用例とレシピ

このドキュメントでは qfeval-data の実用的な例と一般的な使用パターンを紹介します。

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

## 目次

1. [データの読み込み](#データの読み込み)
2. [基本操作](#基本操作)
3. [時系列分析](#時系列分析)
4. [ポートフォリオ分析](#ポートフォリオ分析)
5. [データ変換](#データ変換)
6. [可視化](#可視化)
7. [機械学習との統合](#機械学習との統合)
8. [複数シンボルの操作](#複数シンボルの操作)

---

## データの読み込み

### CSV ファイルから

```python
from qfeval_data import Data

# 基本的な読み込み
data = Data.from_csv("prices.csv")

# dtype を指定
data = Data.from_csv("prices.csv", dtype=torch.float32)
```

**期待される CSV フォーマット:**
<!-- test:skip -->
```csv
timestamp,symbol,open,high,low,close,volume
2024-01-02,AAPL,185.5,186.2,184.1,185.8,50000000
2024-01-02,GOOG,140.0,141.5,139.5,141.0,20000000
2024-01-03,AAPL,186.0,187.5,185.0,186.5,48000000
2024-01-03,GOOG,141.0,142.0,140.0,141.5,19000000
```

### pandas DataFrame から

```python
import pandas as pd
from qfeval_data import Data

# サンプルデータを作成
df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=10, freq="D").repeat(2),
    "symbol": ["AAPL", "GOOG"] * 10,
    "close": [150 + i * 0.5 + (0 if i % 2 == 0 else 10) for i in range(20)],
})

data = Data.from_dataframe(df)
print(data.shape)  # (10, 2)
```

### 生テンソルから

```python
import torch
import numpy as np
from qfeval_data import Data

# テンソルを作成
timestamps = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[D]")
symbols = np.array(["AAPL", "GOOG", "MSFT"])
prices = torch.randn(3, 3) * 10 + 100  # 3 タイムスタンプ x 3 シンボル

data = Data.from_tensors({"close": prices}, timestamps, symbols)
```

### 多次元データ

```python
# タイムスタンプ/シンボルごとの埋め込みベクトル
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

## 基本操作

### データへのアクセス

```python
from qfeval_data import Data

# data はセットアップで作成済み

# 特定のカラムを取得
closes = data.close                    # 属性アクセス
closes = data.get("close")             # メソッドアクセス
ohlc = data.get("open", "high", "low", "close")

# 時間でスライス
first_week = data[:5, :]               # 最初の5タイムスタンプ
jan_data = data["2024-01-01":"2024-01-12", :]

# シンボルでスライス
apple = data[:, "AAPL"]                # 単一シンボル
tech = data[:, ["AAPL", "GOOG", "MSFT"]]  # 複数シンボル

# 組み合わせスライス
apple_jan = data["2024-01-01":"2024-01-12", "AAPL"]
```

### 算術演算

```python
# リターン
returns = data.close.pct_change()

# 対数リターン
log_returns = (data.close / data.close.shift(1)).apply(torch.log)

# スプレッド
spread = data.high - data.low

# カスタム計算
typical_price = (data.high + data.low + data.close) / 3
```

### フィルタリング

```python
# ブールフィルタリング
up_days = data[data.close > data.open]  # マッチしない箇所は NaN に

# 欠損値を削除
clean = data.dropna()

# 欠損値を補完
filled = data.fillna(method="ffill")
```

---

## 時系列分析

### ローリング計算

```python
# 移動平均（ウィンドウサイズ <= データ長）
ma_5 = data.close.moving_average(window=5)

# ボリンジャーバンド
upper, middle, lower = data.close.bollinger_band(window=5, sigma=2.0)
```

### ラグ特徴量

```python
# 過去の値
prev_close = data.close.shift(1)
prev_5_close = data.close.shift(5)

# 将来の値（ターゲット用）
next_close = data.close.shift(-1)
next_return = data.close.shift(-1).pct_change()
```

### リサンプリング

```python
# ティックデータから日次データ
daily = tick_data.daily()

# 週次 OHLCV
weekly = daily.weekly()

# タイムゾーンオフセット付き月次
monthly = daily.monthly(offset=np.timedelta64(9, "h"))

# カスタム間隔
bars_15m = data.downsample(np.timedelta64(15, "m"))
```

---

## ポートフォリオ分析

### 単一銘柄のメトリクス

```python
# 単一銘柄のメトリクスを取得
apple = data[:, "AAPL"]
metrics = apple.close.metrics()
print(metrics.to_dataframe())
#                             annualized_sharpe_ratio  annualized_return  annualized_volatility  maximum_drawdown
# symbol
# AAPL                                          1.25              0.15                   0.12              0.08
```

### クロスセクション分析

```python
# 全銘柄のメトリクスを比較
all_metrics = data.close.metrics()

# 最高シャープレシオを見つける
sharpe = all_metrics.get("annualized_sharpe_ratio")
best_idx = sharpe.tensor.argmax()
best_symbol = data.symbols[best_idx]
print(f"最高シャープレシオ: {best_symbol}")
```

### ポートフォリオリターン

```python
import torch

# 等ウェイトポートフォリオ
weights = torch.ones(data.shape[1]) / data.shape[1]
portfolio_returns = (data.close.pct_change() * weights).sum(axis=1)

# カスタムウェイト
weights = torch.tensor([0.4, 0.3, 0.3])  # AAPL, GOOG, MSFT
portfolio_returns = (data.close.pct_change() * weights).sum(axis=1)

# ポートフォリオ累積リターン
cumulative = (1 + portfolio_returns).cumprod()
```

### 相関分析

```python
# リターンを計算
returns = data.close.pct_change()

# 相関のために numpy に変換
returns_array = returns.dropna().array
import numpy as np
corr_matrix = np.corrcoef(returns_array.T)
print(pd.DataFrame(corr_matrix, index=data.symbols, columns=data.symbols))
```

---

## データ変換

### 正規化

```python
# 時間方向の Z スコア正規化
mean = data.close.mean(axis=0)
std = data.close.std(axis=0)
normalized = (data.close - mean) / std

# Min-Max 正規化
min_val = data.close.min(axis=0)
max_val = data.close.max(axis=0)
scaled = (data.close - min_val) / (max_val - min_val)
```

### 特徴量作成

```python
def create_features(data):
    """一般的なテクニカル特徴量を作成"""
    features = []

    # リターン
    features.append(data.close.pct_change().rename("return_1d"))
    features.append(data.close.pct_change(3).rename("return_3d"))

    # 移動平均（ウィンドウサイズ <= データ長）
    ma_3 = data.close.moving_average(3)
    ma_5 = data.close.moving_average(5)
    features.append((data.close / ma_3 - 1).rename("close_ma3_ratio"))
    features.append((data.close / ma_5 - 1).rename("close_ma5_ratio"))
    features.append((ma_3 / ma_5 - 1).rename("ma3_ma5_ratio"))

    # 出来高比率
    if "volume" in data.columns:
        vol_ma = data.volume.moving_average(5)
        features.append((data.volume / vol_ma).rename("volume_ratio"))

    # すべての特徴量をマージ
    result = features[0]
    for f in features[1:]:
        result = result.merge_columns(f)
    return result

features = create_features(data)
```

### データソースのマージ

```python
# 複数のデータソースをマージ
prices = Data.from_csv("prices.csv")
fundamentals = Data.from_csv("fundamentals.csv")

# 同じタイムスタンプ/シンボル - カラムをマージ
combined = prices.merge_columns(fundamentals)

# 異なるタイムスタンプ/シンボル - 和集合マージ
combined = prices.merge(fundamentals)
```

---

## 可視化

### 基本プロット

```python
import matplotlib.pyplot as plt
from qfeval_data import Data

prices = Data.from_csv("prices.csv")
aapl = prices[:, "AAPL"]

# OHLC データにはローソク足
aapl.candlestick()
plt.title("AAPL")
plt.close()
```

### ローソク足チャート

```python
# 明示的なローソク足（aapl は前の例で作成済み）
aapl.candlestick()
plt.title("AAPL ローソク足")
plt.close()

# カスタムカラー
aapl.candlestick(
    upcolor="#00ff00",
    downcolor="#ff0000",
    width=0.8
)
plt.close()
```

### 折れ線グラフ

```python
# 単一系列
aapl.close.line()
plt.title("AAPL 終値")
plt.close()

# 複数系列
fig, ax = plt.subplots()
aapl.close.line(ax=ax, label="Close")
aapl.close.moving_average(5).line(ax=ax, label="MA5")
plt.legend()
plt.close()
```

### テクニカル指標

```python
# 移動平均オーバーレイ
fig, ax = plt.subplots()
aapl.candlestick(ax=ax)
aapl.close.plot_moving_average(window=5, ax=ax, color="blue")
plt.close()

# ボリンジャーバンド
fig, ax = plt.subplots()
aapl.candlestick(ax=ax)
aapl.close.plot_bollinger_band(window=5, ax=ax)
plt.close()
```

### 複数サブプロット

```python
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# ボリンジャーバンド付き価格
aapl.candlestick(ax=axes[0])
aapl.close.plot_bollinger_band(window=5, ax=axes[0])
axes[0].set_title("価格")

# 出来高
aapl.volume.bar(ax=axes[1])
axes[1].set_title("出来高")

# リターン
aapl.close.pct_change().line(ax=axes[2])
axes[2].set_title("日次リターン")

plt.tight_layout()
plt.close()
```

---

## 機械学習との統合

### PyTorch 用データ準備

```python
import torch
from qfeval_data import Data, Flattener

# data はセットアップで作成済み

# 特徴量とターゲットを作成（Flattener 用に単一カラム）
feature = data.close
target = data.close.pct_change()  # 日次リターン

# アライメント用の Flattener を作成
flattener = Flattener(feature, target)

# テンソルに変換
X = flattener.flatten(feature).unsqueeze(-1)  # 形状: (B, 1)
y = flattener.flatten(target)     # 形状: (B,)

print(f"特徴量形状: {X.shape}")
print(f"ターゲット形状: {y.shape}")
```

### 訓練ループ

```python
import torch.nn as nn
import torch.optim as optim

# シンプルなモデル
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

# 訓練（例のため短いループ）
for epoch in range(10):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
```

### 予測の作成

```python
# 予測を作成
model.eval()
with torch.no_grad():
    predictions = model(X)

# Data 形式に戻す
pred_data = flattener.unflatten(predictions, "prediction")

# 予測形状を確認
print(f"予測形状: {pred_data.shape}")
```

### 時系列分割

```python
# 時間で分割（サンプルデータの日付を使用）
split_date = "2024-01-08"
train_data = data[:split_date, :]
test_data = data[split_date:, :]

print(f"訓練: {train_data.shape}, テスト: {test_data.shape}")
```

---

## 複数シンボルの操作

### クロスセクション操作

```python
# 各タイムスタンプ内でシンボル間のランク付け
def rank_cross_section(data):
    """各タイムスタンプでシンボル間の値をランク付け"""
    return data.apply(
        lambda x: x.argsort(dim=1).argsort(dim=1).float() / (x.shape[1] - 1)
    )

ranked = rank_cross_section(data.close.pct_change())
```

### セクター分析

```python
# セクターマッピングがあると仮定
sector_map = {"AAPL": "Tech", "GOOG": "Tech", "JPM": "Finance", "XOM": "Energy"}
sectors = [sector_map.get(s, "Other") for s in data.symbols]

# セクターでグループ化
tech_symbols = [s for s, sec in zip(data.symbols, sectors) if sec == "Tech"]
tech_data = data[:, tech_symbols]

# セクター平均
tech_avg = tech_data.close.mean(axis=1).rename("tech_avg")
```

### ユニバースフィルタリング

```python
# 流動性でフィルタ（サンプルデータ用に閾値を調整）
avg_volume = data.volume.mean(axis=0)
liquid_mask = avg_volume.tensor > 900000
liquid_symbols = data.symbols[liquid_mask.cpu().numpy()]
liquid_data = data[:, liquid_symbols.tolist()]

# 価格でフィルタ
avg_price = data.close.mean(axis=0)
valid_mask = (avg_price.tensor > 5) & (avg_price.tensor < 1000)
valid_symbols = data.symbols[valid_mask.cpu().numpy()]
```

### ペアトレーディング

```python
# 2銘柄間のスプレッドを計算
spread = data[:, "AAPL"].close - data[:, "GOOG"].close

# スプレッドを正規化
spread_mean = spread.mean(axis=0)
spread_std = spread.std(axis=0)
zscore = (spread - spread_mean) / spread_std

# シグナル生成
long_signal = zscore < -2    # AAPL 買い、GOOG 売り
short_signal = zscore > 2    # AAPL 売り、GOOG 買い
```
