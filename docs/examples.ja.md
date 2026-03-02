# 使用例とレシピ

このドキュメントでは qfeval-data の実用的な例と一般的な使用パターンを紹介します。

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

# dtype と device を指定
data = Data.from_csv("prices.csv", dtype=torch.float32, device="cuda")

# 圧縮ファイルは自動的に処理
data = Data.from_csv("prices.csv.xz")
```

**期待される CSV フォーマット:**
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

data = Data.from_csv("prices.csv")

# 特定のカラムを取得
closes = data.close                    # 属性アクセス
closes = data.get("close")             # メソッドアクセス
ohlc = data.get("open", "high", "low", "close")

# 時間でスライス
first_week = data[:5, :]               # 最初の5タイムスタンプ
jan_data = data["2024-01-01":"2024-01-31", :]

# シンボルでスライス
apple = data[:, "AAPL"]                # 単一シンボル
tech = data[:, ["AAPL", "GOOG", "MSFT"]]  # 複数シンボル

# 組み合わせスライス
apple_jan = data["2024-01-01":"2024-01-31", "AAPL"]
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
# 移動平均
ma_20 = data.close.moving_average(window=20)

# ボリンジャーバンド
upper, middle, lower = data.close.bollinger_band(window=20, sigma=2.0)

# apply を使ったカスタムローリング
def rolling_zscore(x):
    from qfeval_functions import functions
    mean = functions.ma(x, 20, dim=0)
    std = functions.mstd(x, 20, dim=0)
    return (x - mean) / std

zscore = data.close.apply(rolling_zscore)
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
print(all_metrics.to_table())

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
    features.append(data.close.pct_change(5).rename("return_5d"))
    features.append(data.close.pct_change(20).rename("return_20d"))

    # 移動平均
    ma_5 = data.close.moving_average(5)
    ma_20 = data.close.moving_average(20)
    features.append((data.close / ma_5 - 1).rename("close_ma5_ratio"))
    features.append((data.close / ma_20 - 1).rename("close_ma20_ratio"))
    features.append((ma_5 / ma_20 - 1).rename("ma5_ma20_ratio"))

    # ボラティリティ
    features.append(data.close.pct_change().apply(
        lambda x: x.abs().rolling(20).mean()
    ).rename("volatility_20d"))

    # 出来高比率
    if "volume" in data.columns:
        vol_ma = data.volume.moving_average(20)
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

data = Data.from_csv("prices.csv")
apple = data[:, "AAPL"]

# プロットタイプを自動検出（OHLC にはローソク足）
apple.plot()
plt.title("AAPL")
plt.show()
```

### ローソク足チャート

```python
# 明示的なローソク足
apple.candlestick()
plt.title("AAPL ローソク足")
plt.show()

# カスタムカラー
apple.candlestick(
    upcolor="#00ff00",
    downcolor="#ff0000",
    width=0.8
)
plt.show()
```

### 折れ線グラフ

```python
# 単一系列
apple.close.line()
plt.title("AAPL 終値")
plt.show()

# 複数系列
fig, ax = plt.subplots()
apple.close.line(ax=ax, label="Close")
apple.close.moving_average(20).line(ax=ax, label="MA20")
plt.legend()
plt.show()
```

### テクニカル指標

```python
# 移動平均オーバーレイ
fig, ax = plt.subplots()
apple.candlestick(ax=ax)
apple.close.plot_moving_average(window=20, ax=ax, color="blue")
plt.show()

# ボリンジャーバンド
fig, ax = plt.subplots()
apple.candlestick(ax=ax)
apple.close.plot_bollinger_band(ax=ax)
plt.show()
```

### 複数サブプロット

```python
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# ボリンジャーバンド付き価格
apple.candlestick(ax=axes[0])
apple.close.plot_bollinger_band(ax=axes[0])
axes[0].set_title("価格")

# 出来高
apple.volume.bar(ax=axes[1])
axes[1].set_title("出来高")

# リターン
apple.close.pct_change().line(ax=axes[2])
axes[2].set_title("日次リターン")

plt.tight_layout()
plt.show()
```

---

## 機械学習との統合

### PyTorch 用データ準備

```python
import torch
from qfeval_data import Data, Flattener

# データ読み込み
data = Data.from_csv("prices.csv")

# 特徴量とターゲットを作成
features = data.get(["open", "high", "low", "close", "volume"])
target = data.close.shift(-1).pct_change()  # 翌日リターン

# NaN を含む行を削除
features = features.dropna()
target = target.dropna()

# アライメント用の Flattener を作成
flattener = Flattener(features, target)

# テンソルに変換
X = flattener.flatten(features)  # 形状: (B, 5)
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

# 訓練
for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
```

### 予測の作成

```python
# 予測を作成
model.eval()
with torch.no_grad():
    predictions = model(X)

# Data 形式に戻す
pred_data = flattener.unflatten(predictions, "prediction")

# 予測を確認
print(pred_data.to_dataframe().head())

# 予測メトリクスを計算
actual = flattener.unflatten(y, "actual")
error = (pred_data - actual).abs()
print(f"平均絶対誤差: {error.mean().tensor.item():.6f}")
```

### 時系列分割

```python
# 時間で分割
split_date = "2023-07-01"
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
# 流動性でフィルタ
avg_volume = data.volume.mean(axis=0)
liquid_mask = avg_volume.tensor > 1_000_000
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
