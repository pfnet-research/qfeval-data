# qfeval-data ドキュメント

[[English](README.md)]

**qfeval-data** は金融時系列データを効率的に操作するための Python ライブラリです。タイムスタンプとシンボルでインデックス付けされた金融データを扱うための、PyTorch テンソルをベースにした特殊なデータ構造である `Data` クラスを提供します。

## 主な機能

- **PyTorch バックエンド**: PyTorch テンソルによる GPU アクセラレーションの完全サポート
- **遅延スライシング**: 不要なコピーを行わない効率的なデータアクセス
- **金融特化**: OHLCV データ、メトリクス、テクニカル指標の組み込みサポート
- **柔軟な I/O**: CSV、DataFrame からの読み込み、またはテンソルからの直接構築
- **可視化**: matplotlib による統合されたローソク足チャートとプロット機能

## インストール

```bash
pip install qfeval-data

# プロット機能付き
pip install qfeval-data[plot]
```

## クイックスタート

```python
from qfeval_data import Data
import pandas as pd

# CSV から読み込み
data = Data.from_csv("prices.csv")

# または DataFrame から
df = pd.DataFrame({
    "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "symbol": ["AAPL", "GOOG", "AAPL", "GOOG"],
    "close": [150.0, 140.0, 152.0, 142.0],
})
data = Data.from_dataframe(df)

# アクセスと操作
returns = data.pct_change()
avg_return = returns.mean(axis=0)

# メトリクスの計算
metrics = data.metrics()
print(metrics.to_dataframe())
```

## ドキュメント目次

- [Data クラスリファレンス](data.ja.md) - `Data` クラスの完全な API リファレンス
- [Flattener リファレンス](flattener.ja.md) - `Data` とフラットテンソル間の変換
- [ユーティリティ関数](util.ja.md) - 配列と時間操作のヘルパー関数
- [使用例](examples.ja.md) - 一般的な使用パターンとレシピ

## 必要条件

- Python >= 3.9
- PyTorch
- NumPy
- pandas
- qfeval-functions

## ライセンス

詳細は [LICENSE](../LICENSE) ファイルを参照してください。
