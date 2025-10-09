# qfeval-data
[![python](https://img.shields.io/badge/python-%3E=3.9-blue.svg)](https://pypi.org/project/qfeval_data/)
[![pypi](https://img.shields.io/pypi/v/qfeval_data.svg)](https://pypi.org/project/qfeval_data/)
[![CI](https://github.com/pfnet-research/qfeval-data/actions/workflows/ci-python.yaml/badge.svg)](https://github.com/pfnet-research/qfeval-data/actions/workflows/ci-python.yaml)
[![codecov](https://codecov.io/gh/pfnet-research/qfeval-data/graph/badge.svg?token=5A02B1JV7V)](https://codecov.io/gh/pfnet-research/qfeval-data)
[![downloads](https://img.shields.io/pypi/dm/qfeval_data)](https://pypi.org/project/qfeval_data)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

qfevalは、Preferred Networks 金融チームが開発している、金融時系列処理のためのフレームワークです。
データ形式の仕様定義、金融時系列データを効率的に扱うためのクラス/関数群、および金融時系列モデルの評価フレームワークが含まれます。

qfeval-dataは、qfevalの中でも、金融時系列データを効率的に扱うためのデータフレームを提供します。

---

qfeval is a framework developed by Preferred Networks' Financial Solutions team for processing financial time series data.
It includes: data format specification definitions, a set of classes/functions for efficiently handling financial time series data, and a framework for evaluating financial time series models.

qfeval-data provides data frames specifically designed for efficiently handling financial time series data within qfeval.

## Installation

```bash
pip install qfeval_data
```

## Usage
TBD

## リリース手順

1. `release/X.X.X` のブランチを作成する。
2. version.yaml (Bump) のワークフローが実行され、`Bumping version from Z.Z.Z to X.X.X` というタイトルのプルリクエストが作成されるので、これをマージする。
3. `release/X.X.X` ブランチを `master` にマージするプルリクエスト（タイトルは `Release/X.X.X` のままで OK）を作成する。
4. 他の人から Approval を得て、`Release/X.X.X` のプルリクエストのマージをする。
5. [Release ワークフロー](https://github.com/pfnet-research/qfeval-data/actions/workflows/release.yaml) が走るのでこれの完了を待ち、 PyPI の [qfeval/data](https://pypi.org/project/qfeval_data/#history) で新しいバージョンが追加されたことを確認する。
