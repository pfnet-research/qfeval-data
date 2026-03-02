# qfeval-data

[[日本語](README.ja.md)]

[![python](https://img.shields.io/badge/python-%3E=3.9-blue.svg)](https://pypi.org/project/qfeval_data/)
[![pypi](https://img.shields.io/pypi/v/qfeval_data.svg)](https://pypi.org/project/qfeval_data/)
[![CI](https://github.com/pfnet-research/qfeval-data/actions/workflows/ci-python.yaml/badge.svg)](https://github.com/pfnet-research/qfeval-data/actions/workflows/ci-python.yaml)
[![codecov](https://codecov.io/gh/pfnet-research/qfeval-data/graph/badge.svg?token=5A02B1JV7V)](https://codecov.io/gh/pfnet-research/qfeval-data)
[![downloads](https://img.shields.io/pypi/dm/qfeval_data)](https://pypi.org/project/qfeval_data)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

qfeval is a framework developed by Preferred Networks' Financial Solutions team for processing financial time series data.
It includes: data format specification definitions, a set of classes/functions for efficiently handling financial time series data, and a framework for evaluating financial time series models.

qfeval-data provides data frames specifically designed for efficiently handling financial time series data within qfeval.

## Installation

```bash
pip install qfeval_data
```

## Usage

See [docs/README.md](docs/README.md) for detailed documentation.

## Release Process

1. Create a `release/X.X.X` branch.
2. The version.yaml (Bump) workflow will run and create a pull request titled `Bumping version from Z.Z.Z to X.X.X`. Merge this PR.
3. Create a pull request to merge the `release/X.X.X` branch into `master` (the title `Release/X.X.X` is fine).
4. Get approval from another team member and merge the `Release/X.X.X` pull request.
5. Wait for the [Release workflow](https://github.com/pfnet-research/qfeval-data/actions/workflows/release.yaml) to complete, then verify the new version appears on PyPI at [qfeval/data](https://pypi.org/project/qfeval_data/#history).
