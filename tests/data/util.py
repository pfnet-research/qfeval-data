import io
import typing

import numpy as np
import pandas as pd


def timestamps(n: int) -> np.ndarray:
    start_timestamp: np.ndarray = np.array("2000-01-01", dtype="datetime64[D]")
    ts = start_timestamp + np.arange(n)
    return typing.cast(np.ndarray, ts.astype("datetime64[us]"))


def symbols(n: int) -> np.ndarray:
    return np.array([f"X-TEST:{i:04}" for i in range(n)])


def create_simple_dataframe() -> pd.DataFrame:
    return (
        pd.read_csv(
            io.StringIO(
                "timestamp,symbol,open,high,low,close\n"
                + "2010-01-01,AAPL,100,120,90,100\n"
                + "2010-01-01,GOOG,200,220,190,200\n"
                + "2010-01-02,AAPL,110,130,100,110\n"
                + "2010-01-03,GOOG,190,210,180,200\n"
            )
        )
        .set_index(["timestamp", "symbol"])
        .astype("float32")
        .reset_index()
    )
