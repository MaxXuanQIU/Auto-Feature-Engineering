import pandas as pd
from typing import Tuple
from .config import DATA_PATH, TARGET_COL

def load_data(path: str = DATA_PATH, show_info: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    data = pd.read_csv(path)
    if show_info:
        print(data.head())
        print(data.info())
        print(f"数据形状: {data.shape}")
    y = data[TARGET_COL]
    X = data.drop(TARGET_COL, axis=1)
    return X, y, data