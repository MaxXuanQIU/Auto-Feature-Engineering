from typing import List, Tuple
import numpy as np
import pandas as pd
import featuretools as ft
from .config import ID_COL

DESIRED_TRANS = {
    'add_numeric', 'subtract_numeric', 'multiply_numeric', 'divide_numeric',
    'absolute', 'negate', 'natural_logarithm', 'square_root', 'square',
    'and', 'or', 'not'
}

def available_transform_primitives() -> List[str]:
    avail = ft.primitives.list_primitives()
    return sorted(set(avail[avail['type'] == 'transform']['name']))

def run_dfs(data: pd.DataFrame, categorical_features: list, target_col: str,
            max_depth: int = 1) -> Tuple[pd.DataFrame, pd.Series, list]:
    df = data.copy()
    # 布尔化二元类别
    df[categorical_features] = df[categorical_features].astype(bool)

    es = ft.EntitySet(id='heart_failure_data')
    es = es.add_dataframe(
        dataframe_name='patients',
        dataframe=df,
        index=ID_COL,
        make_index=True
    )

    avail_trans = set(available_transform_primitives())
    trans_primitives = sorted(list(DESIRED_TRANS & avail_trans))
    print("使用的 transform primitives:", trans_primitives)

    features, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name='patients',
        trans_primitives=trans_primitives,
        agg_primitives=[],
        ignore_columns={'patients': [target_col]},
        max_depth=max_depth,
        verbose=True
    )

    # 清理与对齐
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.mean())
    # 目标对齐
    patients_df = es['patients']
    if ID_COL in patients_df.columns:
        y_auto = patients_df.set_index(ID_COL).loc[features.index, target_col]
    else:
        y_auto = patients_df.loc[features.index, target_col]

    # 去除任何潜在的目标泄漏列
    X_auto = features.drop(columns=[target_col], errors='ignore')
    return X_auto, y_auto, feature_defs