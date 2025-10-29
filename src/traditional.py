import numpy as np
import pandas as pd
from typing import Dict
from .model import make_preprocessor, make_logreg, make_pipeline, evaluate_cv

def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ['creatinine_phosphokinase', 'platelets', 'serum_creatinine']:
        out[f'{col}_log1p'] = np.log1p(out[col].astype(float))
    return out

def add_bins_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['age_bin'] = pd.qcut(out['age'], q=5, duplicates='drop').astype('category')
    out['time_bin'] = pd.qcut(out['time'], q=5, duplicates='drop').astype('category')
    return out

def add_ratio_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6
    out['cpk_per_age'] = out['creatinine_phosphokinase'] / (out['age'] + eps)
    out['platelets_per_sodium'] = out['platelets'] / (out['serum_sodium'] + eps)
    out['ef_per_age'] = out['ejection_fraction'] / (out['age'] + eps)
    out['creatinine_over_platelets'] = out['serum_creatinine'] / (out['platelets'] + eps)
    out['time_over_age'] = out['time'] / (out['age'] + eps)
    out['sodium_minus_creatinine'] = out['serum_sodium'] - out['serum_creatinine']
    return out

def evaluate_variant(name: str, X: pd.DataFrame, y: pd.Series,
                     num_cols, cat_cols, cv, logreg_params, use_poly: bool = False) -> Dict:
    pre = make_preprocessor(num_cols, cat_cols, poly_interactions=use_poly)
    clf = make_logreg(logreg_params)
    pipe = make_pipeline(pre, clf)
    metrics = evaluate_cv(pipe, X, y, cv)
    return {'方案': name, 'AUC': round(metrics['AUC'], 4),
            'Accuracy': round(metrics['Accuracy'], 4), 'F1': round(metrics['F1'], 4)}