from typing import Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

def make_preprocessor(num_cols, cat_cols, poly_interactions: bool = False) -> ColumnTransformer:
    if poly_interactions:
        num_transform = Pipeline(steps=[
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ('scaler', StandardScaler())
        ])
    else:
        num_transform = StandardScaler()
    pre = ColumnTransformer(
        transformers=[
            ('num', num_transform, num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    return pre

def make_logreg(params: Dict) -> LogisticRegression:
    return LogisticRegression(**params)

def evaluate_cv(estimator, X, y, cv) -> Dict[str, float]:
    scores = cross_validate(
        estimator, X, y, cv=cv,
        scoring={'auc': 'roc_auc', 'acc': 'accuracy', 'f1': 'f1'},
        n_jobs=None, return_train_score=False
    )
    return {
        'AUC': np.mean(scores['test_auc']),
        'Accuracy': np.mean(scores['test_acc']),
        'F1': np.mean(scores['test_f1']),
    }

def make_pipeline(preprocessor: ColumnTransformer, clf: LogisticRegression) -> Pipeline:
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])