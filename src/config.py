import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Data Configuration
DATA_PATH = 'data/heart_failure_clinical_records.csv'
TARGET_COL = 'DEATH_EVENT'
ID_COL = 'id'

# Feature Column Definitions
NUMERIC_FEATURES = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction',
    'platelets', 'serum_creatinine', 'serum_sodium', 'time'
]
CATEGORICAL_FEATURES = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

# Cross-Validation Configuration
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model Hyperparameters
LOGREG_BASE_PARAMS = dict(random_state=42, solver='lbfgs', max_iter=2000, C=0.5)
LOGREG_AUTO_PARAMS = dict(random_state=42, solver='lbfgs', max_iter=2000, C=0.5)

# Output
RESULTS_CSV = 'results_comparison.csv'