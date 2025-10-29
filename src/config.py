import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 数据配置
DATA_PATH = 'data/heart_failure_clinical_records.csv'
TARGET_COL = 'DEATH_EVENT'
ID_COL = 'id'

# 特征列定义
NUMERIC_FEATURES = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction',
    'platelets', 'serum_creatinine', 'serum_sodium', 'time'
]
CATEGORICAL_FEATURES = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

# 交叉验证配置
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 模型超参
LOGREG_BASE_PARAMS = dict(random_state=42, solver='lbfgs', max_iter=2000, C=0.5)
LOGREG_AUTO_PARAMS = dict(random_state=42, solver='lbfgs', max_iter=2000, C=0.5)

# 输出
RESULTS_CSV = 'results_comparison.csv'