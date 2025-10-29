import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import featuretools as ft

# 1. 数据加载与探索
data = pd.read_csv('data/heart_failure_clinical_records.csv')
print(data.head())
print(data.info())
print(f"数据形状: {data.shape}")

# 定义目标变量
y = data['DEATH_EVENT']
X_base = data.drop('DEATH_EVENT', axis=1)

# 2. 基线模型 - 「弱」手工特征
# 2.1 区分数值和类别特征（根据数据集描述）
numeric_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
categorical_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

# 2.2 创建一个预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features) # drop='first'避免共线性
    ])

# 2.3 创建建模管道
base_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# 2.4 使用交叉验证评估基线模型
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_auc_scores = cross_val_score(base_model, X_base, y, cv=cv, scoring='roc_auc')
base_acc_scores = cross_val_score(base_model, X_base, y, cv=cv, scoring='accuracy')
base_f1_scores = cross_val_score(base_model, X_base, y, cv=cv, scoring='f1')

print("--- 基线模型性能 (交叉验证平均) ---")
print(f"逻辑回归 AUC: {base_auc_scores.mean():.4f} (+/- {base_auc_scores.std() * 2:.4f})")
print(f"逻辑回归 准确率: {base_acc_scores.mean():.4f} (+/- {base_acc_scores.std() * 2:.4f})")
print(f"逻辑回归 F1-score: {base_f1_scores.mean():.4f} (+/- {base_f1_scores.std() * 2:.4f})")

# 3. 自动化特征工程
# 3.1 为FeatureTools准备数据（它需要一个EntitySet）
es = ft.EntitySet(id='heart_failure_data')
# 添加主表
es = es.add_dataframe(dataframe_name='patients',
                      dataframe=data,
                      index='id', # 如果没有id列，可以用`data.reset_index().rename(columns={'index': 'id'})`创建一个
                      make_index=True) # 如果原数据没有索引，让FT自动创建一个

# 3.2 深度特征合成 (DFS)
# 这里就是「魔法」发生的地方！一键生成大量特征。
features, feature_defs = ft.dfs(entityset=es,
                                target_dataframe_name='patients',
                                max_depth=2, # 控制特征复杂度，2层就够
                                verbose=True)

# 清理因DFS可能产生的无穷大和缺失值
features = features.replace([np.inf, -np.inf], np.nan)
features = features.fillna(features.mean()) # 用均值填充NaN

# 确保目标变量还在
if 'DEATH_EVENT' in features.columns:
    y_auto = features['DEATH_EVENT']
    X_auto = features.drop('DEATH_EVENT', axis=1)
    
    # 在深度特征合成之后，添加这些检查代码：
    print("\n=== 特征生成验证 ===")
    print(f"原始特征数量: {X_base.shape[1]}")
    print(f"自动化特征生成后特征数量: {X_auto.shape[1]}")
    print(f"理论上应该生成的特征数量: {len(feature_defs)}")

    # 查看具体生成了哪些特征
    print("\n--- 新生成的特征列表 ---")
    for i, feature in enumerate(feature_defs):
        print(f"{i+1}: {feature.get_name()}")

    # 查看前几行新特征数据
    print("\n--- 新特征数据样例 ---")
    print(X_auto.head())
    print(X_auto.columns.tolist())  # 查看所有列名

    # 比较原始数据和特征工程后的列名
    print("\n--- 列名对比 ---")
    original_columns = set(X_base.columns)
    new_columns = set(X_auto.columns)
    print(f"原始列: {original_columns}")
    print(f"新列: {new_columns}")
    print(f"新增的列: {new_columns - original_columns}")
    print(f"消失的列: {original_columns - new_columns}")
    
    print(f"\n自动化特征生成后，特征数量从 {X_base.shape[1]} 增加到 {X_auto.shape[1]}")

    # 4. 在新特征上训练并评估模型
    # 我们仍然使用同样的逻辑回归模型，保证对比公平
    auto_model = Pipeline(steps=[
        ('preprocessor', StandardScaler()), # 现在所有特征都是数值，只需标准化
        ('classifier', LogisticRegression(random_state=42))
    ])

    auto_auc_scores = cross_val_score(auto_model, X_auto, y_auto, cv=cv, scoring='roc_auc')
    auto_acc_scores = cross_val_score(auto_model, X_auto, y_auto, cv=cv, scoring='accuracy')
    auto_f1_scores = cross_val_score(auto_model, X_auto, y_auto, cv=cv, scoring='f1')

    print("\n--- 自动化特征工程后模型性能 (交叉验证平均) ---")
    print(f"逻辑回归 AUC: {auto_auc_scores.mean():.4f} (+/- {auto_auc_scores.std() * 2:.4f})")
    print(f"逻辑回归 准确率: {auto_acc_scores.mean():.4f} (+/- {auto_acc_scores.std() * 2:.4f})")
    print(f"逻辑回归 F1-score: {auto_f1_scores.mean():.4f} (+/- {auto_f1_scores.std() * 2:.4f})")

    # 5. 性能对比
    improvement_auc = (auto_auc_scores.mean() - base_auc_scores.mean()) / base_auc_scores.mean() * 100
    print(f"\n*** AUC 相对提升了: {improvement_auc:.2f}% ***")

else:
    print("错误：未找到目标变量'DEATH_EVENT'。")