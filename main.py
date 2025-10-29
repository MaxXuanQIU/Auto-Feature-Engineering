import pandas as pd
from src.config import (
    DATA_PATH, TARGET_COL, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    CV, LOGREG_BASE_PARAMS, LOGREG_AUTO_PARAMS, RESULTS_CSV
)
from src.data import load_data
from src.model import make_preprocessor, make_logreg, make_pipeline, evaluate_cv
from src.traditional import (
    add_log_features, add_bins_features, add_ratio_diff_features, evaluate_variant
)
from src.autofe import run_dfs

# 1) 读数与基本分割
X_base, y, data = load_data(DATA_PATH, show_info=True)

# 2) 基线：标准缩放+OHE
pre_base = make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, poly_interactions=False)
clf_base = make_logreg(LOGREG_BASE_PARAMS)
pipe_base = make_pipeline(pre_base, clf_base)
base_metrics = evaluate_cv(pipe_base, X_base, y, CV)
print("--- 基线模型性能 (交叉验证平均) ---")
print(f"逻辑回归 AUC: {base_metrics['AUC']:.4f}")
print(f"逻辑回归 准确率: {base_metrics['Accuracy']:.4f}")
print(f"逻辑回归 F1-score: {base_metrics['F1']:.4f}")

# 3) 自动化特征工程
X_auto, y_auto, feature_defs = run_dfs(data, CATEGORICAL_FEATURES, TARGET_COL, max_depth=1)
pre_auto = make_preprocessor(num_cols=[], cat_cols=[], poly_interactions=False)  # 仅做标准化
# 上面这个预处理器只含数值缩放，如果 X_auto 全是数值，可直接用 StandardScaler；
# 这里复用 make_preprocessor 简化接口：不给列，就等价空变换。
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
auto_model = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', make_logreg(LOGREG_AUTO_PARAMS))])
auto_metrics = evaluate_cv(auto_model, X_auto, y_auto, CV)
print("\n--- 自动化特征工程后模型性能 (交叉验证平均) ---")
print(f"逻辑回归 AUC: {auto_metrics['AUC']:.4f}")
print(f"逻辑回归 准确率: {auto_metrics['Accuracy']:.4f}")
print(f"逻辑回归 F1-score: {auto_metrics['F1']:.4f}")

# 4) 多种“较弱传统特征工程”方案
results = []
results.append({'方案': 'Baseline(标准缩放+OHE)',
                'AUC': round(base_metrics['AUC'], 4),
                'Accuracy': round(base_metrics['Accuracy'], 4),
                'F1': round(base_metrics['F1'], 4)})

# 4.1 Log1p
X_log = add_log_features(X_base)
num_log = NUMERIC_FEATURES + [c for c in X_log.columns if c.endswith('_log1p')]
cat_log = CATEGORICAL_FEATURES
results.append(evaluate_variant('Log1p(偏态数值)', X_log, y, num_log, cat_log, CV, LOGREG_BASE_PARAMS))

# 4.2 分箱
X_bin = add_bins_features(X_base)
num_bin = NUMERIC_FEATURES
cat_bin = CATEGORICAL_FEATURES + ['age_bin', 'time_bin']
results.append(evaluate_variant('分箱(age/time)', X_bin, y, num_bin, cat_bin, CV, LOGREG_BASE_PARAMS))

# 4.3 比率/差分
X_ratio = add_ratio_diff_features(X_base)
new_ratio_cols = [c for c in X_ratio.columns if c not in X_base.columns and c not in CATEGORICAL_FEATURES]
num_ratio = NUMERIC_FEATURES + new_ratio_cols
cat_ratio = CATEGORICAL_FEATURES
results.append(evaluate_variant('简单比率/差分', X_ratio, y, num_ratio, cat_ratio, CV, LOGREG_BASE_PARAMS))

# 4.4 二阶交互（仅交互）
results.append(evaluate_variant('二阶数值交互(仅交互)', X_base, y, NUMERIC_FEATURES, CATEGORICAL_FEATURES, CV, LOGREG_BASE_PARAMS, use_poly=True))

# 4.5 组合弱特征
X_weak_all = add_bins_features(add_ratio_diff_features(add_log_features(X_base)))
new_num_cols = [c for c in X_weak_all.columns if c not in X_base.columns and c not in ['age_bin', 'time_bin']]
num_weak_all = NUMERIC_FEATURES + new_num_cols
cat_weak_all = CATEGORICAL_FEATURES + ['age_bin', 'time_bin']
results.append(evaluate_variant('组合弱特征(log+bin+ratio)', X_weak_all, y, num_weak_all, cat_weak_all, CV, LOGREG_BASE_PARAMS))

# 4.6 自动化特征工程对比
results.append({'方案': 'AutoFE(Featuretools DFS)',
                'AUC': round(auto_metrics['AUC'], 4),
                'Accuracy': round(auto_metrics['Accuracy'], 4),
                'F1': round(auto_metrics['F1'], 4)})

# 5) 输出对比表
results_df = pd.DataFrame(results)
print("\n=== 各方案对比（5折CV均值） ===")
print(results_df.to_string(index=False))
results_df.to_csv(RESULTS_CSV, index=False, encoding='utf-8-sig')
print(f"\n结果已保存: {RESULTS_CSV}")