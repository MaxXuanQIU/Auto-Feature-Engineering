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

# 1) Load data and basic split
X_base, y, data = load_data(DATA_PATH, show_info=True)

# 2) Baseline: Standard Scaling + OHE
pre_base = make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, poly_interactions=False)
clf_base = make_logreg(LOGREG_BASE_PARAMS)
pipe_base = make_pipeline(pre_base, clf_base)
base_metrics = evaluate_cv(pipe_base, X_base, y, CV)
print("--- Baseline Model Performance (Cross-Validation Mean) ---")
print(f"Logistic Regression AUC: {base_metrics['AUC']:.4f}")
print(f"Logistic Regression Accuracy: {base_metrics['Accuracy']:.4f}")
print(f"Logistic Regression F1-score: {base_metrics['F1']:.4f}")

# 3) Automated Feature Engineering
X_auto, y_auto, feature_defs = run_dfs(data, CATEGORICAL_FEATURES, TARGET_COL, max_depth=1)
pre_auto = make_preprocessor(num_cols=[], cat_cols=[], poly_interactions=False)  # Only perform standardization
# The preprocessor above only includes numerical scaling. If X_auto is all numerical, StandardScaler can be used directly;
# Here, make_preprocessor is reused to simplify the interface: not providing columns is equivalent to an empty transformation.
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
auto_model = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', make_logreg(LOGREG_AUTO_PARAMS))])
auto_metrics = evaluate_cv(auto_model, X_auto, y_auto, CV)
print("\n--- Model Performance After Automated Feature Engineering (Cross-Validation Mean) ---")
print(f"Logistic Regression AUC: {auto_metrics['AUC']:.4f}")
print(f"Logistic Regression Accuracy: {auto_metrics['Accuracy']:.4f}")
print(f"Logistic Regression F1-score: {auto_metrics['F1']:.4f}")

# 4) Various "Weaker Traditional Feature Engineering" Scenarios
results = []
results.append({'Scenario': 'Baseline(Standard Scaling+OHE)',
                'AUC': round(base_metrics['AUC'], 4),
                'Accuracy': round(base_metrics['Accuracy'], 4),
                'F1': round(base_metrics['F1'], 4)})

# 4.1 Log1p
X_log = add_log_features(X_base)
num_log = NUMERIC_FEATURES + [c for c in X_log.columns if c.endswith('_log1p')]
cat_log = CATEGORICAL_FEATURES
results.append(evaluate_variant('Log1p(Skewed Numerics)', X_log, y, num_log, cat_log, CV, LOGREG_BASE_PARAMS))

# 4.2 Binning
X_bin = add_bins_features(X_base)
num_bin = NUMERIC_FEATURES
cat_bin = CATEGORICAL_FEATURES + ['age_bin', 'time_bin']
results.append(evaluate_variant('Binning(age/time)', X_bin, y, num_bin, cat_bin, CV, LOGREG_BASE_PARAMS))

# 4.3 Ratio/Difference
X_ratio = add_ratio_diff_features(X_base)
new_ratio_cols = [c for c in X_ratio.columns if c not in X_base.columns and c not in CATEGORICAL_FEATURES]
num_ratio = NUMERIC_FEATURES + new_ratio_cols
cat_ratio = CATEGORICAL_FEATURES
results.append(evaluate_variant('Simple Ratio/Difference', X_ratio, y, num_ratio, cat_ratio, CV, LOGREG_BASE_PARAMS))

# 4.4 Second-order Interactions (Interactions only)
results.append(evaluate_variant('Second-order Numeric Interactions(Interactions only)', X_base, y, NUMERIC_FEATURES, CATEGORICAL_FEATURES, CV, LOGREG_BASE_PARAMS, use_poly=True))

# 4.5 Combined Weak Features
X_weak_all = add_bins_features(add_ratio_diff_features(add_log_features(X_base)))
new_num_cols = [c for c in X_weak_all.columns if c not in X_base.columns and c not in ['age_bin', 'time_bin']]
num_weak_all = NUMERIC_FEATURES + new_num_cols
cat_weak_all = CATEGORICAL_FEATURES + ['age_bin', 'time_bin']
results.append(evaluate_variant('Combined Weak Features(log+bin+ratio)', X_weak_all, y, num_weak_all, cat_weak_all, CV, LOGREG_BASE_PARAMS))

# 4.6 Automated Feature Engineering Comparison
results.append({'Scenario': 'AutoFE(Featuretools DFS)',
                'AUC': round(auto_metrics['AUC'], 4),
                'Accuracy': round(auto_metrics['Accuracy'], 4),
                'F1': round(auto_metrics['F1'], 4)})

# 5) Output comparison table
results_df = pd.DataFrame(results)
print("\n=== Comparison of Scenarios (5-fold CV mean) ===")
print(results_df.to_string(index=False))
results_df.to_csv(RESULTS_CSV, index=False, encoding='utf-8-sig')
print(f"\nResults saved to: {RESULTS_CSV}")