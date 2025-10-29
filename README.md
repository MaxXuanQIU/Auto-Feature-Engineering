# Auto-Feature-Engineering

A small, reproducible baseline to compare “weak” traditional feature engineering vs. automated feature engineering (Featuretools DFS) on tabular classification.

## What’s inside

- Baseline: StandardScaler + OneHotEncoder + LogisticRegression (5-fold CV).
- Weak traditional FE recipes (modular, easy to extend):
  - Log1p for skewed numeric columns
  - Quantile binning for age/time
  - Simple ratios and differences
  - Second-order numeric interactions (interaction-only)
  - Combined weak features (log1p + binning + ratios)
- AutoFE: Featuretools DFS with transform primitives selected dynamically per your local installation.
- Unified evaluation: AUC / Accuracy / F1 (5-fold CV) and a comparison table saved to CSV.

## Project structure

```
Auto-Feature-Engineering/
├─ src/
│  ├─ __init__.py
│  ├─ config.py              # Data path, columns, CV, model params, output path
│  ├─ data.py                # Data loader
│  ├─ model.py               # Preprocessor, pipeline, CV evaluator
│  ├─ traditional.py         # Weak FE recipes + evaluator
│  └─ autofe.py              # Featuretools DFS wrapper
├─ data/
│  └─ heart_failure_clinical_records.csv  # Example dataset (configure path in config.py)
├─ main.py                   # Orchestration and result comparison
└─ README.md
```

## Setup (Windows, PowerShell)

````powershell
# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -U pip
pip install pandas numpy scikit-learn featuretools woodwork
````

Optional: if you use a constraints file or requirements.txt, adjust accordingly.

## Configure

Edit src/config.py:

- DATA_PATH: path to your CSV.
- TARGET_COL: target column name.
- NUMERIC_FEATURES / CATEGORICAL_FEATURES: base schema for baseline and weak FE.
- CV and LOGREG_*: CV and model hyperparameters.
- RESULTS_CSV: output path for the comparison table.

## Run

````powershell
python main.py
````

Console will print:
- Baseline metrics (AUC / Accuracy / F1, CV average)
- AutoFE metrics
- A final comparison table and save it to RESULTS_CSV

Example table (values are reproducible, based on heart_failure_clinical_records.csv):
| 方案 | AUC | Accuracy | F1 |
|------|-----|----------|-----|
| Baseline(标准缩放+OHE) | 0.9003 | 0.8442 | 0.7387 |
| Log1p(偏态数值) | 0.9052 | 0.8494 | 0.7476 |
| 分箱(age/time) | 0.9269 | 0.8668 | 0.7731 |
| 简单比率/差分 | 0.9079 | 0.8480 | 0.7462 |
| 二阶数值交互(仅交互) | 0.9239 | 0.8682 | 0.7861 |
| 组合弱特征(log+bin+ratio) | 0.9344 | 0.8778 | 0.7973 |
| AutoFE(Featuretools DFS) | 0.9484 | 0.8916 | 0.8197 |

## How it works

- Traditional FE is implemented in src/traditional.py and evaluated through a shared pipeline/evaluator in src/model.py.
- AutoFE uses src/autofe.py:
  - Converts binary categorical columns to bool
  - Builds a single-dataframe EntitySet
  - Selects transform primitives that exist in your local Featuretools install
  - Runs DFS with max_depth=1 for single-table scenarios
  - Aligns the target back by index and returns X_auto, y_auto

## Extend: add a new weak FE recipe

1) Add a function in src/traditional.py that returns a new DataFrame with added columns.

2) In main.py, construct the numeric/categorical column lists for that variant and call evaluate_variant(...). All preprocessing, OHE, scaling, and CV evaluation are handled for you.

Example:

````python
# in traditional.py
def add_mean_centering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ['age', 'time']:
        out[f'{col}_centered'] = out[col] - out[col].mean()
    return out
````

````python
# in main.py
X_mc = add_mean_centering(X_base)
num_mc = NUMERIC_FEATURES + [c for c in X_mc.columns if c.endswith('_centered')]
cat_mc = CATEGORICAL_FEATURES
results.append(evaluate_variant('均值中心化(age/time)', X_mc, y, num_mc, cat_mc, CV, LOGREG_BASE_PARAMS))
````

## Tips and troubleshooting

- DFS produced no new features:
  - Single-table DFS is limited to transform primitives (no aggregations).
  - Ensure trans_primitives is non-empty. The project auto-selects available primitives; upgrade Featuretools for more options.
- ConvergenceWarning from LogisticRegression:
  - Increase max_iter or reduce C (stronger regularization) in src/config.py.
  - Use solver='saga' for high-dimensional data, optionally penalty='l1' for sparsity.
  - Reduce feature count: lower DFS primitive set/limit or prune with downstream selection.
- Want more informative AutoFE:
  - Introduce related child tables and relationships to unlock aggregation primitives in DFS.

## Reproducibility

- CV uses a fixed random_state=42; update in config.py if needed.

## License

- Provide a license if you plan to share; otherwise keep private for internal evaluation.
