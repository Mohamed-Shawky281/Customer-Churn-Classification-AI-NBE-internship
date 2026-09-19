# Credit Card Customer Churn Classification

An end-to-end machine learning pipeline that predicts which credit card customers are likely to churn, built on a 20,000-record banking dataset. The project follows the CRISP-DM workflow — from exploratory analysis and preprocessing through feature engineering, multi-model training, regularization, and final champion selection on a held-out test set.

Developed during an AI track internship at the **National Bank of Egypt (NBE)**, Data Warehouse department.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Pipeline Overview](#pipeline-overview)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing](#preprocessing)
- [Feature Engineering](#feature-engineering)
- [Modelling](#modelling)
- [Results](#results)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Limitations & Future Work](#limitations--future-work)

---

## Problem Statement

Credit card churn is a direct drain on recurring revenue. Every customer who closes or stops using their card represents lost interest income, annual fees, and interchange revenue — and replacing them costs significantly more than retaining them.

The goal of this project is to identify **at-risk customers before they churn**, enabling proactive retention rather than reactive damage control. Because the business cost of missing a churner outweighs the cost of a false alarm, models are evaluated with particular attention to **recall** on the churned class.

---

## Dataset

| Property | Value |
|---|---|
| Records | 20,000 |
| Raw features | 60 (including target) |
| Target | `churn` (binary) |
| Class balance | ~37.2% churned |
| Features after preprocessing | 56 |

### Feature Groups

- **Spend behavior** — category-level and aggregate spend across 1, 3, 6, and 12-month windows
- **Transaction activity** — counts, frequency, active days, recency
- **Risk signals** — payment delays, late-payment flags, credit utilization, inactivity
- **Account attributes** — credit limit, tenure, product holdings, account type
- **Demographics** — age, gender, marital status, education, employment, residence

### Missing Values

Seven fields contained missing values, all handled via imputation rather than dropped:

| Feature | Missing |
|---|---|
| `average_transaction_value` | 20.04% |
| `feature_3` | 20.04% |
| `monthly_spend_cash` | 10.08% |
| `loyalty_score` | 10.08% |
| `total_bank_balance` | 10.08% |
| `monthly_spend_payments` | 9.85% |
| `monthly_spend_installments` | 7.59% |

---

## Pipeline Overview

```
Raw Data (20,000 × 60)
        ↓
Data Quality Checks — nulls, duplicates, cardinality, garbage values
        ↓
Exploratory Data Analysis — distributions, correlations, segment analysis
        ↓
Preprocessing — MICE imputation → encoding → scaling
        ↓
Feature Engineering — cat_sum, age_tier, is_expiring_this_year, balance_minus_limit
        ↓
Train / Validation / Test Split (70 / 15 / 15, stratified)
        ↓
Model Training — 10 algorithms, tuned and regularized
        ↓
Validation-based selection → Final test-set evaluation → Champion model
```

---

## Exploratory Data Analysis

The EDA phase covered distribution analysis, correlation structure, and segment-level churn behavior.

**Techniques applied:**
- Histograms and boxplots across all numeric features to assess skew and outlier density
- Correlation heatmaps (full and lower-triangle) to surface redundant feature pairs
- Pairwise scatter plots for feature pairs with |r| > 0.7
- Per-feature correlation ranking against the target
- Stacked bar charts comparing churn rate across demographic segments
- Grouped boxplots comparing spend distributions by category and churn status

**Notable outcome:** demographic attributes (gender, marital status, residence type, employment status) showed churn rates within roughly one percentage point of each other — a near-flat ~62/38 split across every category. This established early that the predictive signal lives in **behavioral** features, not demographic ones.

---

## Preprocessing

### Imputation — MICE

Missing values were filled using **Multiple Imputation by Chained Equations** (`IterativeImputer`), which models each incomplete column as a function of the others and refines estimates over repeated cycles.

```python
IterativeImputer(
    estimator=HistGradientBoostingRegressor(random_state=42),
    max_iter=10,
    initial_strategy='mean',
    random_state=42
)
```

A gradient-boosting estimator was chosen over the default linear model to capture non-linear relationships between features and to remain numerically stable across widely varying feature scales.

### Encoding

| Type | Features | Rationale |
|---|---|---|
| **Ordinal** | `education_level`, `spend_volatility`, `age_tier` | Categories carry a genuine rank ordering |
| **One-Hot** | `gender`, `marital_status`, `residence_type`, `account_types`, `employment_status` | No inherent ordering; `drop_first=True` avoids the dummy variable trap |

### Scaling

Scaler choice was driven by each feature's observed distribution rather than applied uniformly:

- **StandardScaler** — applied to approximately normally distributed features
- **RobustScaler** — applied to skewed, outlier-heavy features, since it centers on the median and scales by IQR rather than being pulled by extreme values

### Dropped Columns

Identifiers and non-predictive fields were removed: `acc_ref_num`, `cust_id`, `plastic_masked_pan`, `index`, `month_key`, `expiration_date`, and the anonymized `feature_1`–`feature_3`. `act_ref_num` was retained as the DataFrame index so predictions remain traceable to specific accounts.

---

## Feature Engineering

| Feature | Definition | Purpose |
|---|---|---|
| `cat_sum` | Sum of all seven monthly spend categories | Total spend footprint in a single dimension |
| `age_tier` | Age bucketed into life-stage bands, ordinally encoded | Captures life-stage effects that raw age may miss |
| `is_expiring_this_year` | Binary flag derived from `expiration_date` | Card renewal is a natural churn decision point |
| `balance_minus_limit` | `total_bank_balance − credit_limit` | Relative financial headroom against available credit |

---

## Modelling

Ten algorithms were trained and compared:

| Family | Models |
|---|---|
| Linear | Logistic Regression, Linear Discriminant Analysis |
| Kernel | Support Vector Classifier (RBF) |
| Bagging | Random Forest, Extra Trees |
| Boosting | XGBoost, LightGBM, CatBoost, Gradient Boosting, AdaBoost |

### Hyperparameter Tuning

Logistic Regression was tuned via exhaustive grid search across 40 combinations of `C`, `penalty` (L1/L2), `solver`, and `class_weight`, with each combination scored on the validation set across accuracy, precision, recall, F1, and ROC-AUC.

### Overfitting Control

Initial tree-based models showed substantial train-validation gaps — Random Forest with unconstrained depth reached a perfect 1.000 training F1 against 0.834 on validation. A structured regularization pass was applied:

| Model | Interventions |
|---|---|
| **XGBoost** | `max_depth` 5→3, `learning_rate` 0.05→0.02, `reg_alpha` 0→1.0, `reg_lambda` 1→10, `min_child_weight` 1→10, `gamma` 0→1.0, early stopping (30 rounds) |
| **LightGBM** | `max_depth` −1→4, `num_leaves` 31→15, `reg_alpha` 0→1.0, `reg_lambda` 0→10, `min_child_samples`→50, early stopping (30 rounds) |
| **CatBoost** | `depth` 6→4, `learning_rate` 0.05→0.02, `l2_leaf_reg` 3→20, early stopping (30 rounds) |
| **Random Forest** | `max_depth` None→6, `min_samples_split` 2→50, `min_samples_leaf` 1→25 |

Early stopping was monitored against the validation set, never the test set, so that final test performance remains an unbiased estimate.

### Evaluation Protocol

A shared `report()` function computed accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and a bootstrapped ROC-AUC standard deviation (500 resamples) for every model on every split. Model selection was performed entirely on the validation set; the test set was evaluated exactly once, at the end.

---

## Results

### Final Test Set Performance

| Rank | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| 1 | **AdaBoost** | 0.8630 | 0.7316 | **0.9982** | **0.8444** | 0.9567 | 0.8933 |
| 2 | CatBoost | 0.8630 | 0.7316 | 0.9982 | 0.8444 | 0.9482 | 0.8917 |
| 3 | LDA | 0.8620 | 0.7317 | 0.9937 | 0.8428 | 0.9558 | 0.9191 |
| 4 | Logistic Regression | 0.8630 | 0.7398 | 0.9749 | 0.8413 | 0.9577 | 0.9237 |
| 5 | SVC | 0.8657 | 0.7767 | 0.8970 | 0.8326 | 0.9519 | 0.9093 |
| 6 | XGBoost | 0.8657 | 0.7888 | 0.8729 | 0.8287 | 0.9571 | 0.9214 |
| 7 | Extra Trees | 0.8697 | 0.8168 | 0.8380 | 0.8272 | 0.9578 | 0.9251 |
| 8 | Gradient Boosting | 0.8640 | 0.7899 | 0.8648 | 0.8256 | 0.9555 | 0.9188 |
| 9 | LightGBM | 0.8640 | 0.7899 | 0.8648 | 0.8256 | 0.9562 | 0.9179 |
| 10 | Random Forest | 0.8693 | **0.8988** | 0.7314 | 0.8065 | 0.9579 | 0.9261 |

### Champion Model — AdaBoost

```
Accuracy   : 0.8630
Precision  : 0.7316
Recall     : 0.9982
F1-Score   : 0.8444
ROC-AUC    : 0.9567
```

AdaBoost achieves near-total recall, correctly flagging 99.8% of customers who actually churned. For a retention use case this is the operationally meaningful trade-off: the model misses almost no at-risk customers, at the cost of some false positives that a retention campaign can absorb far more cheaply than a lost account.

The ranking is metric-dependent and worth noting explicitly. Random Forest delivers the strongest precision (0.8988) and the best ROC-AUC and PR-AUC, but its recall of 0.7314 means it misses more than a quarter of genuine churners. Selecting on F1 with a recall-weighted business rationale favors AdaBoost.

---

## Key Findings

### 1. Churn is driven by inactivity, not demographics

Correlation against the target ranked features as follows:

| Rank | Feature | Correlation with churn |
|---|---|---|
| 1 | `inactivity_flag` | **+0.764** |
| 2 | `months_since_last_txn` | **+0.671** |
| 3 | `total_bank_balance` | −0.204 |
| 4 | `total_spend_last_year` | −0.201 |
| 5 | `total_spend_last_6_months` | −0.200 |
| 6 | `max_monthly_spend` | −0.198 |
| 7 | `credit_limit` | −0.197 |

Two recency and engagement signals dominate every other feature by a wide margin. Feature-importance rankings across all tree-based models independently converged on the same two fields, confirming the finding is not an artifact of any single algorithm.

### 2. Demographic attributes carry almost no standalone signal

Gender, marital status, residence type, and employment status each showed churn rates within roughly one percentage point across all their categories. This is a genuinely useful negative result: retention targeting should be driven by behavior, not customer profile.

### 3. Simpler models were competitive with — and more stable than — complex ensembles

Logistic Regression and LDA landed within 0.003 F1 of the champion while exhibiting negligible train-validation gaps. Heavily regularized gradient boosting models required substantial constraint tuning to close their overfitting gaps and still did not outperform the linear baselines by a meaningful margin, suggesting the dataset's signal is largely linearly separable along the dominant inactivity features.

---

## Repository Structure

```
.
├── Churn_Classification.ipynb      # Complete pipeline: EDA → preprocessing → modelling → evaluation
├── README.md
└── models/                         # Serialized trained models (.pkl)
    ├── logistic_regression_tuned.pkl
    ├── adaboost_model.pkl
    ├── xgboost_regularized.pkl
    ├── catboost_regularized.pkl
    ├── lightgbm_regularized.pkl
    ├── random_forest_regularized.pkl
    ├── gradient_boosting_regularized.pkl
    ├── extratrees_model.pkl
    ├── svc_model.pkl
    └── lda_model.pkl
```

---

## Getting Started

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost openpyxl
```

### Running the Notebook

1. Place `credit_card_churn_dataset_V4.xlsx` in the notebook's working directory
2. Open `Churn_Classification.ipynb`
3. Run all cells in order — the pipeline executes end to end, from raw data to champion selection

### Loading a Trained Model

```python
import pickle

with open('models/adaboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
```

> Note: models expect input that has already passed through the same imputation, encoding, and scaling steps used during training. The fitted transformers must be applied to any new data before inference.

---

## Limitations & Future Work

### Known Limitations

- **Preprocessing applied before splitting.** Imputation and scaling were fitted on the full dataset rather than on the training split alone. This introduces mild data leakage and may render reported metrics slightly optimistic. A production pipeline should fit all transformers on training data only and apply them to validation and test via `transform`.
- **Transformers not serialized.** Only the models were pickled. Reusing them on new data requires re-fitting the scalers, encoders, and imputer.
- **Data quality inconsistencies.** A small number of records contain logically impossible values (for example, `average_monthly_spend` below `min_monthly_spend`), likely originating in the source data rather than the pipeline.
- **Anonymized features.** `feature_1`–`feature_3` had no documented meaning and were dropped to preserve interpretability, despite `feature_1` showing a correlation with churn comparable to the top spend features.

### Future Work

- Restructure preprocessing into a leakage-free `Pipeline` / `ColumnTransformer` fitted on training data only
- Serialize fitted transformers alongside models for reproducible inference
- Evaluate SMOTE and cost-sensitive learning to further improve minority-class recall
- Tune the classification threshold explicitly against business cost rather than defaulting to 0.5
- Add SHAP values for per-customer, case-level explanations of churn risk
- Deploy as a scheduled batch scoring job feeding retention alerts into CRM
- Implement drift monitoring to detect degradation as customer behavior shifts

---

## Acknowledgements

Developed as part of the AI track internship at the National Bank of Egypt (NBE), Data Warehouse department.
