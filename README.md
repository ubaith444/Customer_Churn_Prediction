# Telco Customer Churn Prediction and Retention Intelligence

An enterprise-grade, end-to-end Machine Learning system to predict telecom customer attrition, optimize business intervention thresholds, interpret risk drivers with SHAP, and deliver actionable retention recommendations via an interactive web application.

---

## Tech Stack and Architectural Rationale

| Category | Technology | Version | Purpose in Project (What) | Architectural Rationale (Why) |
| :--- | :--- | :--- | :--- | :--- |
| **Core Runtime** | Python | >= 3.11 | Primary language runtime for data pipelines, model training, evaluation scripts, and application logic. | Universal compatibility across the modern ML ecosystem, mature native typing (`typing.Dict`, `typing.Tuple`), and optimized CPython interpreter performance. |
| **Data Manipulation** | pandas | >= 2.0.0 | Tabular data ingestion, missing value imputation, cohort binning (`pd.cut`), schema transformations, and evaluation summaries. | Vectorized columnar execution handles 7,000+ records in milliseconds; expressive indexing and flexible aggregation simplify feature derivation without iterative loops. |
| **Numerical Computing** | NumPy | >= 1.24.0 | Vectorized array transformations, conditional mathematical calculations (`np.where`), threshold sweep computations, and financial simulations. | High-performance C-optimized contiguous array operations deliver near-instant matrix math and vectorized logic for threshold curves and metric derivations. |
| **Machine Learning Pipelines** | scikit-learn | >= 1.3.0 | Unified pipeline composition (`Pipeline`, `ColumnTransformer`), categorical encoding (`OneHotEncoder`), feature scaling (`StandardScaler`), 5-Fold Stratified Cross-Validation (`StratifiedKFold`), baseline models, and evaluation metrics. | Encapsulating data transformations within `Pipeline` guarantees zero data leakage between training folds and held-out validation/test splits. Provides standard interfaces across models. |
| **Gradient Boosting** | XGBoost | >= 2.0.0 | Primary champion model (`XGBClassifier`) tuned via cross-validation with cost-sensitive weighting (`scale_pos_weight = 2.7686`). | State-of-the-art accuracy on tabular data with built-in handling of non-linear interactions. Native `scale_pos_weight` penalizes minority class errors directly during tree splitting without synthetic point generation. |
| **Class Imbalance Benchmarking** | imbalanced-learn | >= 0.11.0 | Benchmark implementation of SMOTE (Synthetic Minority Over-sampling Technique) combined with scikit-learn pipelines. | Enabled rigorous empirical comparison between synthetic oversampling and algorithmic cost-sensitive weighting, validating that algorithmic weighting achieved higher ROC-AUC without synthetic boundary noise. |
| **Explainable AI (XAI)** | SHAP | >= 0.42.0 | TreeExplainer for global feature attribution (beeswarm summary) and individual customer explanations (waterfall plots). | Axiomatically grounded in cooperative game theory (Shapley values); `TreeExplainer` computes exact local and global attributions in polynomial time, unlike heuristic feature importances or model-agnostic approximations. |
| **Visualization & Reporting** | Matplotlib & Seaborn | >= 3.7.0 / >= 0.12.0 | Generation of high-resolution (180 DPI) figures for EDA distributions, ROC curves, PR curves, confusion matrices, and financial ROI curves. | Granular graphical control and reliable programmatic rendering to static files (`reports/figures/`) without external browser or headless rendering dependencies. |
| **Web Application** | Streamlit | >= 1.28.0 | Interactive web interface (`app/app.py`) featuring customer profile presets, manual input controls, dynamic threshold slider, and retention recommendations. | Rapid Python-native UI generation without requiring separate frontend frameworks or REST API boilerplate; allows business stakeholders to interact directly with the trained model pipeline. |
| **Model Serialization** | Joblib | >= 1.3.0 | Persistent serialization and deserialization of fitted transformer and model pipelines (`models/*.pkl`). | Optimized for pipelines containing large NumPy arrays and scikit-learn estimators, providing faster load times and lower disk footprint than standard Python `pickle`. |
| **Development & Experimentation** | Jupyter Notebook | >= 1.0.0 | Nine structured phase notebooks (`notebooks/01_` through `notebooks/09_`) for step-by-step exploratory analysis, experimentation, and documentation. | Interactive code and markdown execution allows transparent tracking of experimental iterations, diagnostic outputs, and statistical conclusions. |

---

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original raw dataset (7,043 rows, 21 columns)
│   └── processed/
│       └── cleaned_churn_data.csv                 # Cleaned dataset (0 missing, numeric charges, binary churn)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb                # Phase 1: Exploration, schema profiling, hidden null detection
│   ├── 02_data_cleaning.ipynb                     # Phase 2: TotalCharges conversion, category standardization
│   ├── 03_eda.ipynb                               # Phase 3: 12 associative statistical plots & distribution checks
│   ├── 04_feature_engineering.ipynb               # Phase 4: 9 domain features created without lookahead bias
│   ├── 05_baseline_model.ipynb                    # Phase 5: LogReg, Decision Tree, Random Forest, XGBoost baselines
│   ├── 06_imbalance_handling.ipynb                # Phase 6: Class weights, scale_pos_weight, SMOTE comparison
│   ├── 07_model_comparison.ipynb                  # Phase 7: 5-Fold Stratified CV tuning & champion selection
│   ├── 08_threshold_analysis.ipynb                # Phase 8: Precision-Recall trade-off & financial ROI simulation
│   └── 09_shap_explainability.ipynb               # Phase 9: Global beeswarm & local waterfall interpretability
│
├── src/
│   ├── __init__.py                                # Package initializer
│   ├── preprocessing.py                           # Clean data & I/O pipelines
│   ├── feature_engineering.py                     # Feature derivation & ColumnTransformer preprocessor
│   ├── train.py                                   # Baseline, cost-sensitive, and tuned model constructors
│   ├── evaluate.py                                # Metrics, model comparison, threshold & ROI simulators
│   └── explain.py                                 # SHAP value computation, beeswarm & waterfall plotters
│
├── models/
│   ├── best_model.pkl                             # Production Champion: Tuned XGBoost Pipeline (0.8453 ROC-AUC)
│   ├── random_forest.pkl                          # Tuned Random Forest Pipeline (0.6382 F1-Score)
│   └── logistic_regression.pkl                    # Tuned Class-Weighted Logistic Regression (0.8419 ROC-AUC)
│
├── reports/
│   ├── figures/
│   │   ├── churn_distribution.png                 # Target variable distribution & imbalance ratio
│   │   ├── contract_vs_churn.png                  # Churn rate across contract commitments
│   │   ├── tenure_vs_churn.png                    # Customer lifetime attrition dynamics
│   │   ├── confusion_matrix.png                   # 4-panel diagnostic confusion matrices
│   │   ├── roc_curve.png                          # Baseline Receiver Operating Characteristic curves
│   │   ├── precision_recall_curve.png             # Baseline Precision-Recall curves
│   │   ├── imbalance_comparison.png               # Recall surge comparison across resampling techniques
│   │   ├── tuned_model_comparison.png             # Performance profile across tuned models
│   │   ├── threshold_analysis.png                 # Precision/Recall & Net Profit curves by threshold
│   │   ├── shap_summary.png                       # Global SHAP beeswarm feature importance
│   │   └── shap_waterfall.png                     # Local customer explanation waterfall case study
│   └── model_results.csv                          # Comprehensive benchmark table tracking 12 model iterations
│
├── app/
│   ├── app.py                                     # Interactive Streamlit Web Application
│   └── utils.py                                   # Model inference, pipeline transformation & recommendation logic
│
├── requirements.txt                               # Project dependencies
├── README.md                                      # Project overview and documentation
├── .gitignore                                     # Git ignore patterns
└── LICENSE                                        # MIT License
```

---

## Model Performance Benchmark

Evaluated on the held-out test set (N = 1,409 customers):

| Rank | Model Architecture | Strategy | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Tuned XGBoost** | `scale_pos_weight = 2.77` | 74.59% | 51.36% | **80.75%** | 0.6279 | **0.8453** | **0.6599** |
| 2 | **Tuned Random Forest** | `class_weight = 'balanced'` | **76.58%** | **54.09%** | 77.81% | **0.6382** | 0.8444 | 0.6545 |
| 3 | **Tuned Logistic Regression** | `class_weight = 'balanced'` | 74.17% | 50.85% | 80.21% | 0.6224 | 0.8419 | 0.6418 |
| 4 | Logistic Regression (SMOTE) | SMOTE Resampling | 73.81% | 50.42% | 79.68% | 0.6176 | 0.8408 | 0.6416 |
| 5 | Random Forest (Baseline) | Default Threshold (0.50) | 79.91% | 65.32% | 51.87% | 0.5782 | 0.8275 | 0.6285 |

---

## Business and Explainability Insights

1. **Contract Type Association**: Month-to-month contracts exhibit the strongest positive association with churn risk (mean SHAP value +0.7851). Migration incentives toward annual commitments offer substantial retention protection.
2. **Tenure Dynamics**: Customer churn concentrates heavily in early tenure intervals (median churned tenure is 10 months versus 38 months for retained customers). Onboarding touchpoints within the first 90 days address early departure risk.
3. **Protective Add-ons**: Subscriptions to Tech Support and Online Security are associated with lower churn rates (approximately 15% versus 31% for accounts without these services).
4. **Optimal Decision Threshold Analysis**:
   - At threshold tau = 0.50, the model achieves the maximum projected net return ($18,600 profit, 105.4% ROI on the test cohort).
   - In resource-constrained environments, raising the threshold to tau = 0.65 reduces outreach costs by 33% while preserving high efficiency (143.7% ROI).

---

## Quick Start Guide

### 1. Environment Setup
```bash
# Clone or navigate to the repository
cd customer-churn-prediction

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser to evaluate customer profiles and explore decision cutoffs.

### 3. Run Analytical Notebooks
```bash
jupyter notebook notebooks/
```
Execute the 9 modular notebooks sequentially to review data preparation, baseline benchmarks, hyperparameter tuning, and SHAP explainability.
