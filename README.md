# Diabetes Predictors

Final Project for **BioE 175: Data-Driven Models and Machine Learning**

This project builds interpretable predictive models for diabetes using the CDC's 2015 Behavioral Risk Factor Surveillance System (BRFSS) dataset. Rather than pursuing black-box accuracy, the goal is to identify which health, behavioral, and socioeconomic factors most strongly predict diabetes risk — and to understand *why*.

---

## Project Overview

Using a cleaned, balanced dataset of 70,692 respondents (50% diabetic / 50% non-diabetic), we apply two primary modeling approaches:

- **LASSO Logistic Regression** — L1-regularized logistic regression that performs simultaneous feature selection and classification, revealing which of 21 health indicators matter most.
- **Partial Least Squares Regression (PLSR)** — A dimensionality reduction method that uncovers multivariate structure and variable interactions in the feature space.

We also explore two feature engineering strategies:
- **Biological interactions** — Interaction terms grounded in physiological relationships (e.g., HighBP × HighChol, BMI × Age).
- **Socioeconomic status (SES) interactions** — Terms that test whether socioeconomic factors (income, education, healthcare access) modify the relationship between health indicators and diabetes risk.

### Key Findings

| Model | Cross-Validated AUC |
|---|---|
| Baseline LASSO | ~0.8247 |
| Bio-Engineered LASSO | ~0.8275 |
| SES-Engineered LASSO | ~0.8250 |

Top risk factors identified: General Health Status, BMI, Age, High Blood Pressure, High Cholesterol.  
Protective factors identified: Higher Income, Heavy Alcohol Consumption (paradoxical), Higher Education.

---

## Project Structure

```
diabetes-predictors/
│
├── Code/                            # Standalone Python scripts (final models)
│   ├── LASSO_Final_Model.py            # Baseline LASSO with hyperparameter tuning
│   ├── PLSR_Final_Model.py             # PLSR component analysis
|   ├── bio_feature_engineering.py      # LASSO with biological interaction terms
│   └── ses_feature_engineering.py      # LASSO with socioeconomic interaction terms
│
├── Data/
│   ├── clean/                                                         # Preprocessed datasets ready for modeling
│   │   ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv  # Primary (balanced binary)
│   │   ├── diabetes_binary_health_indicators_BRFSS2015.csv            # Imbalanced binary
│   │   └── diabetes_012_health_indicators_BRFSS2015.csv               # 3-class version
│   └── raw/
│       └── raw_dataset.md      # Instructions to download the original BRFSS 2015 data
│
├── Notebooks/                              # Jupyter notebooks for exploration and analysis
│   ├── data-preprocessing-notebook.ipynb       # Cleans raw BRFSS data → 22-feature dataset
│   ├── LASSO_Data_Analysis.ipynb               # LASSO hyperparameter search and evaluation
│   ├── feature_engineering_notebook.ipynb      # Bio and SES feature engineering study
│   └── plsr_notebook.ipynb                     # PLSR component analysis
│
├── Results/                    # Output figures from all analyses
│   ├── lasso/                     # Hyperparameter curves and coefficient plots
│   ├── feature-engineering/       # Correlation matrix, VIF, and model comparisons
│   └── plsr/                      # Component selection and loading plots
│
├── Tests/                          # Unit and integration tests
│   ├── LASSO_tests.py                 # Tests for train-test split and CV validity
│   ├── plsr_tests.py                  # Tests for Q²Y range and component behavior
│   └── test_feature_engineering.py    # 33 tests covering the full feature engineering pipeline
│
├── MIT License                    # MIT license for the project
├── README                         # Documentation outlining project and instruction how to install dependencies
├── Dockerfile                     # Container definition for reproducible environment
└── pyproject.toml                 # Python dependencies and project metadata
```

---

## Dataset

**Source:** [CDC BRFSS 2015](https://www.cdc.gov/brfss/) via Kaggle  
**Original size:** 441,456 survey responses, 330 features  
**After preprocessing:** 70,692 rows, 22 columns (50/50 class balance)

| Feature | Type | Description |
|---|---|---|
| Diabetes_binary | Binary (target) | 0 = no diabetes, 1 = pre-diabetes or diabetes |
| HighBP | Binary | Ever told you have high blood pressure |
| HighChol | Binary | High cholesterol diagnosis |
| CholCheck | Binary | Cholesterol check in past 5 years |
| BMI | Continuous | Body Mass Index |
| Smoker | Binary | Smoked at least 100 cigarettes lifetime |
| Stroke | Binary | Ever had a stroke |
| HeartDiseaseorAttack | Binary | Coronary heart disease or heart attack |
| PhysActivity | Binary | Physical activity in past 30 days |
| Fruits | Binary | Consumes fruit 1+ times per day |
| Veggies | Binary | Consumes vegetables 1+ times per day |
| HvyAlcoholConsump | Binary | Heavy drinker (men >14 drinks/wk, women >7) |
| AnyHealthcare | Binary | Has any healthcare coverage |
| NoDocbcCost | Binary | Could not see doctor due to cost in past year |
| GenHlth | Ordinal (1–5) | Self-rated general health (1=excellent, 5=poor) |
| MentHlth | Integer (0–30) | Days of poor mental health in past 30 days |
| PhysHlth | Integer (0–30) | Days of poor physical health in past 30 days |
| DiffWalk | Binary | Serious difficulty walking or climbing stairs |
| Sex | Binary | Biological sex (0=female, 1=male) |
| Age | Ordinal (1–13) | Age category (1=18–24, 13=80+) |
| Education | Ordinal (1–6) | Highest education level completed |
| Income | Ordinal (1–8) | Annual household income bracket |

The preprocessing notebook (`data-preprocessing-notebook.ipynb`) handles all cleaning steps, including feature selection from 330 → 22 variables and creating the balanced 50/50 split.

---

## Installation

### Option 1: Local Python Environment 

**Requirements:** Python 3.10 or higher

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-predictors.git
   cd diabetes-predictors
   ```

2. Install dependencies using `pip`:
   ```bash
   pip install .
   ```
   Or install in editable/development mode:
   ```bash
   pip install -e .
   ```

   This installs all required packages defined in `pyproject.toml`, including:
   `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `statsmodels`, `scipy`, `xgboost`, `jupyter`

3. Verify the install by running the tests:
   ```bash
   python -m pytest Tests/
   ```

### Option 2: Docker 

1. Build the Docker image:
   ```bash
   docker build -t diabetes-predictors .
   ```

2. Run an interactive container:
   ```bash
   docker run -it --rm -v $(pwd):/app diabetes-predictors bash
   ```

---

## Running the Code

### Jupyter Notebooks

Launch JupyterLab from the project root or open Google Colab from a browser: 
```bash
jupyter lab
```
https://colab.research.google.com/


Run notebooks in this order:

| Step | Notebook | Purpose |
|---|---|---|
| 1 | `Notebooks/data-preprocessing-notebook.ipynb` | Clean raw data (skip if using provided clean data) |
| 2 | `Notebooks/LASSO_Data_Analysis.ipynb` | Baseline LASSO model, hyperparameter tuning |
| 3 | `Notebooks/feature_engineering_notebook.ipynb` | Bio and SES feature engineering study |
| 4 | `Notebooks/plsr_notebook.ipynb` | PLSR component analysis |

**Note:** The clean datasets are already included in `Data/clean/`, so Step 1 is optional unless you want to re-run preprocessing from raw data.

### Python Scripts (Final Models)

Each script in `Code/` is self-contained and can be run directly from the project root:

```bash
# Run the baseline LASSO model
python Code/LASSO_Final_Model.py

# Run LASSO with biological feature engineering
python Code/bio_feature_engineering.py

# Run LASSO with socioeconomic feature engineering
python Code/ses_feature_engineering.py
```

Each script loads the clean dataset, trains the model with 5-fold stratified cross-validation, prints performance metrics, and saves result figures to the `Results/` directory.

### Tests
The tests may be ran all at once or individually:

```bash
# Run all tests
python -m pytest Tests/

# Run a specific test file
python -m pytest Tests/test_feature_engineering.py -v
```

---

## Methods Summary

### LASSO Logistic Regression

LASSO (Least Absolute Shrinkage and Selection Operator) applies an L1 penalty to logistic regression coefficients, shrinking uninformative feature weights to exactly zero. This makes it ideal for simultaneous feature selection and classification.

- **Hyperparameter:** `C` (inverse regularization strength) tuned over 10 log-spaced values from 10⁻⁴ to 10⁻¹
- **Solver:** `liblinear` (required for L1 penalty)
- **Cross-validation:** 5-fold stratified, evaluated on ROC-AUC
- **Scaling:** `StandardScaler` fit on training fold only to prevent data leakage

### Partial Least Squares Regression (PLSR)

PLSR finds latent components that simultaneously maximize covariance between predictors and the diabetes outcome. It is particularly useful when features are correlated.

- **Component selection:** Q²Y metric via 5-fold cross-validation (elbow method)
- **Optimal components:** 4 (Components 5–6 showed negligible diabetes loading)

### Feature Engineering

Interaction terms multiply two features together, capturing synergistic or modifier effects. For example, `HighBP × HighChol` tests whether having both conditions is worse than the sum of each individually.

- **Biological interactions:** 18 pairs grounded in clinical and physiological rationale
- **SES interactions:** 18 pairs testing whether socioeconomic context modifies health-to-diabetes pathways
- Both sets expand the feature space from 21 → 39 features before applying LASSO for selection

---

## Results

All output figures are saved in `Results/`:

- `Results/lasso/` — Cross-validation curves, feature coefficient plots
- `Results/feature-engineering/` — Correlation matrix, VIF, model comparison bar charts
- `Results/plsr/` — Component selection plot, loading plots (1D and 2D)

## Authors

Isaac Fine
Paloma Duvergne
Eddie Hahm
Gloria Chen
Chibudom Ofoegbu
