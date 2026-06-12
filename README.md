# diabetes-predictors
Final Project for "BioE 175: Data Driven Models and Machine Learning". Building an explanatory model of diabetes predictors.

Using the CDC's 2015 Behavioral Risk Factor Surveillance System (BRFSS) dataset, we applied LASSO regression, Partial Least Squares Regression (PLSR), and feature-engineered LASSO models to identify important predictors and examine interactions between variables.

The primary goals of the project were to:

- Identify the strongest independent predictors of diabetes.
- Examine relationships among correlated predictors.
- Determine whether combinations of indicators provide additional predictive information beyond individual variables.

**Repository Structure**

├── Data/
│   └── clean/
│       └── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│
├── notebooks/
│   ├── lasso_analysis.ipynb
│   ├── plsr_analysis.ipynb
│   └── feature_engineering.ipynb
│
├── tests/
│   ├── test_lasso.py
│   └── test_plsr.py
│
├── Final_Report.pdf
└── README.md

