"""
Unit and Integration Tests for the LASSO Diabetes Prediction Pipeline

To run:
  pip install pytest
  pytest Tests/LASSO_tests.py -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

URL = "https://raw.githubusercontent.com/IsaacBFine/diabetes-predictors/refs/heads/main/Data/clean/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

# --- FIXTURES ---
# Fixtures load and prepare shared data once so every test uses
# the same split and scaling, keeping results consistent across runs.

@pytest.fixture(scope="module")
def raw_data():
    """Load the full dataset from the project's clean CSV."""
    return pd.read_csv(URL)

@pytest.fixture(scope="module")
def x(raw_data):
    """Feature matrix (all columns except the target)."""
    return raw_data.drop(columns=["Diabetes_binary"])

@pytest.fixture(scope="module")
def y(raw_data):
    """Binary target vector (0 = no diabetes, 1 = diabetes)."""
    return raw_data["Diabetes_binary"].astype(int)

@pytest.fixture(scope="module")
def split_data(x, y):
    """80/20 stratified train-test split with fixed random state for reproducibility."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=67, stratify=y
    )
    return x_train, x_test, y_train, y_test

@pytest.fixture(scope="module")
def x_train(split_data):
    return split_data[0]

@pytest.fixture(scope="module")
def x_test(split_data):
    return split_data[1]

@pytest.fixture(scope="module")
def y_train(split_data):
    return split_data[2]

@pytest.fixture(scope="module")
def y_test(split_data):
    return split_data[3]

@pytest.fixture(scope="module")
def x_train_scaled(x_train):
    """StandardScaler fit only on training data to prevent data leakage."""
    scaler = StandardScaler()
    return scaler.fit_transform(x_train)

@pytest.fixture(scope="module")
def x_test_scaled(x_train, x_test):
    """Test data scaled using the same scaler fit on training data."""
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler.transform(x_test)

@pytest.fixture(scope="module")
def sample_df(raw_data):
    """Small 500-row sample for fast unit tests that don't need full data."""
    return raw_data.sample(500, random_state=67).reset_index(drop=True)

@pytest.fixture(scope="module")
def final_model(x_train_scaled, y_train):
    """Best LASSO model trained with C=0.0022 (selected via cross-validation)."""
    model = LogisticRegression(
        penalty='l1', solver='liblinear', C=0.0022, max_iter=1000, random_state=67
    )
    model.fit(x_train_scaled, y_train)
    return model


# --- UNIT TEST: TRAIN/TEST SPLIT ---

def test_train_test_split(x_train, x_test, y_train, y_test, x):
    """Verify the 80/20 split size and that both halves preserve the 50/50 class balance."""

    # tests to see if data was split correctly (80 training/20 test)
    assert abs(len(x_train) - len(x) * 0.8) <= 1, \
        f"Expected {round(len(x)*0.8)} training samples, got {len(x_train)}"
    assert abs(len(x_test) - len(x) * 0.2) <= 1, \
        f"Expected {round(len(x)*0.2)} test samples, got {len(x_test)}"

    # tests to see if each split maintained a balanced 50/50 of diabetes to non-diabetes
    assert abs(y_train.mean() - 0.5) < 0.1, \
        f"Training set class ratio is {y_train.mean():.2f}, expected ~0.5"
    assert abs(y_test.mean() - 0.5) < 0.1, \
        f"Test set class ratio is {y_test.mean():.2f}, expected ~0.5"


# --- UNIT TEST: CROSS VALIDATION SCORES ARE VALID ---

def test_cross_validation_scores_are_valid(x_train_scaled, y_train):
    """Verify that 5-fold CV AUC and accuracy scores all fall in [0, 1]."""

    # Use the best C found from hyperparameter search
    best_C = 0.0022
    m = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=best_C,
        max_iter=1000,
        random_state=67
    )

    auc = cross_val_score(m, x_train_scaled, y_train, cv=5, scoring='roc_auc')
    acc = cross_val_score(m, x_train_scaled, y_train, cv=5, scoring='accuracy')

    # tests if AUC scores are in valid range (0 to 1)
    assert auc.min() >= 0.0, f"AUC score below 0: {auc.min()}"
    assert auc.max() <= 1.0, f"AUC score above 1: {auc.max()}"

    # tests if accuracy scores are in valid range (0 to 1)
    assert acc.min() >= 0.0, f"Accuracy score below 0: {acc.min()}"
    assert acc.max() <= 1.0, f"Accuracy score above 1: {acc.max()}"


# --- UNIT TEST: CROSS VALIDATION C VALUE SEARCH ---

def test_cross_validation_c_search(sample_df):
    """
    Checks that the C value search loop works correctly:
      - np.logspace(-4, -1, 10) produces 10 values in the right range.
      - The loop actually selects a best_C (doesn't stay None).
      - The best_C is one of the values from the search range.
    """
    x = sample_df.drop(columns=['Diabetes_binary'])
    y = sample_df['Diabetes_binary'].astype(int)

    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=67, stratify=y
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    Cs = np.logspace(-4, -1, 10)

    # Correct number of C values and range
    assert len(Cs) == 10, f"Expected 10 C values, got {len(Cs)}"
    assert abs(Cs[0] - 0.0001) < 1e-6, f"First C should be ~0.0001, got {Cs[0]}"
    assert abs(Cs[-1] - 0.1)   < 1e-6, f"Last C should be ~0.1, got {Cs[-1]}"

    # Run the search loop
    best_C, best_auc = None, 0
    for C in Cs:
        m = LogisticRegression(
            penalty='l1', solver='liblinear', C=C, max_iter=1000, random_state=67
        )
        aucs = cross_val_score(m, x_train_scaled, y_train, cv=5, scoring='roc_auc')
        if aucs.mean() > best_auc:
            best_auc, best_C = aucs.mean(), C

    # A best_C was actually selected
    assert best_C is not None, "The C search loop never updated best_C."

    # best_C came from our search range
    assert best_C in Cs, f"best_C ({best_C}) is not one of the searched values."


# --- UNIT TEST: MODEL OUTPUT ---

def test_model_output(final_model, x_test_scaled):
    """Verify that the fitted LASSO model zeroes out some coefficients and outputs valid probabilities."""

    probs = final_model.predict_proba(x_test_scaled)

    # tests if LASSO zeroed out at least one coefficient
    n_zero = (final_model.coef_[0] == 0).sum()
    assert n_zero > 0, \
        "Lasso should zero out at least some coefficients, but all are non-zero."

    # tests if predicted probabilities are between 0 and 1
    assert probs.min() >= 0.0, "Some predicted probabilities are negative."
    assert probs.max() <= 1.0, "Some predicted probabilities exceed 1."


# --- INTEGRATION TEST: FULL AUC AND ACCURACY PIPELINE ---
# Tests that all components work correctly together

def test_full_pipeline(x, y):
    """End-to-end test: split, scale, C-search, fit, and evaluate both AUC and accuracy pipelines."""

    # split and scale data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=67, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled  = scaler.transform(x_test)

    Cs = np.logspace(-4, -1, 10)

    # Cross-validate using AUC and fitting model
    best_C_auc, best_auc = None, 0
    for C in Cs:
        m = LogisticRegression(
            penalty='l1', solver='liblinear', C=C, max_iter=1000, random_state=67
        )
        aucs = cross_val_score(m, x_train_scaled, y_train, cv=5, scoring='roc_auc')
        if aucs.mean() > best_auc:
            best_auc, best_C_auc = aucs.mean(), C

    model_auc = LogisticRegression(
        penalty='l1', solver='liblinear', C=best_C_auc, max_iter=1000, random_state=67
    )
    model_auc.fit(x_train_scaled, y_train)
    probs_auc = model_auc.predict_proba(x_test_scaled)[:, 1]
    test_auc  = roc_auc_score(y_test, probs_auc)

    # Cross-validate using accuracy and fitting model
    best_C_acc, best_acc = None, 0
    for C in Cs:
        m = LogisticRegression(
            penalty='l1', solver='liblinear', C=C, max_iter=1000, random_state=67
        )
        accs = cross_val_score(m, x_train_scaled, y_train, cv=5, scoring='accuracy')
        if accs.mean() > best_acc:
            best_acc, best_C_acc = accs.mean(), C

    model_acc = LogisticRegression(
        penalty='l1', solver='liblinear', C=best_C_acc, max_iter=1000, random_state=67
    )
    model_acc.fit(x_train_scaled, y_train)
    preds_acc = model_acc.predict(x_test_scaled)
    test_acc  = (preds_acc == y_test).mean()

    # tests whether both pipelines selected a C value
    assert best_C_auc is not None, "AUC pipeline never selected a best_C."
    assert best_C_acc is not None, "Accuracy pipeline never selected a best_C."

    # tests whether AUC and accuracy of models in a valid range (0 to 1)
    assert 0.0 <= test_auc <= 1.0, \
        f"Test AUC {test_auc:.3f} is outside the valid range [0, 1]."
    assert 0.0 <= test_acc <= 1.0, \
        f"Test accuracy {test_acc:.3f} is outside the valid range [0, 1]."
