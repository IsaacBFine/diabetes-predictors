"""
Unit and Integration Tests

to run...
  pip install pytest
  pytest test_diabetes_model_simple.py -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

@pytest.fixture

# UNIT TEST — TRAIN/TEST SPLIT

def test_train_test_split(x_train,x_test,y_train,y_test):

    # tests to see if data was split correctly (80 training/20 test)
    assert abs(len(x_train)-len(x)*0.8) <= 1, \
        f"Expected {round(len(x)*0.8)} training samples, got {len(x_train)}"
    assert abs(len(x_test)-len(x)*0.2) <= 1, \
        f"Expected {round(len(x)*0.2)} test samples, got {len(x_test)}"

    # tests to see if each split maintained a balance 50/50 of diabates to nondiabetes
    assert abs(y_train.mean() - 0.5) < 0.1, \
        f"Training set class ratio is {y_train.mean():.2f}, expected ~0.5"
    assert abs(y_test.mean() - 0.5) < 0.1, \
        f"Test set class ratio is {y_test.mean():.2f}, expected ~0.5"


# UNIT TEST — CROSS VALIDATION SCORES ARE VALID

def test_cross_validation_scores_are_valid(x_train_scaled,y_test):

    m = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=C,
        max_iter=1000,
        random_state=67
    )

    auc = cross_val_score(m, x_train_scaled, y_train, cv=5, scoring='roc_auc')
    acc = cross_val_score(m, x_train_scaled, y_train, cv=5, scoring='accuracy')

    # tests if AUC scores are in valid range (0 to 1)
    assert auc.min() >= 0.0, f"AUC score below 0: {auc_scores.min()}"
    assert auc.max() <= 1.0, f"AUC score above 1: {auc_scores.max()}"

    # tests if accuracy scores are in valid range (0 to 1)
    assert acc.min() >= 0.0, f"Accuracy score below 0: {acc_scores.min()}"
    assert acc.max() <= 1.0, f"Accuracy score above 1: {acc_scores.max()}"


# UNIT TEST 3 — CROSS VALIDATION C VALUE SEARCH

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


# UNIT TEST — MODEL OUTPUT

def test_model_output(final_model,x_test_scaled):

    probs = final_model.predict_proba(x_test_scaled)

    # tests if LASSO zeroed out at least one coefficient
    n_zero = (final_model.coef_[0] == 0).sum()
    assert n_zero > 0, \
        "Lasso should zero out at least some coefficients, but all are non-zero."

    # tests if predicted probabilities are between 0 and 1
    assert probs.min() >= 0.0, "Some predicted probabilities are negative."
    assert probs.max() <= 1.0, "Some predicted probabilities exceed 1."


# INTEGRATION TEST — FULL AUC AND ACCURACY PIPELINE
# Tests that all components work correctly together

def test_full_pipeline(x,y):

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
