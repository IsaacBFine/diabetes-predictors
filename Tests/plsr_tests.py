# --- UNIT AND INTEGRATION TESTS FOR PLSR WORKFLOW ---

import pytest
import numpy as np
import pandas as pd
import random
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

random.seed(67)
np.random.seed(67)

URL = "https://raw.githubusercontent.com/IsaacBFine/diabetes-predictors/refs/heads/main/Data/clean/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

# --- FIXTURES ---
# Shared data preparation so every test uses the same split and scaling.

@pytest.fixture(scope="module")
def raw_data():
    """Load the full dataset from the project's clean CSV."""
    return pd.read_csv(URL)

@pytest.fixture(scope="module")
def X_train(raw_data):
    """Feature matrix training split (80%), stratified to preserve 50/50 class balance."""
    X = raw_data.drop(columns=["Diabetes_binary"])
    Y = raw_data["Diabetes_binary"].astype(int)
    X_train, _, _, _ = train_test_split(X, Y, test_size=0.2, random_state=67, stratify=Y)
    return X_train

@pytest.fixture(scope="module")
def Y_train(raw_data):
    """Target vector training split (80%), stratified to preserve 50/50 class balance."""
    X = raw_data.drop(columns=["Diabetes_binary"])
    Y = raw_data["Diabetes_binary"].astype(int)
    _, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=67, stratify=Y)
    return Y_train

@pytest.fixture(scope="module")
def X_train_scaled(X_train):
    """StandardScaler applied only to training data to prevent leakage into test set."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train)


# TEST 1 — STANDARDIZATION
def test_scaled_data_has_correct_shape(X_train_scaled, X_train):
    """Tests that StandardScaler does not change the number of rows or predictors."""
    assert X_train_scaled.shape == X_train.shape


# TEST 2 — Q²Y COMPONENT SEARCH
def test_q2y_scores_generated_and_valid(X_train_scaled, Y_train):
    """
    Tests that the Q²Y component-selection loop runs correctly
    and returns one valid score for each tested component.
    """
    q2y_scores = []
    components = range(1, 22)

    for n in components:
        pls = PLSRegression(n_components=n)
        scores = cross_val_score(pls, X_train_scaled, Y_train, cv=5, scoring="r2")
        q2y_scores.append(scores.mean())

    assert len(q2y_scores) == len(list(components))
    assert np.all(np.isfinite(q2y_scores))


# TEST 3 — BEST COMPONENT SELECTION
def test_best_component_is_selected(X_train_scaled, Y_train):
    """
    Tests that the code successfully identifies the component number
    with the highest Q²Y score.
    """
    q2y_scores = []
    components = range(1, 22)

    for n in components:
        pls = PLSRegression(n_components=n)
        scores = cross_val_score(pls, X_train_scaled, Y_train, cv=5, scoring="r2")
        q2y_scores.append(scores.mean())

    best_n = list(components)[np.argmax(q2y_scores)]

    assert best_n in components


# TEST 4 — SIX-COMPONENT MODEL FOR COMPONENT 5 VS 6 PLOT
def test_six_component_model_generates_loadings(X_train_scaled, Y_train):
    """
    Tests that the 6-component PLSR model used to examine
    Components 5 and 6 fits successfully and produces loadings.
    """
    pls = PLSRegression(n_components=6)
    pls.fit(X_train_scaled, Y_train)

    assert pls.x_loadings_.shape[1] == 6
    assert pls.y_loadings_.shape[1] == 6


# TEST 5 — FINAL FOUR-COMPONENT MODEL
def test_four_component_model_generates_loadings(X_train_scaled, Y_train):
    """
    Tests that the final 4-component PLSR model fits successfully
    and produces the loadings used for interpretation.
    """
    pls = PLSRegression(n_components=4)
    pls.fit(X_train_scaled, Y_train)

    assert pls.x_loadings_.shape[1] == 4
    assert pls.y_loadings_.shape[1] == 4


# TEST 6 — LOADINGS DATAFRAME STRUCTURE
def test_loadings_dataframe_has_expected_rows(X_train_scaled, Y_train, X_train):
    """
    Tests that the X loading matrix has one row for each predictor
    and four columns for the four retained components.
    """
    pls = PLSRegression(n_components=4)
    pls.fit(X_train_scaled, Y_train)

    x_loadings = pd.DataFrame(
        pls.x_loadings_,
        index=X_train.columns,
        columns=[f"Component {i}" for i in range(1, 5)]
    )

    assert x_loadings.shape == (X_train.shape[1], 4)


# TEST 7 — FULL PLSR WORKFLOW INTEGRATION TEST
def test_full_plsr_workflow_runs(X_train_scaled, Y_train, X_train):
    """
    Tests that the full workflow runs from Q²Y component search
    through final 4-component model fitting and loading extraction.
    """
    q2y_scores = []
    components = range(1, 22)

    for n in components:
        pls = PLSRegression(n_components=n)
        scores = cross_val_score(pls, X_train_scaled, Y_train, cv=5, scoring="r2")
        q2y_scores.append(scores.mean())

    best_n = list(components)[np.argmax(q2y_scores)]

    pls_final = PLSRegression(n_components=4)
    pls_final.fit(X_train_scaled, Y_train)

    x_loadings = pd.DataFrame(
        pls_final.x_loadings_,
        index=X_train.columns,
        columns=[f"Component {i}" for i in range(1, 5)]
    )

    assert best_n in components
    assert x_loadings.shape[1] == 4
