import unittest
import numpy as np
import pandas as pd
import random
random.seed(67)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
 
# Helpers — replicated from the notebook so tests are self-contained ---------------

INTERACTION_PAIRS = [
    ("HighBP",      "HighChol"),
    ("HighBP",      "HeartDiseaseorAttack"),
    ("HighChol",    "HeartDiseaseorAttack"),
    ("HighBP",      "Stroke"),
    ("BMI",         "PhysActivity"),
    ("BMI",         "Age"),
    ("Smoker",      "HvyAlcoholConsump"),
    ("Fruits",      "Veggies"),
    ("GenHlth",     "PhysHlth"),
    ("GenHlth",     "MentHlth"),
    ("PhysHlth",    "DiffWalk"),
    ("GenHlth",     "DiffWalk"),
    ("Income",      "Education"),
    ("Income",      "AnyHealthcare"),
    ("NoDocbcCost", "AnyHealthcare"),
    ("Age",         "HeartDiseaseorAttack"),
    ("Age",         "Stroke"),
    ("Age",         "DiffWalk"),
]

SES_INTERACTION_PAIRS_RAW = [
    ("Income",        "GenHlth"),
    ("Income",        "PhysHlth"),
    ("Income",        "DiffWalk"),
    ("Income",        "BMI"),
    ("Income",        "HighBP"),
    ("Income",        "HeartDiseaseorAttack"),
    ("Education",     "GenHlth"),
    ("Education",     "BMI"),
    ("Education",     "PhysActivity"),
    ("AnyHealthcare", "GenHlth"),
    ("AnyHealthcare", "PhysHlth"),
    ("AnyHealthcare", "HighBP"),
    ("NoDocbcCost",   "GenHlth"),
    ("NoDocbcCost",   "PhysHlth"),
    ("Education",     "HighBP"),
    ("Income",        "Education"),
    ("Income",        "AnyHealthcare"),
    ("NoDocbcCost",   "AnyHealthcare"),
]

# deduplicate (same logic as notebook)
def _dedup_pairs(pairs):
    seen, clean = set(), []
    for a, b in pairs:
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            clean.append((a, b))
    return clean

SES_INTERACTION_PAIRS = _dedup_pairs(SES_INTERACTION_PAIRS_RAW)

EXPECTED_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]


def build_interaction_features(X_raw, pairs):
    """Adds interaction columns (product) to a copy of X_raw."""
    X_eng = X_raw.copy()
    for a, b in pairs:
        X_eng[f"{a} x {b}"] = X_raw[a] * X_raw[b]
    return X_eng


def cv_auc(model, X, y, cv):
    """5-fold CV AUC-ROC (replicated from notebook)."""
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], prob))
    return np.mean(aucs), np.std(aucs)


def make_synthetic_df(n=200, seed=42):
    """
    Small synthetic dataset with the same column schema as the real data.
    Binary columns ~ Bernoulli(0.3), continuous columns ~ N(0,1) clipped.
    Target is a noisy function of a few features.
    """
    rng = np.random.default_rng(seed)

    binary_cols = [
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
    ]
    continuous_cols = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]

    data = {}
    for col in binary_cols:
        data[col] = rng.integers(0, 2, size=n).astype(float)
    for col in continuous_cols:
        data[col] = np.clip(rng.normal(3, 1.5, size=n), 1, 5)

    df = pd.DataFrame(data)

    # simple target correlated with a few predictors
    logit = (
        0.5 * df["HighBP"]
        + 0.4 * df["BMI"]
        - 0.3 * df["PhysActivity"]
        + rng.normal(0, 0.5, n)
    )
    df["Diabetes_binary"] = (logit > logit.median()).astype(float)
    return df



# Unit Tests --------------------------

# Tests for the pair deduplication logic
class TestDeduplication(unittest.TestCase):

    def test_no_duplicates_in_ses_pairs(self):
        seen = set()
        for a, b in SES_INTERACTION_PAIRS:
            key = tuple(sorted((a, b)))
            self.assertNotIn(key, seen, f"Duplicate pair found: ({a}, {b})")
            seen.add(key)

    def test_dedup_removes_reversed_pairs(self):
        pairs = [("A", "B"), ("B", "A"), ("C", "D")]
        result = _dedup_pairs(pairs)
        self.assertEqual(len(result), 2)

    def test_dedup_preserves_order_of_first_occurrence(self):
        pairs = [("A", "B"), ("C", "D"), ("B", "A")]
        result = _dedup_pairs(pairs)
        self.assertEqual(result[0], ("A", "B"))
        self.assertEqual(result[1], ("C", "D"))


# Tests for build_interaction_features().
class TestInteractionFeatureEngineering(unittest.TestCase):

    def setUp(self):
        df = make_synthetic_df()
        self.X_raw = df.drop(columns=["Diabetes_binary"])
        self.y = df["Diabetes_binary"].values

    def test_correct_number_of_new_columns(self):
        X_eng = build_interaction_features(self.X_raw, INTERACTION_PAIRS)
        expected = self.X_raw.shape[1] + len(INTERACTION_PAIRS)
        self.assertEqual(X_eng.shape[1], expected)

    def test_interaction_column_names_correct(self):
        X_eng = build_interaction_features(self.X_raw, INTERACTION_PAIRS)
        for a, b in INTERACTION_PAIRS:
            self.assertIn(f"{a} x {b}", X_eng.columns)

    def test_interaction_values_are_products(self):
        X_eng = build_interaction_features(self.X_raw, INTERACTION_PAIRS)
        a, b = INTERACTION_PAIRS[0]
        expected = self.X_raw[a] * self.X_raw[b]
        pd.testing.assert_series_equal(
            X_eng[f"{a} x {b}"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_original_columns_unchanged(self):
        X_eng = build_interaction_features(self.X_raw, INTERACTION_PAIRS)
        for col in self.X_raw.columns:
            pd.testing.assert_series_equal(X_eng[col], self.X_raw[col])

    def test_no_nans_introduced(self):
        X_eng = build_interaction_features(self.X_raw, INTERACTION_PAIRS)
        self.assertFalse(X_eng.isnull().any().any())

    def test_row_count_preserved(self):
        X_eng = build_interaction_features(self.X_raw, INTERACTION_PAIRS)
        self.assertEqual(X_eng.shape[0], self.X_raw.shape[0])

    def test_binary_interaction_is_logical_and(self):
        """Binary × binary should equal 1 only when both are 1."""
        X_eng = build_interaction_features(self.X_raw, INTERACTION_PAIRS)
        a, b = "HighBP", "HighChol"
        result = X_eng[f"{a} x {b}"]
        expected = ((self.X_raw[a] == 1) & (self.X_raw[b] == 1)).astype(float)
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_ses_interaction_count(self):
        X_ses = build_interaction_features(self.X_raw, SES_INTERACTION_PAIRS)
        expected = self.X_raw.shape[1] + len(SES_INTERACTION_PAIRS)
        self.assertEqual(X_ses.shape[1], expected)


# Tests for StandardScaler behavior used throughout the notebook.
class TestStandardScaler(unittest.TestCase):

    def setUp(self):
        df = make_synthetic_df()
        self.X_raw = df.drop(columns=["Diabetes_binary"])

    def test_scaled_mean_near_zero(self):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_raw)
        means = np.abs(X_scaled.mean(axis=0))
        np.testing.assert_array_less(means, 1e-10)

    def test_scaled_std_near_one(self):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_raw)
        stds = X_scaled.std(axis=0)
        np.testing.assert_allclose(stds, 1.0, atol=1e-6)

    def test_shape_preserved_after_scaling(self):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_raw)
        self.assertEqual(X_scaled.shape, self.X_raw.shape)

# Tests for VIF computation.
class TestVIF(unittest.TestCase):

    def setUp(self):
        df = make_synthetic_df()
        X_raw = df.drop(columns=["Diabetes_binary"])
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X_raw)
        self.feature_names = list(X_raw.columns)

    def test_vif_returns_correct_length(self):
        vifs = [variance_inflation_factor(self.X_scaled, i)
                for i in range(len(self.feature_names))]
        self.assertEqual(len(vifs), len(self.feature_names))

    def test_vif_values_are_positive(self):
        vifs = [variance_inflation_factor(self.X_scaled, i)
                for i in range(len(self.feature_names))]
        for v in vifs:
            self.assertGreater(v, 0)

    def test_vif_dataframe_structure(self):
        vif_df = pd.DataFrame({
            "Feature": self.feature_names,
            "VIF": [variance_inflation_factor(self.X_scaled, i)
                    for i in range(len(self.feature_names))],
        })
        self.assertIn("Feature", vif_df.columns)
        self.assertIn("VIF", vif_df.columns)
        self.assertEqual(len(vif_df), len(self.feature_names))

# Tests for the cv_auc helper
class TestCvAuc(unittest.TestCase):

    def setUp(self):
        df = make_synthetic_df()
        X_raw = df.drop(columns=["Diabetes_binary"])
        self.y = df["Diabetes_binary"].values
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X_raw)
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def test_cv_auc_returns_two_values(self):
        model = LogisticRegression(C=1.0, max_iter=500)
        mean, std = cv_auc(model, self.X_scaled, self.y, self.cv)
        self.assertIsInstance(mean, float)
        self.assertIsInstance(std, float)

    def test_cv_auc_mean_in_valid_range(self):
        model = LogisticRegression(C=1.0, max_iter=500)
        mean, _ = cv_auc(model, self.X_scaled, self.y, self.cv)
        self.assertGreaterEqual(mean, 0.0)
        self.assertLessEqual(mean, 1.0)

    def test_cv_auc_std_non_negative(self):
        model = LogisticRegression(C=1.0, max_iter=500)
        _, std = cv_auc(model, self.X_scaled, self.y, self.cv)
        self.assertGreaterEqual(std, 0.0)

# Tests for the correlation matrix computation.
class TestCorrelationHeatmap(unittest.TestCase):

    def setUp(self):
        df = make_synthetic_df()
        self.X_raw = df.drop(columns=["Diabetes_binary"])

    def test_corr_matrix_is_square(self):
        corr = self.X_raw.corr()
        self.assertEqual(corr.shape[0], corr.shape[1])

    def test_corr_diagonal_is_one(self):
        corr = self.X_raw.corr()
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_corr_values_bounded(self):
        corr = self.X_raw.corr()
        self.assertTrue((corr.values >= -1.0 - 1e-10).all())
        self.assertTrue((corr.values <=  1.0 + 1e-10).all())

    def test_corr_is_symmetric(self):
        corr = self.X_raw.corr()
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)


# Integration Tests --------------

# End-to-end integration tests that exercise the full notebook pipeline on synthetic data.
class TestLassoPipelineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = make_synthetic_df(n=300, seed=0)
        cls.y = df["Diabetes_binary"].values
        cls.X_raw = df.drop(columns=["Diabetes_binary"])
        cls.feature_names = list(cls.X_raw.columns)

        scaler = StandardScaler()
        cls.X_scaled = scaler.fit_transform(cls.X_raw)

        cls.X_eng = build_interaction_features(cls.X_raw, INTERACTION_PAIRS)
        cls.X_eng_scaled = StandardScaler().fit_transform(cls.X_eng)
        cls.eng_feature_names = list(cls.X_eng.columns)
        cls.interaction_names = [f"{a} x {b}" for a, b in INTERACTION_PAIRS]

        cls.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cls.Cs = np.logspace(-3, 1, 10)  # coarse grid to keep tests fast

    def test_baseline_lasso_fits(self):
        model = LogisticRegressionCV(
            Cs=self.Cs, penalty="l1", solver="liblinear",
            cv=self.cv, max_iter=500, random_state=42, scoring="roc_auc",
        )
        model.fit(self.X_scaled, self.y)
        self.assertEqual(len(model.coef_[0]), len(self.feature_names))

    def test_engineered_lasso_has_more_features_than_baseline(self):
        self.assertGreater(len(self.eng_feature_names), len(self.feature_names))

    def test_engineered_lasso_fits(self):
        model = LogisticRegressionCV(
            Cs=self.Cs, penalty="l1", solver="liblinear",
            cv=self.cv, max_iter=500, random_state=42, scoring="roc_auc",
        )
        model.fit(self.X_eng_scaled, self.y)
        self.assertEqual(len(model.coef_[0]), len(self.eng_feature_names))

    def test_coef_series_index_matches_features(self):
        model = LogisticRegressionCV(
            Cs=self.Cs, penalty="l1", solver="liblinear",
            cv=self.cv, max_iter=500, random_state=42, scoring="roc_auc",
        )
        model.fit(self.X_eng_scaled, self.y)
        coef = pd.Series(model.coef_[0], index=self.eng_feature_names)
        self.assertEqual(list(coef.index), self.eng_feature_names)

    def test_interaction_names_are_subset_of_eng_features(self):
        for name in self.interaction_names:
            self.assertIn(name, self.eng_feature_names)

    def test_cv_auc_above_chance(self):
        """A logistic model on features correlated with target should beat 0.5."""
        model = LogisticRegression(C=0.1, penalty="l1", solver="liblinear", max_iter=500)
        mean, _ = cv_auc(model, self.X_scaled, self.y, self.cv)
        self.assertGreater(mean, 0.5)

    def test_compare_dataframe_structure(self):
        """The before-vs-after comparison DataFrame must have the expected columns."""
        lasso_base = LogisticRegressionCV(
            Cs=self.Cs, penalty="l1", solver="liblinear",
            cv=self.cv, max_iter=500, random_state=42, scoring="roc_auc",
        )
        lasso_base.fit(self.X_scaled, self.y)
        coef_base = pd.Series(lasso_base.coef_[0], index=self.feature_names)

        lasso_eng = LogisticRegressionCV(
            Cs=self.Cs, penalty="l1", solver="liblinear",
            cv=self.cv, max_iter=500, random_state=42, scoring="roc_auc",
        )
        lasso_eng.fit(self.X_eng_scaled, self.y)
        coef_eng = pd.Series(lasso_eng.coef_[0], index=self.eng_feature_names)

        compare = pd.DataFrame({
            "baseline":   coef_base,
            "engineered": coef_eng[self.feature_names],
        })
        compare["changed_sign"] = (
            (compare["baseline"] != 0) & (compare["engineered"] != 0) &
            (np.sign(compare["baseline"]) != np.sign(compare["engineered"]))
        )
        compare["dropped"] = (compare["baseline"] != 0) & (compare["engineered"] == 0)
        compare["gained"]  = (compare["baseline"] == 0) & (compare["engineered"] != 0)

        for col in ("baseline", "engineered", "changed_sign", "dropped", "gained"):
            self.assertIn(col, compare.columns)
        self.assertEqual(len(compare), len(self.feature_names))

    def test_ses_pipeline_fits(self):
        X_ses = build_interaction_features(self.X_raw, SES_INTERACTION_PAIRS)
        X_ses_scaled = StandardScaler().fit_transform(X_ses)
        model = LogisticRegressionCV(
            Cs=self.Cs, penalty="l1", solver="liblinear",
            cv=self.cv, max_iter=500, random_state=42, scoring="roc_auc",
        )
        model.fit(X_ses_scaled, self.y)
        self.assertEqual(len(model.coef_[0]), X_ses.shape[1])

    def test_income_quartile_groupby(self):
        """The income-quartile groupby used in the final visualization should work."""
        df_plot = self.X_raw.copy()
        df_plot["Diabetes"] = self.y
        df_plot["IncomeQuartile"] = pd.cut(
            df_plot["Income"], bins=[0, 2, 4, 6, 8],
            labels=["Low", "Mid-Low", "Mid-High", "High"],
        )
        grouped = df_plot.groupby(
            ["GenHlth", "IncomeQuartile"], observed=True
        )["Diabetes"].mean().reset_index()
        self.assertIn("GenHlth", grouped.columns)
        self.assertIn("IncomeQuartile", grouped.columns)
        self.assertIn("Diabetes", grouped.columns)


# Entry point -----------

if __name__ == "__main__":
    unittest.main(verbosity=2)
