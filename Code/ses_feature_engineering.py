"""
ses_feature_engineering.py

Isaac Fine

Socioeconomic (SES) interaction feature engineering for diabetes prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
random.seed(67)
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score



# 1. Load data

url = (
    "https://raw.githubusercontent.com/IsaacBFine/diabetes-predictors/"
    "refs/heads/main/Data/clean/"
    "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
)

df = pd.read_csv(url)

# Separate target from features
target        = "Diabetes_binary"
y             = df[target].values
X_raw         = df.drop(columns=[target])
feature_names = list(X_raw.columns)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")



# 2. Cross-validation setup and regularization grid
#    Same settings used across all models for a fair comparison.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
Cs = np.logspace(-3, 1, 40)   # 40 regularization strengths on a log scale



# 3. Define SES interaction pairs
#    Grouped by the socioeconomic dimension being crossed with health indicators.
#    Each tuple (A, B) creates a new column A × B in the feature matrix.

ses_interaction_pairs_raw = [
  
    # Income × health indicators
    # Does low income amplify the effect of poor health on diabetes risk?
  
    ("Income",        "GenHlth"),
    ("Income",        "PhysHlth"),           # economic stress → physical health days lost
    ("Income",        "DiffWalk"),           # low income + physical limitation
    ("Income",        "BMI"),                # is high BMI riskier in lower-income patients?
    ("Income",        "HighBP"),             # access to BP management
    ("Income",        "HeartDiseaseorAttack"),# cardiac risk moderated by financial resources

    # Education × health indicators
    # Does health literacy moderate how health behaviors translate to outcomes?
  
    ("Education",     "GenHlth"),
    ("Education",     "BMI"),                # health literacy around weight management
    ("Education",     "PhysActivity"),       # knowledge translating to behavior

    # Healthcare access × health indicators
    # Does insurance coverage buffer the health → diabetes pathway?
  
    ("AnyHealthcare", "GenHlth"),
    ("AnyHealthcare", "PhysHlth"),
    ("AnyHealthcare", "HighBP"),             # BP management with insurance
    ("NoDocbcCost",   "GenHlth"),            # couldn't afford care AND poor health
    ("NoDocbcCost",   "PhysHlth"),           # cost barrier + physical health burden

    # SES composite × top clinical predictor
  
    ("Education",     "HighBP"),             # health literacy around hypertension management

    # Pure SES interactions — combined socioeconomic status signals
  
    ("Income",        "Education"),          # wealth + knowledge together
    ("Income",        "AnyHealthcare"),      # wealth determines insurance coverage
    ("NoDocbcCost",   "AnyHealthcare"),      # insured but still couldn't afford care
]

# Deduplicate: if (A, B) and (B, A) both appear, keep the first occurrence only

seen = set()
ses_interaction_pairs = []
for pair in ses_interaction_pairs_raw:
    key = tuple(sorted(pair))
    if key not in seen:
        seen.add(key)
        ses_interaction_pairs.append(pair)

ses_interaction_names = [f"{a} x {b}" for a, b in ses_interaction_pairs]

print(f"\nOriginal features:  {X_raw.shape[1]}")
print(f"SES interactions:   {len(ses_interaction_pairs)}")



# 4. Build the SES-engineered feature matrix
#    Multiply each pair and append as a new column.

X_ses = X_raw.copy()
for a, b in ses_interaction_pairs:
    X_ses[f"{a} x {b}"] = X_raw[a] * X_raw[b]

ses_feature_names = list(X_ses.columns)

print(f"Total features:     {X_ses.shape[1]}")

# Scale so LASSO applies the penalty fairly across all features

X_ses_scaled = StandardScaler().fit_transform(X_ses)



# 5. Fit SES-engineered LASSO
#    L1 penalty automatically zeroes out features that don't add predictive value.

lasso_ses = LogisticRegressionCV(
    Cs=Cs, penalty="l1", solver="liblinear",
    cv=cv, max_iter=1000, random_state=42, scoring="roc_auc"
)
lasso_ses.fit(X_ses_scaled, y)
coef_ses = pd.Series(lasso_ses.coef_[0], index=ses_feature_names)

print(f"\n[SES Engineered] Best C = {lasso_ses.C_[0]:.5f} | "
      f"Non-zero features: {(coef_ses != 0).sum()} / {len(ses_feature_names)}")

print("\nAll selected features (sorted by magnitude):")
print(coef_ses[coef_ses != 0].sort_values(key=abs, ascending=False).to_string())


# 6. Which SES interaction terms did LASSO keep?
#    If LASSO zeroes out an interaction, it means the interaction didn't add
#    explanatory power beyond the individual components already in the model.

sel_ses = coef_ses[ses_interaction_names]
sel_ses = sel_ses[sel_ses != 0].sort_values(key=abs, ascending=False)

print("\n=== SES interaction terms selected by Lasso ===")
if len(sel_ses):
    print(sel_ses.to_string())
    print("\n--- Component status (were the individual features also kept?) ---")
    for name in sel_ses.index:
        a, b = name.split(" x ", 1)
        ca = coef_ses.get(a, 0)
        cb = coef_ses.get(b, 0)
        print(f"  {name}: {sel_ses[name]:+.4f}")
        print(f"    {a}: {ca:+.4f}{'  (zeroed out)' if ca == 0 else ''}")
        print(f"    {b}: {cb:+.4f}{'  (zeroed out)' if cb == 0 else ''}")
else:
    print("  None selected — Lasso preferred original features over SES interactions.")


# 7. AUC-ROC comparison helper
#    Runs k-fold CV and returns mean + std AUC so we can compare models fairly.

def cv_auc(model, X, y, cv):
    """
    Compute mean and standard deviation of ROC-AUC across cross-validation folds.

    Parameters
    ----------
    model : sklearn estimator
        An unfitted classifier with predict_proba support (e.g. LogisticRegression).
    X : np.ndarray
        Feature matrix (scaled), shape (n_samples, n_features).
    y : np.ndarray or pd.Series
        Binary target labels (0 = no diabetes, 1 = diabetes).
    cv : sklearn cross-validator
        A splitter such as StratifiedKFold that yields (train_idx, test_idx) pairs.

    Returns
    -------
    mean_auc : float
        Mean ROC-AUC across all folds.
    std_auc : float
        Standard deviation of ROC-AUC across all folds.
    """
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], prob))
    return np.mean(aucs), np.std(aucs)

auc_ses = cv_auc(
    LogisticRegression(C=lasso_ses.C_[0], penalty="l1",
                       solver="liblinear", max_iter=1000),
    X_ses_scaled, y, cv
)

# Hardcoded baseline and bio-engineered results from the notebook
# (avoids re-running those models every time this script is called)
results = {
    "Baseline":       (0.8247, 0.0022),
    "Bio Engineered": (0.8275, 0.0018),
    "SES Engineered": auc_ses,
}

print("\n=== 5-fold CV AUC-ROC ===")
for name, (mean, std) in results.items():
    print(f"  {name:<15}  {mean:.4f} +/- {std:.4f}")



# 8. Visualizations


# 8a. Feature importance chart for the SES engineered model 

nonzero_ses = coef_ses[coef_ses != 0].copy()
nonzero_ses = nonzero_ses.reindex(nonzero_ses.abs().sort_values(ascending=True).index)
is_ses_int  = nonzero_ses.index.isin(ses_interaction_names)
bar_colors  = ["#2ecc71" if inter else "#4a90d9" for inter in is_ses_int]

fig, ax = plt.subplots(figsize=(10, 10))
bars = ax.barh(nonzero_ses.index, nonzero_ses.values, color=bar_colors,
               edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Logistic Lasso Coefficient (log-odds)", fontsize=12)
ax.set_title(
    "SES Engineered Lasso — Feature Importance\n"
    "(green = SES interaction term, blue = original feature)",
    fontsize=13, pad=15,
)
legend_elements = [
    Patch(facecolor="#4a90d9", label="Original feature"),
    Patch(facecolor="#2ecc71", label="SES interaction term"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=11)
for bar, val in zip(bars, nonzero_ses.values):
    x_pos = val + 0.008 if val >= 0 else val - 0.008
    ha = "left" if val >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=7.5)
plt.tight_layout()
plt.show()

# 8b. Model performance comparison bar chart 

means  = [v[0] for v in results.values()]
stds   = [v[1] for v in results.values()]
labels = list(results.keys())

plt.figure(figsize=(7, 4))
bars = plt.bar(labels, means, yerr=stds, capsize=5,
               color=["gray", "steelblue", "salmon"], alpha=0.85)
for bar, mean in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width() / 2, mean + 0.0005,
             f"{mean:.4f}", ha="center", fontsize=10)
plt.ylabel("5-Fold CV AUC-ROC")
plt.title("Model Performance Comparison")
plt.ylim(min(means) - 0.01, max(means) + 0.01)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

"""
8c. Diabetes rate by GenHlth score split across income quartiles 
      This visualization shows the raw data pattern that motivates the
      Income x GenHlth interaction term — does income modify how general
      health translates into diabetes risk?
"""

df_plot = X_raw.copy()
df_plot["Diabetes"] = y
df_plot["IncomeQuartile"] = pd.cut(
    df_plot["Income"], bins=[0, 2, 4, 6, 8],
    labels=["Low", "Mid-Low", "Mid-High", "High"],
)

grouped = df_plot.groupby(
    ["GenHlth", "IncomeQuartile"], observed=True
)["Diabetes"].mean().reset_index()
pivot = grouped.pivot(index="GenHlth", columns="IncomeQuartile", values="Diabetes")

fig, ax = plt.subplots(figsize=(8, 5))
colors_q = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
for col, color in zip(pivot.columns, colors_q):
    ax.plot(pivot.index, pivot[col] * 100, marker="o",
            label=col, color=color, lw=2)
ax.set_xlabel("General Health Score (1=Excellent → 5=Poor)", fontsize=11)
ax.set_ylabel("Diabetes Rate (%)", fontsize=11)
ax.set_title(
    "Diabetes Rate by General Health Score and Income Quartile\n"
    "(shows how income modifies the GenHlth → diabetes pathway)",
    fontsize=12,
)
ax.legend(title="Income Quartile", fontsize=10)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["1\nExcellent", "2\nVery Good", "3\nGood", "4\nFair", "5\nPoor"])
plt.tight_layout()
plt.show()
