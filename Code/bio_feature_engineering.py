"""
bio_feature_engineering.py 

Isaac Fine

Biological / physiological interaction feature engineering for diabetes prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Separate the target variable from the features
target        = "Diabetes_binary"
y             = df[target].values
X_raw         = df.drop(columns=[target])
feature_names = list(X_raw.columns)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# 2. Scale the raw features
#    StandardScaler puts every feature on the same scale (mean=0, std=1),
#    which is required for LASSO so the penalty is applied fairly across
#    features with different units / ranges.

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)


# 3. Define interaction pairs
#    Each tuple (A, B) creates a new feature A × B.
#    - binary × binary  → 1 only when the patient has BOTH conditions
#    - continuous × binary → continuous value "switched on" when binary = 1
#    - continuous × continuous → standard multiplicative interaction

interaction_pairs = [
    ("HighBP",      "HighChol"),            # both hypertension AND high cholesterol
    ("HighBP",      "HeartDiseaseorAttack"), # hypertension compounding cardiac history
    ("HighChol",    "HeartDiseaseorAttack"), # cholesterol compounding cardiac history
    ("HighBP",      "Stroke"),              # hypertension + stroke history
    ("BMI",         "PhysActivity"),        # is high BMI riskier without exercise?
    ("BMI",         "Age"),                 # does age amplify BMI-related risk?
    ("Smoker",      "HvyAlcoholConsump"),   # combined substance use
    ("Fruits",      "Veggies"),             # combined healthy diet signal
    ("GenHlth",     "PhysHlth"),            # general health × physical health days
    ("GenHlth",     "MentHlth"),            # general health × mental health days
    ("PhysHlth",    "DiffWalk"),            # physical health days × mobility difficulty
    ("GenHlth",     "DiffWalk"),            # general health × mobility difficulty
    ("Income",      "Education"),           # socioeconomic composite
    ("Income",      "AnyHealthcare"),       # wealth determines insurance access
    ("NoDocbcCost", "AnyHealthcare"),       # insured but still can't afford care
    ("Age",         "HeartDiseaseorAttack"),# age amplifying cardiac risk
    ("Age",         "Stroke"),             # age amplifying stroke history
    ("Age",         "DiffWalk"),           # age amplifying mobility difficulty
]


# 4. Build the engineered feature matrix
#    Multiply each pair of columns and append the result as a new column.

X_eng = X_raw.copy()
for a, b in interaction_pairs:
    X_eng[f"{a} x {b}"] = X_raw[a] * X_raw[b]

interaction_names = [f"{a} x {b}" for a, b in interaction_pairs]
eng_feature_names = list(X_eng.columns)

print(f"Original features: {X_raw.shape[1]}  ->  After engineering: {X_eng.shape[1]}")

# Scale the engineered matrix the same way as the baseline
X_eng_scaled = StandardScaler().fit_transform(X_eng)


# 5. Cross-validation setup
#    StratifiedKFold keeps the class balance the same in every fold,
#    which matters because the dataset is 50/50 split but we want to be safe.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
Cs = np.logspace(-3, 1, 40)   # regularization strengths to search over


# 6. Fit baseline LASSO (no engineered features)
#    LogisticRegressionCV automatically picks the best C (inverse regularization
#    strength) via cross-validation. L1 penalty drives weak features to exactly 0.

lasso_base = LogisticRegressionCV(
    Cs=Cs, penalty="l1", solver="liblinear",
    cv=cv, max_iter=1000, random_state=42, scoring="roc_auc"
)
lasso_base.fit(X_scaled, y)
coef_base = pd.Series(lasso_base.coef_[0], index=feature_names)

print(f"\n[Baseline] Best C = {lasso_base.C_[0]:.5f} | "
      f"Non-zero features: {(coef_base != 0).sum()} / {len(feature_names)}")


# 7. Fit engineered LASSO (with interaction features)

lasso_eng = LogisticRegressionCV(
    Cs=Cs, penalty="l1", solver="liblinear",
    cv=cv, max_iter=1000, random_state=42, scoring="roc_auc"
)
lasso_eng.fit(X_eng_scaled, y)
coef_eng = pd.Series(lasso_eng.coef_[0], index=eng_feature_names)

print(f"\n[Engineered] Best C = {lasso_eng.C_[0]:.5f} | "
      f"Non-zero features: {(coef_eng != 0).sum()} / {len(eng_feature_names)}")
print("\nSelected features (sorted by magnitude):")
print(coef_eng[coef_eng != 0].sort_values(key=abs, ascending=False).to_string())


# 8. Before vs. after comparison
#    Check which features changed sign, were dropped, or were newly selected
#    after adding interaction terms.

compare = pd.DataFrame({
    "baseline":   coef_base,
    "engineered": coef_eng[feature_names],   # only original features for a fair compare
})
compare["changed_sign"] = (
    (compare["baseline"] != 0) & (compare["engineered"] != 0) &
    (np.sign(compare["baseline"]) != np.sign(compare["engineered"]))
)
compare["dropped"] = (compare["baseline"] != 0) & (compare["engineered"] == 0)
compare["gained"]  = (compare["baseline"] == 0) & (compare["engineered"] != 0)

print("\n=== Sign flips ===")
flips = compare[compare["changed_sign"]][["baseline", "engineered"]]
print(flips.to_string() if len(flips) else "  None")

print("\n=== Dropped after engineering ===")
print(compare[compare["dropped"]].index.tolist() or "  None")

print("\n=== Newly selected after engineering ===")
gained = compare[compare["gained"]][["baseline", "engineered"]]
print(gained.to_string() if len(gained) else "  None")

# Which interaction terms did LASSO keep?
sel_int = coef_eng[interaction_names]
sel_int = sel_int[sel_int != 0].sort_values(key=abs, ascending=False)

print("\n=== Interaction terms selected by Lasso ===")
if len(sel_int):
    print(sel_int.to_string())
else:
    print("  None — Lasso preferred original features over interactions.")


# 9. AUC-ROC comparison helper
#    Runs k-fold CV and returns mean + std AUC so we can compare models fairly.

def cv_auc(model, X, y, cv):
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], prob))
    return np.mean(aucs), np.std(aucs)

# Refit single LogisticRegression models with the best C found above

results = {
    "Baseline":   cv_auc(
        LogisticRegression(C=lasso_base.C_[0], penalty="l1",
                           solver="liblinear", max_iter=1000),
        X_scaled, y, cv
    ),
    "Engineered": cv_auc(
        LogisticRegression(C=lasso_eng.C_[0], penalty="l1",
                           solver="liblinear", max_iter=1000),
        X_eng_scaled, y, cv
    ),
}

print("\n=== 5-fold CV AUC-ROC ===")
for name, (mean, std) in results.items():
    print(f"  {name:<12}  {mean:.4f} +/- {std:.4f}")


# 10. Visualizations:

#  Normalized coefficient comparison (baseline vs engineered) 

norm_b = compare["baseline"]   / (compare["baseline"].abs().max()   + 1e-10)
norm_e = compare["engineered"] / (compare["engineered"].abs().max() + 1e-10)
order  = compare["baseline"].abs().sort_values(ascending=False).index

x, w = np.arange(len(order)), 0.35
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - w/2, norm_b[order], w, label="Baseline",   color="steelblue", alpha=0.85)
ax.bar(x + w/2, norm_e[order], w, label="Engineered", color="salmon",    alpha=0.85)
for i, feat in enumerate(order):
    if compare.loc[feat, "changed_sign"]:
        ax.annotate("!", xy=(i, 0.05), ha="center", color="red",
                    fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Normalized coefficient")
ax.set_title("Lasso: baseline vs. engineered (original features — ! = sign flip)")
ax.legend()
plt.tight_layout()
plt.show()

# 10b. Engineered LASSO feature importance chart
nonzero = coef_eng[coef_eng != 0].copy()
nonzero = nonzero.reindex(nonzero.abs().sort_values(ascending=True).index)
is_interaction = nonzero.index.isin(interaction_names)
colors = ["#e07b54" if inter else "#4a90d9" for inter in is_interaction]

fig, ax = plt.subplots(figsize=(10, 10))
bars = ax.barh(nonzero.index, nonzero.values, color=colors,
               edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Logistic Lasso Coefficient (log-odds)", fontsize=12)
ax.set_title(
    "Engineered Lasso — Feature Importance\n"
    "(positive = increases diabetes risk, negative = protective)",
    fontsize=13, pad=15,
)
legend_elements = [
    Patch(facecolor="#4a90d9", label="Original feature"),
    Patch(facecolor="#e07b54", label="Interaction term"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=11)
for bar, val in zip(bars, nonzero.values):
    x_pos = val + 0.01 if val >= 0 else val - 0.01
    ha = "left" if val >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=7.5)
plt.tight_layout()
plt.show()

# AUC bar chart 
means = [v[0] for v in results.values()]
stds  = [v[1] for v in results.values()]
plt.figure(figsize=(5, 4))
plt.bar(results.keys(), means, yerr=stds, capsize=5,
        color=["steelblue", "salmon"], alpha=0.85)
plt.ylabel("5-fold CV AUC-ROC")
plt.title("Predictive performance — before vs. after engineering")
plt.ylim(min(means) - 0.02, max(means) + 0.02)
plt.tight_layout()
plt.show()
