"""
========================================================
  Employee Attrition Prediction & Agentic AI System
  CS Third-Year Project | Random Forest + Rule Engine
========================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────

print("=" * 60)
print("  EMPLOYEE ATTRITION PREDICTION SYSTEM")
print("=" * 60)

DATA_PATH   = "enriched_employees2.csv"
OUTPUT_PATH = "attrition_results.csv"

df = pd.read_csv(DATA_PATH)
print(f"\n✅  Loaded dataset: {df.shape[0]} employees, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2.  PRE-PROCESSING
# ─────────────────────────────────────────────

# Drop columns that carry no predictive signal
drop_cols = [
    "EmployeeCount", "EmployeeNumber", "Over18",
    "StandardHours"
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Encode target
df["Attrition_Binary"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Encode all remaining object columns
le = LabelEncoder()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# Feature matrix
feature_cols = [
    c for c in df.columns
    if c not in (
        ["Attrition", "Attrition_Binary"]
        + cat_cols                         # raw string columns out
        + [c for c in df.columns if "AgeGroup" in c or
           "SalaryBand" in c or "TenureGroup" in c]
    )
]
# Keep only numeric + encoded columns
feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in feature_cols if c != "Attrition_Binary"]

X = df[feature_cols]
y = df["Attrition_Binary"]

print(f"   Features used for training: {len(feature_cols)}")

# ─────────────────────────────────────────────
# 3.  TRAIN RANDOM FOREST MODEL
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",   # handles imbalanced attrition data
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred      = rf.predict(X_test)
y_proba     = rf.predict_proba(X_test)[:, 1]   # probability of attrition

# ─────────────────────────────────────────────
# 4.  MODEL EVALUATION
# ─────────────────────────────────────────────

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "─" * 60)
print("  MODEL PERFORMANCE")
print("─" * 60)
print(f"  Accuracy          : {accuracy * 100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))
print("  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# Top 10 feature importances
fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n  Top 10 Most Important Features:")
for feat, imp in fi.head(10).items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────
# 5.  PREDICT ON FULL DATASET
# ─────────────────────────────────────────────

full_proba  = rf.predict_proba(X)[:, 1]
full_pred   = rf.predict(X)

df["ML_AttritionProbability"] = np.round(full_proba * 100, 2)
df["ML_PredictedAttrition"]   = np.where(full_pred == 1, "Yes", "No")

# ─────────────────────────────────────────────
# 6.  AGENTIC AI – RULE-BASED RISK ENGINE
# ─────────────────────────────────────────────

def compute_risk_flags(row):
    """
    Rule engine that examines multiple HR risk dimensions
    and returns a composite list of triggered flags.
    """
    flags = []

    # --- Compensation ---
    if row.get("MonthlyIncome", 99999) < 3000:
        flags.append("LOW_SALARY")
    if row.get("PercentSalaryHike", 99) < 12:
        flags.append("BELOW_AVG_HIKE")

    # --- Work-Life Balance ---
    if row.get("OverTime") == "Yes":
        flags.append("OVERTIME_STRESS")
    if row.get("WorkLifeBalance", 99) <= 1:
        flags.append("POOR_WLB")

    # --- Satisfaction ---
    if row.get("JobSatisfaction", 99) <= 1:
        flags.append("LOW_JOB_SATISFACTION")
    if row.get("EnvironmentSatisfaction", 99) <= 1:
        flags.append("POOR_ENVIRONMENT")
    if row.get("RelationshipSatisfaction", 99) <= 1:
        flags.append("POOR_RELATIONSHIPS")

    # --- Career Growth ---
    if row.get("YearsSinceLastPromotion", 0) >= 4:
        flags.append("STAGNANT_CAREER")
    if row.get("TrainingTimesLastYear", 99) == 0:
        flags.append("NO_TRAINING")
    if row.get("JobInvolvement", 99) <= 1:
        flags.append("LOW_INVOLVEMENT")

    # --- Tenure / Flight Risk ---
    if row.get("YearsAtCompany", 99) <= 2:
        flags.append("NEW_HIRE_FLIGHT_RISK")
    if row.get("NumCompaniesWorked", 0) >= 5:
        flags.append("HIGH_JOB_HOPPER")

    # --- ML Model Signal ---
    if row.get("ML_AttritionProbability", 0) >= 70:
        flags.append("ML_HIGH_RISK")
    elif row.get("ML_AttritionProbability", 0) >= 50:
        flags.append("ML_MEDIUM_RISK")

    return flags


# Intervention recommendation library
INTERVENTIONS = {
    "LOW_SALARY":            "💰 Conduct immediate salary benchmarking and adjust compensation to market rate.",
    "BELOW_AVG_HIKE":        "📈 Review annual appraisal cycle; ensure hike is at least at industry median.",
    "OVERTIME_STRESS":       "⏱️  Audit workload distribution; enforce overtime caps and offer comp time.",
    "POOR_WLB":              "🏡 Introduce flexible/remote work options and mandatory wellness days.",
    "LOW_JOB_SATISFACTION":  "💬 Schedule 1-on-1 with manager to identify and resolve dissatisfaction drivers.",
    "POOR_ENVIRONMENT":      "🏢 Review team dynamics, physical workspace, and DEI climate surveys.",
    "POOR_RELATIONSHIPS":    "🤝 Facilitate team-building activities and peer mentoring programs.",
    "STAGNANT_CAREER":       "🚀 Create an individual development plan (IDP) and fast-track promotion review.",
    "NO_TRAINING":           "📚 Immediately enroll in skill-development programs; assign a learning budget.",
    "LOW_INVOLVEMENT":       "🎯 Re-align role responsibilities; consider lateral move to a higher-impact project.",
    "NEW_HIRE_FLIGHT_RISK":  "🎓 Strengthen onboarding; assign a buddy/mentor for the first 6 months.",
    "HIGH_JOB_HOPPER":       "🔗 Offer retention bonus tied to 2-year tenure milestone.",
    "ML_HIGH_RISK":          "🚨 AI model flags very high attrition probability — escalate to CHRO.",
    "ML_MEDIUM_RISK":        "⚠️  AI model flags moderate attrition probability — place on HR watch list.",
}


def compute_risk_tier(flags, ml_prob):
    """Assign an overall risk tier based on flag count and ML probability."""
    flag_count = len(flags)
    if ml_prob >= 70 or flag_count >= 6:
        return "🔴 CRITICAL"
    elif ml_prob >= 50 or flag_count >= 4:
        return "🟠 HIGH"
    elif ml_prob >= 30 or flag_count >= 2:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"


def generate_recommendations(flags):
    """Map flags to deduplicated intervention recommendations."""
    recs = []
    for flag in flags:
        rec = INTERVENTIONS.get(flag)
        if rec and rec not in recs:
            recs.append(rec)
    return recs


# ─────────────────────────────────────────────
# 7.  APPLY AGENT TO ALL EMPLOYEES
# ─────────────────────────────────────────────

print("\n" + "─" * 60)
print("  AGENTIC AI – PROCESSING ALL EMPLOYEES …")
print("─" * 60)

risk_tiers     = []
flag_lists     = []
rec_lists      = []
action_counts  = []

for _, row in df.iterrows():
    flags = compute_risk_flags(row)
    recs  = generate_recommendations(flags)
    tier  = compute_risk_tier(flags, row["ML_AttritionProbability"])

    risk_tiers.append(tier)
    flag_lists.append(" | ".join(flags) if flags else "NONE")
    rec_lists.append(" | ".join(recs)   if recs  else "No immediate action required.")
    action_counts.append(len(recs))

df["RiskTier"]           = risk_tiers
df["RiskFlags"]          = flag_lists
df["HRRecommendations"]  = rec_lists
df["InterventionCount"]  = action_counts

# ─────────────────────────────────────────────
# 8.  SUMMARY STATS
# ─────────────────────────────────────────────

tier_counts = df["RiskTier"].value_counts()
print("\n  Risk Distribution Across Workforce:")
for tier, count in tier_counts.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {tier:<18}  {count:>4} employees  ({pct:5.1f}%)  {bar}")

critical_df = df[df["RiskTier"] == "🔴 CRITICAL"]
print(f"\n  ⚠️   Critical-risk employees identified: {len(critical_df)}")

if len(critical_df) > 0:
    print("\n  Sample Critical-Risk Employees (top 5):")
    cols_to_show = [
        "JobRole", "Department", "MonthlyIncome",
        "ML_AttritionProbability", "RiskFlags"
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    print(critical_df[cols_to_show].head(5).to_string(index=True))

# ─────────────────────────────────────────────
# 9.  EXPORT RESULTS
# ─────────────────────────────────────────────

# Select clean output columns
output_cols = [
    "Age", "Department", "JobRole", "Gender",
    "MonthlyIncome", "OverTime", "JobSatisfaction",
    "WorkLifeBalance", "YearsAtCompany", "YearsSinceLastPromotion",
    "RiskScore", "Attrition",
    "ML_AttritionProbability", "ML_PredictedAttrition",
    "RiskTier", "RiskFlags", "HRRecommendations", "InterventionCount"
]
output_cols = [c for c in output_cols if c in df.columns]

df[output_cols].to_csv(OUTPUT_PATH, index=False)
print(f"\n✅  Results exported → {OUTPUT_PATH}  ({len(df)} rows)")

print("\n" + "=" * 60)
print("  DONE. Review attrition_results.csv for full HR report.")
print("=" * 60 + "\n")
