import os
import joblib
import pandas as pd

from agent.cfo_agent import run_cfo_analysis


# ------------------------------------------------------------
# 1. Point the hackathon app to your local thesis project
# ------------------------------------------------------------

os.environ["AI_CFO_THESIS_ROOT"] = (
    "/Users/humairaali/Revised_AI_CFO_Thesis"
)


# ------------------------------------------------------------
# 2. Load the Chapter 3 modeling panel
# ------------------------------------------------------------

panel_path = (
    "/Users/humairaali/"
    "Revised_AI_CFO_Thesis/"
    "data/processed/"
    "chapter3_model_panel_final_v1.parquet"
)

panel_df = pd.read_parquet(panel_path)


# ------------------------------------------------------------
# 3. Recreate sic_code exactly as Chapter 3 did
# ------------------------------------------------------------

panel_df["sic"] = panel_df["sic"].astype(str)

panel_df["sic_code"] = pd.factorize(
    panel_df["sic"]
)[0]


# ------------------------------------------------------------
# 4. Load the Chapter 3 distress model
# ------------------------------------------------------------

model_path = (
    "/Users/humairaali/"
    "Revised_AI_CFO_Thesis/"
    "models/"
    "xgb_distress_h1.joblib"
)

model = joblib.load(model_path)

model_feature_names = list(
    model.feature_names_in_
)


# ------------------------------------------------------------
# 5. Select a company-quarter that exists in both
#    Chapter 3 and Chapter 4
# ------------------------------------------------------------

sample_match = panel_df[
    (panel_df["cik"].astype(str) == "1000045")
    & (
        pd.to_datetime(panel_df["period"])
        == pd.Timestamp("2021-03-31")
    )
]

if sample_match.empty:
    raise ValueError(
        "Nicholas Financial 2021-03-31 was not found "
        "in the Chapter 3 modeling panel."
    )

sample_row = sample_match.iloc[0]


# ------------------------------------------------------------
# 6. Build the company input
#
#    The 88 model features are used by Chapter 3.
#    CIK and period are included so Chapter 4 can locate the
#    corresponding governance-aware recommendation record.
# ------------------------------------------------------------

company_data = {
    feature: sample_row[feature]
    for feature in model_feature_names
}

company_data["cik"] = str(
    sample_row["cik"]
)

company_data["period"] = (
    pd.to_datetime(
        sample_row["period"]
    ).strftime("%Y-%m-%d")
)


# ------------------------------------------------------------
# 7. Run the AI CFO orchestration layer
# ------------------------------------------------------------

result = run_cfo_analysis(
    company_data
)


# ------------------------------------------------------------
# 8. Display integration results
# ------------------------------------------------------------

print("=" * 70)
print("AI CFO CHAPTER 3 + CHAPTER 4 INTEGRATION TEST")
print("=" * 70)

print("\nCompany:")
print(sample_row["company_name"])

print("\nPeriod:")
print(company_data["period"])

print("\nActual next-quarter distress:")
print(
    int(sample_row["target_distress_h1"])
)

print("\nFORECAST RESULT:")
print(result["forecast"])

print("\nRECOMMENDATION RESULT:")
print(result["recommendations"])

print("\nEXPLAINABILITY RESULT:")
print(result["explanation"])

print("\nSIMULATION RESULT:")
print(result["simulation"])
