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
# 2. Load the real Chapter 3 modeling panel
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
# 4. Load the real Chapter 3 distress model
#    so we can use its exact feature order
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
# 5. Select a real distressed company-quarter
# ------------------------------------------------------------

sample_row = panel_df[
    panel_df["target_distress_h1"] == 1
].iloc[0]


# ------------------------------------------------------------
# 6. Build the company input expected by the AI CFO
# ------------------------------------------------------------

company_data = {
    feature: sample_row[feature]
    for feature in model_feature_names
}


# ------------------------------------------------------------
# 7. Run the AI CFO orchestration layer
# ------------------------------------------------------------

result = run_cfo_analysis(
    company_data
)


# ------------------------------------------------------------
# 8. Display the test case and results
# ------------------------------------------------------------

print("=" * 70)
print("AI CFO INTEGRATION TEST")
print("=" * 70)

print("\nCompany:")
print(sample_row["company_name"])

print("\nYear / Quarter:")
print(
    int(sample_row["year"]),
    "/",
    int(sample_row["quarter"])
)

print("\nActual next-quarter distress:")
print(
    int(sample_row["target_distress_h1"])
)

print("\nAI CFO results:")
print(result)
