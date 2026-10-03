from pathlib import Path

import joblib
import pandas as pd

from agent.cfo_agent import run_cfo_analysis


# ------------------------------------------------------------
# 1. Locate deployment artifacts inside the repository
# ------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

DEPLOY_ROOT = (
    REPO_ROOT
    / "deployment_artifacts"
)

panel_path = (
    DEPLOY_ROOT
    / "data"
    / "chapter3_deployment_panel.parquet"
)

model_path = (
    DEPLOY_ROOT
    / "models"
    / "xgb_distress_h1.joblib"
)


# ------------------------------------------------------------
# 2. Load deployment Chapter 3 panel
# ------------------------------------------------------------

panel_df = pd.read_parquet(
    panel_path
)


# ------------------------------------------------------------
# 3. Load Chapter 3 distress model
# ------------------------------------------------------------

model = joblib.load(
    model_path
)

model_feature_names = list(
    model.feature_names_in_
)


# ------------------------------------------------------------
# 4. Select Nicholas Financial test case
# ------------------------------------------------------------

sample_match = panel_df[
    (
        panel_df["cik"]
        .astype(str)
        == "1000045"
    )
    & (
        pd.to_datetime(
            panel_df["period"]
        )
        == pd.Timestamp(
            "2021-03-31"
        )
    )
]

if sample_match.empty:
    raise ValueError(
        "Nicholas Financial 2021-03-31 was not found "
        "in the deployment Chapter 3 panel."
    )

sample_row = (
    sample_match.iloc[0]
)


# ------------------------------------------------------------
# 5. Build company input
# ------------------------------------------------------------

company_data = {
    feature: sample_row[
        feature
    ]
    for feature in model_feature_names
}

company_data["cik"] = str(
    sample_row[
        "cik"
    ]
)

company_data["period"] = (
    pd.to_datetime(
        sample_row[
            "period"
        ]
    )
    .strftime(
        "%Y-%m-%d"
    )
)


# ------------------------------------------------------------
# 6. Run complete AI CFO workflow
# ------------------------------------------------------------

result = run_cfo_analysis(
    company_data
)


# ------------------------------------------------------------
# 7. Display integration results
# ------------------------------------------------------------

print(
    "=" * 70
)

print(
    "AI CFO DEPLOYMENT INTEGRATION TEST"
)

print(
    "=" * 70
)

print(
    "\nCompany:"
)

print(
    sample_row[
        "company_name"
    ]
)

print(
    "\nPeriod:"
)

print(
    company_data[
        "period"
    ]
)

print(
    "\nFORECAST RESULT:"
)

print(
    result[
        "forecast"
    ]
)

print(
    "\nRECOMMENDATION RESULT:"
)

print(
    result[
        "recommendations"
    ]
)

print(
    "\nEXPLAINABILITY RESULT:"
)

print(
    result[
        "explanation"
    ]
)

print(
    "\nSIMULATION RESULT:"
)

print(
    result[
        "simulation"
    ]
)

print(
    "\nEXECUTIVE BRIEF:"
)

print(
    result[
        "executive_brief"
    ]
)
