import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    ROOT
    / "deployment_artifacts"
    / "models"
    / "chapter5_xai"
)


ACTION_MODEL_FILES = {
    "Accelerate Receivables Collection":
        "shap_surrogate_accelerate_receivables_collection.joblib",

    "Delay Capital Expenditures":
        "shap_surrogate_delay_capital_expenditures.joblib",

    "Hold Strategy":
        "shap_surrogate_hold_strategy.joblib",

    "Improve Working Capital Efficiency":
        "shap_surrogate_improve_working_capital_efficiency.joblib",

    "Increase Cash Reserves":
        "shap_surrogate_increase_cash_reserves.joblib",

    "Increase Revenue Growth Investment":
        "shap_surrogate_increase_revenue_growth_investment.joblib",

    "Reduce Debt Exposure":
        "shap_surrogate_reduce_debt_exposure.joblib",

    "Reduce Operating Expenses":
        "shap_surrogate_reduce_operating_expenses.joblib",

    "Reduce SG&A Expenses":
        "shap_surrogate_reduce_sganda_expenses.joblib",

    "Refinance Debt":
        "shap_surrogate_refinance_debt.joblib",
}


FEATURES = [
    "Financial_Health_Score",
    "Distress_Probability",
    "Regime_Growth",
    "Regime_Balanced",
    "Regime_Defensive",
]


def build_surrogate_input(
    financial_health,
    distress_probability,
    strategic_regime,
):

    return pd.DataFrame(
        [{
            "Financial_Health_Score": financial_health,
            "Distress_Probability": distress_probability,
            "Regime_Growth":
                1 if strategic_regime == "Growth" else 0,
            "Regime_Balanced":
                1 if strategic_regime == "Balanced" else 0,
            "Regime_Defensive":
                1 if strategic_regime == "Defensive" else 0,
        }],
        columns=FEATURES,
    )


def explain_recommendation(
    action,
    financial_health,
    distress_probability,
    strategic_regime,
):

    if action not in ACTION_MODEL_FILES:
        raise ValueError(
            f"No Chapter 5 surrogate model found for action: {action}"
        )

    model_path = MODEL_DIR / ACTION_MODEL_FILES[action]

    model = joblib.load(model_path)

    X = build_surrogate_input(
        financial_health=financial_health,
        distress_probability=distress_probability,
        strategic_regime=strategic_regime,
    )

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP versions represent binary classifiers differently.
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])[0]

    else:
        values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[0, :, 1]
        elif values.ndim == 2:
            values = values[0]

    contributions = pd.DataFrame({
        "Feature": FEATURES,
        "Value": X.iloc[0].values,
        "SHAP_Contribution": values,
    })

    contributions["Absolute_Contribution"] = (
        contributions["SHAP_Contribution"].abs()
    )

    contributions = contributions.sort_values(
        "Absolute_Contribution",
        ascending=False,
    ).reset_index(drop=True)

    return {
        "Action": action,
        "Surrogate_Probability": probability,
        "Input": X,
        "Contributions": contributions,
    }
