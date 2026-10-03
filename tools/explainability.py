import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ------------------------------------------------------------
# Chapter 5 surrogate model mapping
# ------------------------------------------------------------

SURROGATE_MODEL_FILES = {
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


# ------------------------------------------------------------
# Locate Chapter 5 model directory
# ------------------------------------------------------------

def _get_chapter5_model_dir():
    """
    Locate the Chapter 5 SHAP surrogate model directory.

    AI_CFO_THESIS_ROOT must point to the local dissertation project.
    """

    thesis_root = os.environ.get("AI_CFO_THESIS_ROOT")

    if not thesis_root:
        raise EnvironmentError(
            "AI_CFO_THESIS_ROOT is not set. "
            "Set it to the local Revised_AI_CFO_Thesis folder."
        )

    thesis_root = Path(thesis_root)

    return (
        thesis_root
        / "models"
        / "chapter5_xai"
    )


# ------------------------------------------------------------
# Build the 5-feature Chapter 5 surrogate input
# ------------------------------------------------------------

def _build_surrogate_input(recommendation_result: dict) -> pd.DataFrame:
    """
    Construct the five Chapter 5 surrogate-model features from
    the Chapter 4 recommendation output.
    """

    regime = recommendation_result["strategic_regime"]

    xai_input = pd.DataFrame(
        [{
            "Financial_Health_Score":
                float(
                    recommendation_result[
                        "financial_health_score"
                    ]
                ),

            "Distress_Probability":
                float(
                    recommendation_result[
                        "chapter4_distress_probability"
                    ]
                ),

            "Regime_Growth":
                1 if regime == "Growth" else 0,

            "Regime_Balanced":
                1 if regime == "Balanced" else 0,

            "Regime_Defensive":
                1 if regime == "Defensive" else 0,
        }]
    )

    return xai_input


# ------------------------------------------------------------
# Explain Chapter 4 recommendation using Chapter 5 SHAP model
# ------------------------------------------------------------

def explain_prediction(
    model_output: dict,
    recommendation_result: dict = None
) -> dict:
    """
    Explain the top Chapter 4 recommendation using the
    corresponding Chapter 5 SHAP surrogate model.

    SHAP values describe contribution to the surrogate model's
    recommendation prediction. They do NOT establish causality.
    """

    if recommendation_result is None:
        return {
            "tool": "explain_prediction",
            "status": "missing_recommendation_context",
            "message": (
                "Chapter 5 explainability requires the Chapter 4 "
                "recommendation result."
            ),
        }

    if recommendation_result.get("status") != "success":
        return {
            "tool": "explain_prediction",
            "status": "recommendation_unavailable",
            "message": (
                "A successful Chapter 4 recommendation is required "
                "before SHAP explanation can be generated."
            ),
        }

    recommendations = recommendation_result.get(
        "recommendations",
        []
    )

    if not recommendations:
        return {
            "tool": "explain_prediction",
            "status": "no_recommendation",
            "message": (
                "No ranked Chapter 4 recommendation was available "
                "for explanation."
            ),
        }

    # Top-ranked Chapter 4 recommendation
    top_recommendation = recommendations[0]["action"]

    if top_recommendation not in SURROGATE_MODEL_FILES:
        return {
            "tool": "explain_prediction",
            "status": "unsupported_recommendation",
            "recommendation": top_recommendation,
            "message": (
                "No Chapter 5 surrogate model is registered "
                "for this recommendation."
            ),
        }

    # --------------------------------------------------------
    # Load the corresponding Chapter 5 surrogate model
    # --------------------------------------------------------

    model_dir = _get_chapter5_model_dir()

    surrogate_path = (
        model_dir
        / SURROGATE_MODEL_FILES[top_recommendation]
    )

    surrogate_model = joblib.load(
        surrogate_path
    )

    # --------------------------------------------------------
    # Construct Chapter 5 input
    # --------------------------------------------------------

    xai_input = _build_surrogate_input(
        recommendation_result
    )

    expected_features = list(
        surrogate_model.feature_names_in_
    )

    xai_input = xai_input[
        expected_features
    ]

    # --------------------------------------------------------
    # Surrogate recommendation probability
    # --------------------------------------------------------

    surrogate_probability = float(
        surrogate_model.predict_proba(
            xai_input
        )[0, 1]
    )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        surrogate_model
    )

    shap_values = explainer.shap_values(
        xai_input
    )

    shap_array = np.asarray(
        shap_values
    )

    # --------------------------------------------------------
    # Handle binary classifier SHAP output
    #
    # Current environment returns:
    # (observations, features, classes)
    #
    # We explain class 1:
    # recommendation selected / supported.
    # --------------------------------------------------------

    if shap_array.ndim == 3:
        positive_class_shap = (
            shap_array[0, :, 1]
        )

    elif shap_array.ndim == 2:
        positive_class_shap = (
            shap_array[0]
        )

    else:
        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{shap_array.shape}"
        )

    # --------------------------------------------------------
    # Build feature-level explanation
    # --------------------------------------------------------

    driver_details = []

    for feature_name, shap_value in zip(
        expected_features,
        positive_class_shap
    ):
        feature_value = float(
            xai_input.iloc[0][feature_name]
        )

        shap_value = float(
            shap_value
        )

        if shap_value > 0:
            direction = "supports recommendation"

        elif shap_value < 0:
            direction = "opposes recommendation"

        else:
            direction = "neutral"

        driver_details.append(
            {
                "feature": feature_name,
                "feature_value": feature_value,
                "shap_value": shap_value,
                "absolute_shap_value": abs(
                    shap_value
                ),
                "direction": direction,
            }
        )

    # Rank strongest drivers first
    driver_details = sorted(
        driver_details,
        key=lambda x: x[
            "absolute_shap_value"
        ],
        reverse=True,
    )

    # --------------------------------------------------------
    # Return structured Chapter 5 result
    # --------------------------------------------------------

    return {
        "tool": "explain_prediction",
        "status": "success",

        "recommendation_explained":
            top_recommendation,

        "chapter4_recommendation_score":
            float(
                recommendations[0]["score"]
            ),

        "surrogate_predicted_probability":
            surrogate_probability,

        "strategic_regime":
            recommendation_result[
                "strategic_regime"
            ],

        "financial_health_score":
            float(
                recommendation_result[
                    "financial_health_score"
                ]
            ),

        "distress_probability":
            float(
                recommendation_result[
                    "chapter4_distress_probability"
                ]
            ),

        "drivers":
            driver_details,

        "method":
            "Chapter 5 SHAP surrogate explanation",

        "interpretation_note": (
            "SHAP values explain how the Chapter 5 surrogate "
            "model produced its recommendation prediction. "
            "They represent predictive contribution and should "
            "not be interpreted as causal effects."
        ),
    }
