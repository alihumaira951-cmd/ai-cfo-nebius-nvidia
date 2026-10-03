from tools.forecasting import predict_financials
from tools.recommendations import recommend_actions
from tools.explainability import explain_prediction
from tools.simulation import simulate_strategy

from services.nebius import (
    nebius_available,
    generate_nebius_response,
)


def _build_executive_prompt(
    company_data: dict,
    forecast_result: dict,
    recommendation_result: dict,
    explanation_result: dict,
    simulation_result: dict,
) -> list:
    """
    Build the evidence package that will be sent to
    NVIDIA Nemotron through Nebius Token Factory.

    The model is instructed to summarize the tool outputs
    without inventing financial values or causal claims.
    """

    system_message = {
        "role": "system",
        "content": (
            "You are the executive reasoning layer for AI CFO. "
            "Use only the quantitative results provided by the "
            "AI CFO tools. Do not invent numbers, forecasts, "
            "recommendations, or causal claims. "
            "Clearly distinguish predictive results, SHAP "
            "explanations, governance-aware recommendations, "
            "and simulated PPO scenario outcomes. "
            "Present the result as concise executive "
            "decision-support, not as guaranteed financial advice."
        ),
    }

    user_message = {
        "role": "user",
        "content": (
            "Create an executive AI CFO decision brief from the "
            "following tool outputs.\n\n"
            f"Company identifiers:\n"
            f"{company_data}\n\n"
            f"Chapter 3 forecast:\n"
            f"{forecast_result}\n\n"
            f"Chapter 4 recommendations:\n"
            f"{recommendation_result}\n\n"
            f"Chapter 5 SHAP explanation:\n"
            f"{explanation_result}\n\n"
            f"Chapter 6 PPO simulation:\n"
            f"{simulation_result}\n\n"
            "The brief should explain:\n"
            "1. Current financial risk outlook\n"
            "2. Top strategic recommendation\n"
            "3. Why that recommendation was selected\n"
            "4. What the PPO simulation suggests over the next "
            "four quarters\n"
            "5. Important governance or model limitations\n"
        ),
    }

    return [
        system_message,
        user_message,
    ]


def _build_local_brief(
    forecast_result: dict,
    recommendation_result: dict,
    explanation_result: dict,
    simulation_result: dict,
) -> dict:
    """
    Provide a structured local fallback when Nebius credentials
    are not yet available.
    """

    distress = (
        forecast_result
        .get("financial_distress", {})
    )

    recommendations = (
        recommendation_result
        .get("recommendations", [])
    )

    top_recommendation = (
        recommendations[0]
        if recommendations
        else None
    )

    drivers = (
        explanation_result
        .get("drivers", [])
    )

    strongest_driver = (
        drivers[0]
        if drivers
        else None
    )

    simulation_summary = (
        simulation_result
        .get("summary", {})
    )

    return {
        "status": "local_fallback",
        "reason": (
            "Nebius Token Factory is not configured yet."
        ),
        "risk_outlook": {
            "distress_probability":
                distress.get(
                    "distress_probability"
                ),
            "predicted_distress":
                distress.get(
                    "predicted_distress"
                ),
            "classification_threshold":
                distress.get(
                    "classification_threshold"
                ),
        },
        "top_recommendation":
            top_recommendation,
        "strongest_explanation_driver":
            strongest_driver,
        "simulation_summary":
            simulation_summary,
        "note": (
            "This is a structured local summary of the "
            "underlying AI CFO tools. Once Nebius access is "
            "configured, NVIDIA Nemotron will generate the "
            "executive narrative from the same evidence."
        ),
    }


def run_cfo_analysis(
    company_data: dict
) -> dict:
    """
    Run the complete AI CFO workflow.

    Current architecture:

    Chapter 3
        Predictive financial modeling

    Chapter 4
        Governance-aware recommendations

    Chapter 5
        SHAP recommendation explanation

    Chapter 6
        PPO strategy simulation

    Nemotron / Nebius
        Executive reasoning and synthesis layer
    """

    # --------------------------------------------------------
    # Chapter 3
    # --------------------------------------------------------

    forecast_result = predict_financials(
        company_data
    )

    # --------------------------------------------------------
    # Chapter 4
    # --------------------------------------------------------

    recommendation_result = recommend_actions(
        company_data
    )

    # --------------------------------------------------------
    # Chapter 5
    # --------------------------------------------------------

    explanation_result = explain_prediction(
        model_output=forecast_result,
        recommendation_result=recommendation_result,
    )

    # --------------------------------------------------------
    # Chapter 6
    # --------------------------------------------------------

    simulation_result = simulate_strategy(
        company_data
    )

    # --------------------------------------------------------
    # Nemotron executive reasoning layer
    # --------------------------------------------------------

    if nebius_available():

        messages = _build_executive_prompt(
            company_data=company_data,
            forecast_result=forecast_result,
            recommendation_result=recommendation_result,
            explanation_result=explanation_result,
            simulation_result=simulation_result,
        )

        executive_brief = (
            generate_nebius_response(
                messages
            )
        )

    else:

        executive_brief = (
            _build_local_brief(
                forecast_result=forecast_result,
                recommendation_result=recommendation_result,
                explanation_result=explanation_result,
                simulation_result=simulation_result,
            )
        )

    # --------------------------------------------------------
    # Combined AI CFO result
    # --------------------------------------------------------

    return {
        "forecast":
            forecast_result,

        "recommendations":
            recommendation_result,

        "explanation":
            explanation_result,

        "simulation":
            simulation_result,

        "executive_brief":
            executive_brief,
    }


