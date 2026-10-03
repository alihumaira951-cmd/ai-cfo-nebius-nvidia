from tools.forecasting import predict_financials
from tools.recommendations import recommend_actions
from tools.explainability import explain_prediction
from tools.simulation import simulate_strategy


def run_cfo_analysis(company_data: dict) -> dict:
    """
    Orchestration layer for the AI CFO system.

    The current workflow connects:
    - Chapter 3 predictive modeling
    - Chapter 4 governance-aware recommendations
    - Chapter 5 SHAP-based explainability
    - Chapter 6 simulation placeholder
    """

    # --------------------------------------------------------
    # Chapter 3: Forecasting / financial distress
    # --------------------------------------------------------

    forecast_result = predict_financials(
        company_data
    )

    # --------------------------------------------------------
    # Chapter 4: Governance-aware recommendations
    # --------------------------------------------------------

    recommendation_result = recommend_actions(
        company_data
    )

    # --------------------------------------------------------
    # Chapter 5: Explain the top Chapter 4 recommendation
    # --------------------------------------------------------

    explanation_result = explain_prediction(
        model_output=forecast_result,
        recommendation_result=recommendation_result,
    )

    # --------------------------------------------------------
    # Chapter 6: Strategy simulation
    # --------------------------------------------------------

    simulation_result = simulate_strategy(
        company_data
    )

    # --------------------------------------------------------
    # Combined AI CFO output
    # --------------------------------------------------------

    return {
        "forecast": forecast_result,
        "recommendations": recommendation_result,
        "explanation": explanation_result,
        "simulation": simulation_result,
    }
