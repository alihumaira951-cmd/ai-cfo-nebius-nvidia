from tools.forecasting import predict_financials
from tools.recommendations import recommend_actions
from tools.explainability import explain_prediction
from tools.simulation import simulate_strategy


def run_cfo_analysis(company_data: dict) -> dict:
    """
    Simple orchestration layer for the AI CFO system.

    This function demonstrates how the future Nemotron-powered agent
    will coordinate multiple financial decision-support tools.
    """

    forecast_result = predict_financials(company_data)

    recommendation_result = recommend_actions(company_data)

    explanation_result = explain_prediction(forecast_result)

    simulation_result = simulate_strategy(company_data)

    return {
        "forecast": forecast_result,
        "recommendations": recommendation_result,
        "explanation": explanation_result,
        "simulation": simulation_result
    }
