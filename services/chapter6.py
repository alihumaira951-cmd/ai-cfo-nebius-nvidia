from services.chapter6_state import build_chapter6_state
from tools.simulation import simulate_strategy


def run_custom_company_simulation(
    *,
    company_name,
    period,
    revenue,
    gross_profit,
    operating_expenses,
    operating_income,
    net_income,
    operating_cash_flow,
    sga_expense,
    cash,
    current_assets,
    current_liabilities,
    total_assets,
    total_debt,
    equity,
    revenue_growth,
    current_ratio,
    cash_ratio,
    debt_to_assets,
    debt_to_equity,
    return_on_assets,
    asset_turnover,
    operating_margin,
    distress_probability,
    fed_funds_rate,
    baa_credit_spread,
    consumer_sentiment,
    yield_curve_inverted,
    quarter,
    sic,
):
    """
    Build the 24-variable Chapter 6 state for a new company
    and run the trained PPO policy for a four-quarter
    scenario simulation.

    This is scenario-based decision support, not a guaranteed
    forecast of real-world outcomes.
    """

    state_result = build_chapter6_state(
        revenue=revenue,
        gross_profit=gross_profit,
        operating_expenses=operating_expenses,
        operating_income=operating_income,
        net_income=net_income,
        operating_cash_flow=operating_cash_flow,
        sga_expense=sga_expense,
        cash=cash,
        current_assets=current_assets,
        current_liabilities=current_liabilities,
        total_assets=total_assets,
        total_debt=total_debt,
        equity=equity,
        revenue_growth=revenue_growth,
        current_ratio=current_ratio,
        cash_ratio=cash_ratio,
        debt_to_assets=debt_to_assets,
        debt_to_equity=debt_to_equity,
        return_on_assets=return_on_assets,
        asset_turnover=asset_turnover,
        operating_margin=operating_margin,
        distress_probability=distress_probability,
        fed_funds_rate=fed_funds_rate,
        baa_credit_spread=baa_credit_spread,
        consumer_sentiment=consumer_sentiment,
        yield_curve_inverted=yield_curve_inverted,
        quarter=quarter,
        sic=sic,
    )

    simulation_result = simulate_strategy({
        "company_name": company_name,
        "period": period,
        "economic_regime": state_result["economic_regime"],
        "custom_state": state_result["state"],
    })

    simulation_result["initial_economic_regime"] = (
        state_result["economic_regime"]
    )

    simulation_result["initial_economic_regime_code"] = (
        state_result["economic_regime_code"]
    )

    simulation_result["sic_division"] = (
        state_result["sic_division"]
    )

    simulation_result["custom_state_method"] = (
        "Chapter 6 state constructed from user financial inputs, "
        "Chapter 3 distress probability, quarter-specific macroeconomic "
        "conditions, and the documented Chapter 6 proxy rules used when "
        "longitudinal variables are unavailable."
    )

    return simulation_result
