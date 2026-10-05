import numpy as np


FED_HIGH = 0.77
SPREAD_HIGH = 2.20
SENT_LOW = 58.8


REGIME_MAP = {
    "normal_expansion": 0,
    "weak_demand": 1,
    "monetary_tightening": 2,
    "tightening_recession_risk": 3,
    "credit_stress": 4,
}


def sic_to_division(sic_value):
    try:
        sic = int(float(sic_value))
    except Exception:
        return 0

    if 100 <= sic <= 999:
        return 1
    elif 1000 <= sic <= 1499:
        return 2
    elif 1500 <= sic <= 1799:
        return 3
    elif 2000 <= sic <= 3999:
        return 4
    elif 4000 <= sic <= 4999:
        return 5
    elif 5000 <= sic <= 5199:
        return 6
    elif 5200 <= sic <= 5999:
        return 7
    elif 6000 <= sic <= 6799:
        return 8
    elif 7000 <= sic <= 8999:
        return 9
    elif 9100 <= sic <= 9729:
        return 10

    return 0


def classify_economic_regime(
    fed_funds_rate,
    baa_credit_spread,
    consumer_sentiment,
    yield_curve_inverted,
):
    high_rates = fed_funds_rate >= FED_HIGH
    credit_stress = baa_credit_spread >= SPREAD_HIGH
    weak_sentiment = consumer_sentiment <= SENT_LOW
    inverted = bool(yield_curve_inverted == 1)

    if credit_stress and weak_sentiment:
        return "credit_stress"

    if high_rates and (inverted or weak_sentiment):
        return "tightening_recession_risk"

    if high_rates:
        return "monetary_tightening"

    if weak_sentiment:
        return "weak_demand"

    return "normal_expansion"


def safe_ratio(numerator, denominator):
    if denominator is None or denominator == 0:
        return 0.0

    return float(numerator / denominator)


def build_chapter6_state(
    *,
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

    profit_margin = safe_ratio(
        net_income,
        revenue,
    )

    gross_margin = safe_ratio(
        gross_profit,
        revenue,
    )

    ocf_margin = float(
        np.clip(
            safe_ratio(
                operating_cash_flow,
                revenue,
            ),
            -5,
            5,
        )
    )

    if operating_expenses != 0:
        cash_runway = abs(cash) / abs(operating_expenses)
    else:
        opex_proxy = (
            abs(revenue)
            * (
                1
                - np.clip(
                    profit_margin,
                    -1,
                    1,
                )
            )
        )

        cash_runway = (
            abs(cash) / opex_proxy
            if opex_proxy != 0
            else 0.0
        )

    cash_runway = float(
        np.clip(
            cash_runway,
            0,
            20,
        )
    )

    revenue_growth_qoq = float(
        revenue_growth
    )

    revenue_growth_yoy = float(
        revenue_growth
    )

    revenue_rolling_4q_mean = float(
        revenue
    )

    revenue_volatility_4q = float(
        np.clip(
            abs(revenue_growth),
            0,
            5,
        )
    )

    sga_ratio = float(
        np.clip(
            safe_ratio(
                sga_expense,
                revenue,
            ),
            -5,
            5,
        )
    )

    economic_regime = classify_economic_regime(
        fed_funds_rate=fed_funds_rate,
        baa_credit_spread=baa_credit_spread,
        consumer_sentiment=consumer_sentiment,
        yield_curve_inverted=yield_curve_inverted,
    )

    economic_regime_code = REGIME_MAP[
        economic_regime
    ]

    sic_division = sic_to_division(
        sic
    )

    state = {
        "cash_ratio":
            float(cash_ratio),

        "current_ratio":
            float(current_ratio),

        "profit_margin":
            float(profit_margin),

        "debt_ratio":
            float(debt_to_assets),

        "revenue_growth_qoq":
            float(revenue_growth_qoq),

        "gross_margin":
            float(gross_margin),

        "operating_margin":
            float(operating_margin),

        "return_on_assets":
            float(return_on_assets),

        "asset_turnover":
            float(asset_turnover),

        "revenue_growth_yoy":
            float(revenue_growth_yoy),

        "cash_runway_quarters":
            float(cash_runway),

        "ocf_margin":
            float(ocf_margin),

        "debt_to_equity":
            float(debt_to_equity),

        "revenue_rolling_4q_mean":
            float(revenue_rolling_4q_mean),

        "revenue_volatility_4q":
            float(revenue_volatility_4q),

        "sga_ratio":
            float(sga_ratio),

        "distress_prob_h1":
            float(distress_probability),

        "fed_funds_rate":
            float(fed_funds_rate),

        "baa_credit_spread":
            float(baa_credit_spread),

        "consumer_sentiment":
            float(consumer_sentiment),

        "yield_curve_inverted":
            float(yield_curve_inverted),

        "quarter":
            float(quarter),

        "sic_division":
            float(sic_division),

        "economic_regime_code":
            float(economic_regime_code),
    }

    return {
        "state": state,
        "economic_regime": economic_regime,
        "economic_regime_code":
            economic_regime_code,
        "sic_division":
            sic_division,
    }
