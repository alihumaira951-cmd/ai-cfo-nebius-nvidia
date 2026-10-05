import numpy as np
import pandas as pd


def normalize_score(value, low, high):
    if pd.isna(value):
        return 0.5

    if value <= low:
        return 0.0

    if value >= high:
        return 1.0

    return (value - low) / (high - low)


def compute_financial_health_score(row):

    liquidity = np.mean([
        normalize_score(row["current_ratio"], 0.5, 2.0),
        normalize_score(row["cash_ratio"], 0.05, 0.50),
        normalize_score(row["ocf_to_assets"], -0.05, 0.15),
    ])

    leverage = np.mean([
        1 - normalize_score(row["debt_to_assets"], 0.20, 0.80),
        1 - normalize_score(row["debt_to_equity"], 0.20, 2.50),
        normalize_score(row["ocf_to_debt"], 0.00, 0.30),
    ])

    profitability = np.mean([
        normalize_score(row["net_margin"], -0.10, 0.20),
        normalize_score(row["operating_margin"], -0.10, 0.25),
        normalize_score(row["return_on_assets"], -0.05, 0.15),
        normalize_score(row["return_on_equity"], -0.10, 0.25),
    ])

    growth = np.mean([
        normalize_score(row["revenue_growth"], -0.20, 0.30),
        normalize_score(row["asset_growth"], -0.20, 0.30),
        normalize_score(row["cash_growth"], -0.30, 0.50),
    ])

    efficiency = np.mean([
        normalize_score(row["asset_turnover"], 0.20, 2.00),
        normalize_score(row["inventory_turnover"], 1.00, 12.00),
    ])

    overall = np.mean([
        liquidity,
        leverage,
        profitability,
        growth,
        efficiency,
    ])

    return {
        "Liquidity_Score": float(liquidity),
        "Leverage_Score": float(leverage),
        "Profitability_Score": float(profitability),
        "Growth_Score": float(growth),
        "Efficiency_Score": float(efficiency),
        "Financial_Health_Score": float(overall),
    }


def classify_regime(distress, health):

    if distress < 0.05 and health >= 0.35:
        return "Growth"

    if distress >= 0.20:
        return "Defensive"

    return "Balanced"


def compute_action_scores_v3(row):

    health = row["Financial_Health_Score"]
    liquidity = row["Liquidity_Score"]
    leverage = row["Leverage_Score"]
    profitability = row["Profitability_Score"]
    growth = row["Growth_Score"]
    efficiency = row["Efficiency_Score"]

    distress = row["Distress_Probability"]

    forecast_ocf = row["Forecast_OCF"]
    forecast_ebitda = row["Forecast_EBITDA"]

    regime = row["Strategic_Regime"]

    ocf_negative = 1 if forecast_ocf < 0 else 0
    ebitda_negative = 1 if forecast_ebitda < 0 else 0

    defensive_pressure = (
        0.50 * distress
        + 0.25 * ocf_negative
        + 0.25 * ebitda_negative
    )

    growth_capacity = (
        0.40 * health
        + 0.30 * profitability
        + 0.30 * (1 - distress)
    )

    scores = {}

    if regime == "Growth":

        scores["Hold Strategy"] = (
            0.65 * growth_capacity
            + 0.35 * health
        )

        scores["Increase Revenue Growth Investment"] = (
            0.55 * growth_capacity
            + 0.25 * liquidity
            + 0.20 * (1 - growth)
        )

        scores["Improve Working Capital Efficiency"] = (
            0.40 * (1 - efficiency)
            + 0.20 * health
            + 0.40 * (1 - distress)
        )

        scores["Reduce Operating Expenses"] = 0.15 * defensive_pressure
        scores["Reduce SG&A Expenses"] = 0.15 * defensive_pressure
        scores["Delay Capital Expenditures"] = 0.15 * defensive_pressure
        scores["Increase Cash Reserves"] = 0.15 * defensive_pressure
        scores["Accelerate Receivables Collection"] = 0.15 * defensive_pressure
        scores["Reduce Debt Exposure"] = 0.15 * defensive_pressure
        scores["Refinance Debt"] = 0.15 * defensive_pressure

    elif regime == "Defensive":

        scores["Reduce Operating Expenses"] = (
            0.50 * defensive_pressure
            + 0.30 * (1 - profitability)
            + 0.20 * (1 - health)
        )

        scores["Reduce SG&A Expenses"] = (
            0.50 * defensive_pressure
            + 0.30 * (1 - profitability)
            + 0.20 * (1 - efficiency)
        )

        scores["Increase Cash Reserves"] = (
            0.60 * defensive_pressure
            + 0.25 * (1 - liquidity)
            + 0.15 * (1 - health)
        )

        scores["Reduce Debt Exposure"] = (
            0.50 * defensive_pressure
            + 0.35 * (1 - leverage)
            + 0.15 * (1 - health)
        )

        scores["Delay Capital Expenditures"] = (
            0.50 * defensive_pressure
            + 0.30 * (1 - liquidity)
            + 0.20 * (1 - health)
        )

        scores["Refinance Debt"] = (
            0.40 * defensive_pressure
            + 0.40 * (1 - leverage)
            + 0.20 * (1 - liquidity)
        )

        scores["Accelerate Receivables Collection"] = (
            0.50 * defensive_pressure
            + 0.25 * (1 - liquidity)
            + 0.25 * (1 - efficiency)
        )

        scores["Hold Strategy"] = 0.10 * growth_capacity
        scores["Increase Revenue Growth Investment"] = 0.10 * growth_capacity

        scores["Improve Working Capital Efficiency"] = (
            0.20 * (1 - efficiency)
        )

    else:

        scores["Hold Strategy"] = (
            0.60 * growth_capacity
            + 0.40 * health
        )

        scores["Increase Revenue Growth Investment"] = (
            0.50 * growth_capacity
            + 0.20 * liquidity
            + 0.30 * (1 - growth)
        )

        scores["Improve Working Capital Efficiency"] = (
            0.35 * (1 - efficiency)
            + 0.25 * (1 - liquidity)
            + 0.20 * health
            + 0.20 * (1 - distress)
        )

        scores["Reduce Operating Expenses"] = (
            0.50 * defensive_pressure
            + 0.25 * (1 - profitability)
            + 0.25 * (1 - health)
        )

        scores["Reduce SG&A Expenses"] = (
            0.50 * defensive_pressure
            + 0.30 * (1 - profitability)
            + 0.20 * (1 - efficiency)
        )

        scores["Delay Capital Expenditures"] = (
            0.50 * defensive_pressure
            + 0.30 * (1 - liquidity)
            + 0.20 * (1 - health)
        )

        scores["Increase Cash Reserves"] = (
            0.60 * defensive_pressure
            + 0.25 * (1 - liquidity)
            + 0.15 * (1 - health)
        )

        scores["Accelerate Receivables Collection"] = (
            0.50 * defensive_pressure
            + 0.25 * (1 - liquidity)
            + 0.25 * (1 - efficiency)
        )

        scores["Reduce Debt Exposure"] = (
            0.50 * defensive_pressure
            + 0.35 * (1 - leverage)
            + 0.15 * (1 - health)
        )

        scores["Refinance Debt"] = (
            0.40 * defensive_pressure
            + 0.40 * (1 - leverage)
            + 0.20 * (1 - liquidity)
        )

    return scores


def rank_recommendations(row, top_n=3):

    scores = compute_action_scores_v3(row)

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]
