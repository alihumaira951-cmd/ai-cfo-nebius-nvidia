import os
from pathlib import Path

import pandas as pd


def _get_recommendation_path():
    """
    Locate the Chapter 4 final recommendation artifact.

    Set AI_CFO_THESIS_ROOT to the local dissertation project root.
    """

    thesis_root = os.environ.get("AI_CFO_THESIS_ROOT")

    if not thesis_root:
        raise EnvironmentError(
            "AI_CFO_THESIS_ROOT is not set. "
            "Set it to the local Revised_AI_CFO_Thesis folder."
        )

    thesis_root = Path(thesis_root)

    recommendation_path = (
        thesis_root
        / "results"
        / "tables"
        / "chapter4"
        / "chapter4_final_recommendations.csv"
    )

    return recommendation_path


def recommend_actions(financial_state: dict) -> dict:
    """
    Return governance-aware Chapter 4 recommendations
    for a specific company-quarter.

    Expected identifiers in financial_state:
    - cik
    - period

    The function looks up the corresponding Chapter 4
    recommendation record and returns the strategic regime,
    financial health score, distress probability, and top
    three ranked executive recommendations.
    """

    recommendation_path = _get_recommendation_path()

    recommendations_df = pd.read_csv(recommendation_path)

    required_fields = ["cik", "period"]

    missing_fields = [
        field
        for field in required_fields
        if field not in financial_state
    ]

    if missing_fields:
        return {
            "tool": "recommend_actions",
            "status": "missing_identifiers",
            "missing_fields": missing_fields,
            "message": (
                "Chapter 4 recommendation lookup requires "
                "both cik and period."
            ),
        }

    input_cik = str(financial_state["cik"])
    input_period = pd.to_datetime(
        financial_state["period"]
    ).strftime("%Y-%m-%d")

    recommendations_df["cik"] = (
        recommendations_df["cik"].astype(str)
    )

    recommendations_df["period"] = pd.to_datetime(
        recommendations_df["period"]
    ).dt.strftime("%Y-%m-%d")

    match = recommendations_df[
        (recommendations_df["cik"] == input_cik)
        & (recommendations_df["period"] == input_period)
    ]

    if match.empty:
        return {
            "tool": "recommend_actions",
            "status": "not_found",
            "cik": input_cik,
            "period": input_period,
            "message": (
                "No Chapter 4 recommendation record was found "
                "for this company-quarter."
            ),
        }

    row = match.iloc[0]

    return {
        "tool": "recommend_actions",
        "status": "success",
        "company_name": row["company_name"],
        "cik": row["cik"],
        "period": row["period"],
        "strategic_regime": row["Strategic_Regime"],
        "financial_health_score": float(
            row["Financial_Health_Score"]
        ),
        "chapter4_distress_probability": float(
            row["Distress_Probability"]
        ),
        "recommendations": [
            {
                "rank": 1,
                "action": row["Recommendation_1"],
                "score": float(
                    row["Recommendation_1_Score"]
                ),
            },
            {
                "rank": 2,
                "action": row["Recommendation_2"],
                "score": float(
                    row["Recommendation_2_Score"]
                ),
            },
            {
                "rank": 3,
                "action": row["Recommendation_3"],
                "score": float(
                    row["Recommendation_3_Score"]
                ),
            },
        ],
    }
