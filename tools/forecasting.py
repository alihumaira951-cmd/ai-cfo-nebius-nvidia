from pathlib import Path

import joblib
import pandas as pd


DISTRESS_THRESHOLD = 0.77


def _get_artifact_path():
    """
    Locate the packaged Chapter 3 XGBoost distress model.
    """

    repo_root = Path(__file__).resolve().parents[1]

    return (
        repo_root
        / "deployment_artifacts"
        / "models"
        / "xgb_distress_h1.joblib"
    )


def predict_financial_distress(company_features: dict) -> dict:
    """
    Predict one-quarter-ahead financial distress using the
    Chapter 3 XGBoost distress classifier.

    The exact feature order is read directly from the trained
    model to preserve the original Chapter 3 input specification.
    """

    model_path = _get_artifact_path()

    model = joblib.load(
        model_path
    )

    feature_names = list(
        model.feature_names_in_
    )

    missing_features = [
        feature
        for feature in feature_names
        if feature not in company_features
    ]

    if missing_features:
        return {
            "tool": "predict_financial_distress",
            "status": "missing_features",
            "missing_feature_count": len(
                missing_features
            ),
            "missing_features": missing_features,
        }

    model_input = pd.DataFrame(
        [[
            company_features[feature]
            for feature in feature_names
        ]],
        columns=feature_names,
    )

    distress_probability = float(
        model.predict_proba(
            model_input
        )[0, 1]
    )

    distress_prediction = int(
        distress_probability
        >= DISTRESS_THRESHOLD
    )

    return {
        "tool": "predict_financial_distress",
        "status": "success",
        "horizon": "H1",
        "distress_probability":
            distress_probability,
        "classification_threshold":
            DISTRESS_THRESHOLD,
        "predicted_distress":
            distress_prediction,
        "feature_count":
            len(feature_names),
    }


def predict_financials(
    company_data: dict
) -> dict:
    """
    Main forecasting interface used by the AI CFO agent.
    """

    distress_result = (
        predict_financial_distress(
            company_data
        )
    )

    return {
        "tool": "predict_financials",
        "status":
            distress_result.get(
                "status"
            ),
        "financial_distress":
            distress_result,
    }
