import os
from pathlib import Path

import joblib
import pandas as pd


DISTRESS_THRESHOLD = 0.77


def _get_artifact_paths():
    """
    Locate the existing Chapter 3 research artifacts.

    Set AI_CFO_THESIS_ROOT to the local root folder containing
    the dissertation models and results folders.
    """

    thesis_root = os.environ.get("AI_CFO_THESIS_ROOT")

    if not thesis_root:
        raise EnvironmentError(
            "AI_CFO_THESIS_ROOT is not set. "
            "Set it to the local Revised_AI_CFO_Thesis folder."
        )

    thesis_root = Path(thesis_root)

    model_path = thesis_root / "models" / "xgb_distress_h1.joblib"

    feature_registry_path = (
        thesis_root
        / "results"
        / "tables"
        / "chapter3_final_feature_registry.csv"
    )

    return model_path, feature_registry_path


def predict_financial_distress(company_features: dict) -> dict:
    """
    Predict one-quarter-ahead financial distress using the
    Chapter 3 XGBoost distress classifier.

    The model expects the exact 88-feature registry used during
    Chapter 3 model development.
    """

    model_path, feature_registry_path = _get_artifact_paths()

    model = joblib.load(model_path)

    registry_df = pd.read_csv(feature_registry_path)
    registry_features = registry_df["feature"].tolist()

    # Use the exact feature order stored in the trained XGBoost model.
    feature_names = list(model.feature_names_in_)

    # Safety check:
    # The saved registry and trained model must contain
    # the same predictors.
    if set(feature_names) != set(registry_features):
        raise ValueError(
            "Feature registry does not match the features stored "
            "in the trained model."
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
            "missing_feature_count": len(missing_features),
            "missing_features": missing_features,
        }

    model_input = pd.DataFrame(
        [[company_features[feature] for feature in feature_names]],
        columns=feature_names,
    )

    distress_probability = float(
        model.predict_proba(model_input)[0, 1]
    )

    distress_prediction = int(
        distress_probability >= DISTRESS_THRESHOLD
    )

    return {
        "tool": "predict_financial_distress",
        "status": "success",
        "horizon": "H1",
        "distress_probability": distress_probability,
        "classification_threshold": DISTRESS_THRESHOLD,
        "predicted_distress": distress_prediction,
        "feature_count": len(feature_names),
    }


def predict_financials(company_data: dict) -> dict:
    """
    Main forecasting interface used by the AI CFO agent.

    Additional Chapter 3 revenue, operating cash flow,
    and EBITDA forecasts will be added here after
    distress-model integration is verified.
    """

    distress_result = predict_financial_distress(company_data)

    return {
        "tool": "predict_financials",
        "status": distress_result.get("status"),
        "financial_distress": distress_result,
    }
