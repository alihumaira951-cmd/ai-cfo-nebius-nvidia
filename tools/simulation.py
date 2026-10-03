import os
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


# ------------------------------------------------------------
# Chapter 6 state-variable order
# ------------------------------------------------------------

STATE_VARS = [
    "cash_ratio",
    "current_ratio",
    "profit_margin",
    "debt_ratio",
    "revenue_growth_qoq",
    "gross_margin",
    "operating_margin",
    "return_on_assets",
    "asset_turnover",
    "revenue_growth_yoy",
    "cash_runway_quarters",
    "ocf_margin",
    "debt_to_equity",
    "revenue_rolling_4q_mean",
    "revenue_volatility_4q",
    "sga_ratio",
    "distress_prob_h1",
    "fed_funds_rate",
    "baa_credit_spread",
    "consumer_sentiment",
    "yield_curve_inverted",
    "quarter",
    "sic_division",
    "economic_regime_code",
]


# ------------------------------------------------------------
# Locate Chapter 6 artifacts
# ------------------------------------------------------------

def _get_chapter6_paths():
    """
    Locate the finalized Chapter 6 PPO model, environment panel,
    and governance-aware action registry.
    """

    thesis_root = os.environ.get(
        "AI_CFO_THESIS_ROOT"
    )

    if not thesis_root:
        raise EnvironmentError(
            "AI_CFO_THESIS_ROOT is not set. "
            "Set it to the local Revised_AI_CFO_Thesis folder."
        )

    thesis_root = Path(thesis_root)

    model_path = (
        thesis_root
        / "models"
        / "rl"
        / "ppo_smbgym_real_distress_model.zip"
    )

    panel_path = (
        thesis_root
        / "data"
        / "processed"
        / "chapter6_smbgym_training_eligible_real_distress_panel.parquet"
    )

    action_registry_path = (
        thesis_root
        / "results"
        / "tables"
        / "chapter6"
        / "chapter6_action_space_registry.csv"
    )

    return (
        model_path,
        panel_path,
        action_registry_path,
    )


# ------------------------------------------------------------
# Locate company-quarter in finalized Chapter 6 panel
# ------------------------------------------------------------

def _get_company_state(
    company_data: dict,
    panel_df: pd.DataFrame,
):
    """
    Locate a specific company-quarter in the finalized
    Chapter 6 real-distress environment panel.
    """

    if "cik" not in company_data:
        return None, {
            "tool": "simulate_strategy",
            "status": "missing_identifiers",
            "missing_fields": ["cik"],
        }

    if "period" not in company_data:
        return None, {
            "tool": "simulate_strategy",
            "status": "missing_identifiers",
            "missing_fields": ["period"],
        }

    input_cik = str(
        company_data["cik"]
    )

    input_period = pd.to_datetime(
        company_data["period"]
    ).strftime("%Y-%m-%d")

    working_df = panel_df.copy()

    working_df["cik"] = (
        working_df["cik"]
        .astype(str)
    )

    working_df["period"] = (
        pd.to_datetime(
            working_df["period"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    match = working_df[
        (working_df["cik"] == input_cik)
        & (
            working_df["period"]
            == input_period
        )
    ]

    if match.empty:
        return None, {
            "tool": "simulate_strategy",
            "status": "not_found",
            "cik": input_cik,
            "period": input_period,
            "message": (
                "No matching company-quarter was found "
                "in the finalized Chapter 6 real-distress panel."
            ),
        }

    return match.iloc[0], None


# ------------------------------------------------------------
# Normalize Chapter 6 state
# ------------------------------------------------------------

def _build_normalized_observation(
    company_row: pd.Series,
    panel_df: pd.DataFrame,
):
    """
    Reproduce the Chapter 6 environment normalization.

    Each state variable is standardized using the mean and
    standard deviation of the finalized Chapter 6 panel and
    clipped to the PPO observation range [-3, 3].
    """

    means = (
        panel_df[STATE_VARS]
        .mean()
    )

    stds = (
        panel_df[STATE_VARS]
        .std()
        .clip(lower=1e-6)
    )

    raw_state = (
        company_row[STATE_VARS]
        .astype(float)
    )

    normalized_state = (
        (raw_state - means)
        / stds
    ).clip(
        -3.0,
        3.0,
    )

    observation = (
        normalized_state
        .to_numpy(
            dtype=np.float32
        )
    )

    return (
        raw_state,
        normalized_state,
        observation,
    )


# ------------------------------------------------------------
# Chapter 6 PPO strategy selection
# ------------------------------------------------------------

def simulate_strategy(
    strategy_input: dict
) -> dict:
    """
    Use the finalized Chapter 6 PPO policy to select a
    governance-aware financial strategy for a company-quarter.

    The policy operates on the 24-variable state representation
    used in the dissertation's SMBGym real-distress environment.

    This function currently performs PPO policy inference for the
    starting state. Multi-quarter stochastic rollout simulation
    will be added separately.

    Results are model-based decision support and should not be
    interpreted as guaranteed real-world business outcomes.
    """

    (
        model_path,
        panel_path,
        action_registry_path,
    ) = _get_chapter6_paths()

    # --------------------------------------------------------
    # Load finalized Chapter 6 artifacts
    # --------------------------------------------------------

    panel_df = pd.read_parquet(
        panel_path
    )

    action_df = pd.read_csv(
        action_registry_path
    )

    # --------------------------------------------------------
    # Find company-quarter
    # --------------------------------------------------------

    company_row, error_result = (
        _get_company_state(
            strategy_input,
            panel_df,
        )
    )

    if error_result is not None:
        return error_result

    # --------------------------------------------------------
    # Build PPO observation
    # --------------------------------------------------------

    (
        raw_state,
        normalized_state,
        observation,
    ) = _build_normalized_observation(
        company_row,
        panel_df,
    )

    if observation.shape != (24,):
        raise ValueError(
            "Chapter 6 PPO observation must contain "
            f"24 variables. Received {observation.shape}."
        )

    # --------------------------------------------------------
    # Load trained PPO policy
    # --------------------------------------------------------

    ppo_model = PPO.load(
        model_path
    )

    # --------------------------------------------------------
    # Select deterministic PPO action
    # --------------------------------------------------------

    action, _ = ppo_model.predict(
        observation,
        deterministic=True,
    )

    action_id = int(
        np.asarray(action).item()
    )

    # --------------------------------------------------------
    # Decode action through Chapter 6 registry
    # --------------------------------------------------------

    action_match = action_df[
        action_df["action_id"]
        == action_id
    ]

    if action_match.empty:
        raise ValueError(
            f"PPO selected action {action_id}, "
            "but it does not exist in the Chapter 6 "
            "action registry."
        )

    selected_action = (
        action_match.iloc[0]
    )

    # --------------------------------------------------------
    # Return structured Chapter 6 result
    # --------------------------------------------------------

    return {
        "tool": "simulate_strategy",
        "status": "success",

        "company_name":
            company_row.get(
                "company_name"
            ),

        "cik":
            str(
                company_row.get(
                    "cik"
                )
            ),

        "period":
            pd.to_datetime(
                company_row.get(
                    "period"
                )
            ).strftime(
                "%Y-%m-%d"
            ),

        "policy":
            "PPO",

        "environment":
            "SMBGymRealDistressEnv",

        "state_dimension":
            len(STATE_VARS),

        "action_space_size":
            int(
                len(action_df)
            ),

        "selected_action_id":
            action_id,

        "selected_strategy": {
            "cost_control":
                int(
                    selected_action[
                        "cost_control"
                    ]
                ),

            "pricing_strategy":
                int(
                    selected_action[
                        "pricing_strategy"
                    ]
                ),

            "hiring_strategy":
                int(
                    selected_action[
                        "hiring_strategy"
                    ]
                ),

            "debt_strategy":
                int(
                    selected_action[
                        "debt_strategy"
                    ]
                ),

            "investment_strategy":
                int(
                    selected_action[
                        "investment_strategy"
                    ]
                ),

            "governance_profile":
                selected_action[
                    "governance_profile"
                ],

            "governance_penalty":
                float(
                    selected_action[
                        "governance_penalty"
                    ]
                ),

            "description":
                selected_action[
                    "action_description"
                ],
        },

        "initial_state": {
            variable: float(
                raw_state[variable]
            )
            for variable in STATE_VARS
        },

        "method":
            (
                "Chapter 6 PPO policy inference "
                "using the finalized real-distress "
                "SMBGym state representation"
            ),

        "interpretation_note": (
            "The PPO-selected strategy is generated within "
            "the Chapter 6 simulated financial environment. "
            "It represents model-based scenario decision "
            "support and is not a guaranteed real-world outcome."
        ),

        "rollout_status":
            "not_yet_connected",
    }
      
