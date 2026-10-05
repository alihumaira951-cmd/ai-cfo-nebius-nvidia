import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from tools.smbgym_env import SMBGymRealDistressEnv


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


ROLLOUT_QUARTERS = 4
ROLLOUT_NOISE = 0.02
ROLLOUT_SEED = 42


# ------------------------------------------------------------
# Locate packaged Chapter 6 artifacts
# ------------------------------------------------------------

def _get_chapter6_paths():
    """
    Locate the deployment-safe Chapter 6 PPO model,
    company-state table, saved training scalers, and
    action registry.
    """

    repo_root = Path(
        __file__
    ).resolve().parents[1]

    artifact_root = (
        repo_root
        / "deployment_artifacts"
    )

    model_path = (
        artifact_root
        / "models"
        / "rl"
        / "ppo_smbgym_real_distress_model.zip"
    )

    company_state_path = (
        artifact_root
        / "data"
        / "chapter6_company_states.parquet"
    )

    scaler_path = (
        artifact_root
        / "data"
        / "chapter6_scalers.json"
    )

    action_registry_path = (
        artifact_root
        / "tables"
        / "chapter6_action_space_registry.csv"
    )

    return (
        model_path,
        company_state_path,
        scaler_path,
        action_registry_path,
    )


# ------------------------------------------------------------
# Find company-quarter
# ------------------------------------------------------------

def _get_company_state(
    company_data: dict,
    company_state_df: pd.DataFrame,
):
    """
    Locate the requested company-quarter in the packaged
    Chapter 6 company-state table, or accept a validated
    custom 24-variable Chapter 6 state.
    """

    # --------------------------------------------------------
    # Custom-company state
    # --------------------------------------------------------

    if "custom_state" in company_data:

        custom_state = company_data["custom_state"]

        missing_state_vars = [
            variable
            for variable in STATE_VARS
            if variable not in custom_state
        ]

        if missing_state_vars:
            return None, {
                "tool": "simulate_strategy",
                "status": "missing_custom_state_variables",
                "missing_fields": missing_state_vars,
                "message": (
                    "The supplied custom Chapter 6 state "
                    "is incomplete."
                ),
            }

        row_data = {
            variable: float(custom_state[variable])
            for variable in STATE_VARS
        }

        row_data.update({
            "cik": str(
                company_data.get(
                    "cik",
                    "CUSTOM",
                )
            ),
            "company_name": company_data.get(
                "company_name",
                "Custom Company",
            ),
            "period": company_data.get(
                "period",
                "2021-03-31",
            ),
            "economic_regime": company_data.get(
                "economic_regime",
                "custom",
            ),
            "economic_regime_code": float(
                custom_state[
                    "economic_regime_code"
                ]
            ),
        })

        return pd.Series(row_data), None

    # --------------------------------------------------------
    # Existing packaged historical-company path
    # --------------------------------------------------------

    missing_fields = [
        field
        for field in [
            "cik",
            "period",
        ]
        if field not in company_data
    ]

    if missing_fields:
        return None, {
            "tool":
                "simulate_strategy",
            "status":
                "missing_identifiers",
            "missing_fields":
                missing_fields,
            "message": (
                "Chapter 6 simulation requires "
                "both cik and period."
            ),
        }

    input_cik = str(
        company_data["cik"]
    )

    input_period = pd.to_datetime(
        company_data["period"]
    ).strftime(
        "%Y-%m-%d"
    )

    panel_cik = (
        company_state_df["cik"]
        .astype(str)
    )

    panel_period = (
        pd.to_datetime(
            company_state_df[
                "period"
            ]
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    match = company_state_df[
        (
            panel_cik
            == input_cik
        )
        & (
            panel_period
            == input_period
        )
    ]

    if match.empty:
        return None, {
            "tool":
                "simulate_strategy",
            "status":
                "not_found",
            "cik":
                input_cik,
            "period":
                input_period,
            "message": (
                "No matching company-quarter was "
                "found in the packaged Chapter 6 "
                "company-state table."
            ),
        }

    return (
        match.iloc[0],
        None,
    )


# ------------------------------------------------------------
# Load saved Chapter 6 training scalers
# ------------------------------------------------------------

def _load_scalers(
    scaler_path: Path
):
    """
    Load the exact means and standard deviations computed
    from the original Chapter 6 training-eligible panel.
    """

    with open(
        scaler_path,
        "r",
    ) as file:

        scalers = json.load(
            file
        )

    means = scalers.get(
        "means",
        {},
    )

    stds = scalers.get(
        "stds",
        {},
    )

    missing_means = [
        var
        for var in STATE_VARS
        if var not in means
    ]

    missing_stds = [
        var
        for var in STATE_VARS
        if var not in stds
    ]

    if (
        missing_means
        or missing_stds
    ):
        raise ValueError(
            "Chapter 6 scaler file is incomplete. "
            f"Missing means: {missing_means}; "
            f"missing stds: {missing_stds}."
        )

    return (
        means,
        stds,
    )


# ------------------------------------------------------------
# Initialize environment at requested company state
# ------------------------------------------------------------

def _initialize_company_environment(
    company_state_df: pd.DataFrame,
    action_df: pd.DataFrame,
    company_row: pd.Series,
    scaler_means: dict,
    scaler_stds: dict,
):
    """
    Build the finalized Chapter 6 environment using the
    saved training-panel normalization statistics, then
    set the initial state to the requested company-quarter.

    This preserves the original PPO observation scaling
    without packaging the full Chapter 6 research panel.
    """

    env = SMBGymRealDistressEnv(
        data=company_state_df,
        state_vars=STATE_VARS,
        action_registry=action_df,
        max_steps=ROLLOUT_QUARTERS,
        stochastic_noise=ROLLOUT_NOISE,
        seed=ROLLOUT_SEED,
        scaler_means=scaler_means,
        scaler_stds=scaler_stds,
    )

    raw_state = {
        variable: float(
            company_row[
                variable
            ]
        )
        for variable in STATE_VARS
    }

    env.current_raw_state = (
        raw_state.copy()
    )

    env.current_info = {
        "cik":
            str(
                company_row.get(
                    "cik"
                )
            ),
        "company_name":
            company_row.get(
                "company_name"
            ),
        "economic_regime":
            company_row.get(
                "economic_regime"
            ),
        "economic_regime_code":
            company_row.get(
                "economic_regime_code"
            ),
    }

    env.step_count = 0

    env.initial_distress = float(
        raw_state[
            "distress_prob_h1"
        ]
    )

    env.baseline_distress_probability = float(
        np.clip(
            env.initial_distress,
            1e-6,
            1.0 - 1e-6,
        )
    )

    env.baseline_distress_logit = (
        env._probability_to_logit(
            env.baseline_distress_probability
        )
    )

    env.current_obs = (
        env._normalize_state(
            env.current_raw_state
        )
    )

    return env


# ------------------------------------------------------------
# Run Chapter 6 PPO simulation
# ------------------------------------------------------------

def simulate_strategy(
    strategy_input: dict
) -> dict:
    """
    Run a four-quarter PPO-driven strategy simulation using
    the finalized Chapter 6 SMBGym real-distress environment.

    The deployment version uses:
    - the trained PPO model,
    - packaged company-quarter states,
    - exact training normalization statistics,
    - the original action registry,
    - the original transition and reward logic.

    Results remain scenario-based decision support and are
    not guaranteed real-world business outcomes.
    """

    (
        model_path,
        company_state_path,
        scaler_path,
        action_registry_path,
    ) = _get_chapter6_paths()

    company_state_df = (
        pd.read_parquet(
            company_state_path
        )
    )

    action_df = (
        pd.read_csv(
            action_registry_path
        )
        .sort_values(
            "action_id"
        )
        .reset_index(
            drop=True
        )
    )

    (
        scaler_means,
        scaler_stds,
    ) = _load_scalers(
        scaler_path
    )

    (
        company_row,
        error_result,
    ) = _get_company_state(
        strategy_input,
        company_state_df,
    )

    if error_result is not None:
        return error_result

    env = (
        _initialize_company_environment(
            company_state_df=
                company_state_df,
            action_df=
                action_df,
            company_row=
                company_row,
            scaler_means=
                scaler_means,
            scaler_stds=
                scaler_stds,
        )
    )

    initial_state = (
        env.current_raw_state.copy()
    )

    ppo_model = PPO.load(
        model_path
    )

    trajectory = []

    terminated = False
    truncated = False

    while (
        not terminated
        and not truncated
    ):

        state_before = (
            env.current_raw_state.copy()
        )

        action, _ = (
            ppo_model.predict(
                env.current_obs,
                deterministic=True,
            )
        )

        action_id = int(
            np.asarray(
                action
            ).item()
        )

        action_match = action_df[
            action_df[
                "action_id"
            ]
            == action_id
        ]

        if action_match.empty:
            raise ValueError(
                f"PPO selected action {action_id}, "
                "but that action is not present "
                "in the Chapter 6 registry."
            )

        selected_action = (
            action_match.iloc[0]
        )

        (
            next_obs,
            reward,
            terminated,
            truncated,
            step_info,
        ) = env.step(
            action_id
        )

        state_after = (
            env.current_raw_state.copy()
        )

        trajectory.append(
            {
                "quarter":
                    int(
                        step_info[
                            "step"
                        ]
                    ),

                "action_id":
                    action_id,

                "strategy":
                    selected_action[
                        "action_description"
                    ],

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

                "reward":
                    float(
                        reward
                    ),

                "profit_margin_before":
                    float(
                        state_before[
                            "profit_margin"
                        ]
                    ),

                "profit_margin_after":
                    float(
                        state_after[
                            "profit_margin"
                        ]
                    ),

                "cash_ratio_before":
                    float(
                        state_before[
                            "cash_ratio"
                        ]
                    ),

                "cash_ratio_after":
                    float(
                        state_after[
                            "cash_ratio"
                        ]
                    ),

                "current_ratio_before":
                    float(
                        state_before[
                            "current_ratio"
                        ]
                    ),

                "current_ratio_after":
                    float(
                        state_after[
                            "current_ratio"
                        ]
                    ),

                "debt_ratio_before":
                    float(
                        state_before[
                            "debt_ratio"
                        ]
                    ),

                "debt_ratio_after":
                    float(
                        state_after[
                            "debt_ratio"
                        ]
                    ),

                "revenue_growth_before":
                    float(
                        state_before[
                            "revenue_growth_qoq"
                        ]
                    ),

                "revenue_growth_after":
                    float(
                        state_after[
                            "revenue_growth_qoq"
                        ]
                    ),

                "distress_probability_before":
                    float(
                        state_before[
                            "distress_prob_h1"
                        ]
                    ),

                "distress_probability_after":
                    float(
                        state_after[
                            "distress_prob_h1"
                        ]
                    ),

                "bankrupt":
                    bool(
                        step_info[
                            "bankrupt"
                        ]
                    ),

                "recovered":
                    bool(
                        step_info[
                            "recovered"
                        ]
                    ),
            }
        )

    final_state = (
        env.current_raw_state.copy()
    )

    total_reward = float(
        sum(
            step[
                "reward"
            ]
            for step in trajectory
        )
    )

    first_action = (
        trajectory[0]
        if trajectory
        else None
    )

    return {
        "tool":
            "simulate_strategy",

        "status":
            "success",

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
            len(
                STATE_VARS
            ),

        "action_space_size":
            int(
                len(
                    action_df
                )
            ),

        "simulation_horizon_quarters":
            ROLLOUT_QUARTERS,

        "initial_selected_action":
            first_action,

        "trajectory":
            trajectory,

        "summary": {
            "quarters_simulated":
                len(
                    trajectory
                ),

            "total_reward":
                total_reward,

            "initial_profit_margin":
                float(
                    initial_state[
                        "profit_margin"
                    ]
                ),

            "final_profit_margin":
                float(
                    final_state[
                        "profit_margin"
                    ]
                ),

            "initial_cash_ratio":
                float(
                    initial_state[
                        "cash_ratio"
                    ]
                ),

            "final_cash_ratio":
                float(
                    final_state[
                        "cash_ratio"
                    ]
                ),

            "initial_debt_ratio":
                float(
                    initial_state[
                        "debt_ratio"
                    ]
                ),

            "final_debt_ratio":
                float(
                    final_state[
                        "debt_ratio"
                    ]
                ),

            "initial_distress_probability":
                float(
                    initial_state[
                        "distress_prob_h1"
                    ]
                ),

            "final_distress_probability":
                float(
                    final_state[
                        "distress_prob_h1"
                    ]
                ),

            "bankruptcy_occurred":
                any(
                    step[
                        "bankrupt"
                    ]
                    for step in trajectory
                ),

            "recovery_occurred":
                any(
                    step[
                        "recovered"
                    ]
                    for step in trajectory
                ),
        },

        "method": (
            "Four-quarter sequential PPO rollout using "
            "the finalized Chapter 6 real-distress "
            "SMBGym environment with the original "
            "training normalization statistics."
        ),

        "interpretation_note": (
            "The trajectory is generated by the Chapter 6 "
            "simulated financial environment using modeled "
            "transition assumptions and stochastic variation. "
            "It is scenario-based decision support and should "
            "not be interpreted as a guaranteed forecast of "
            "real-world financial outcomes."
        ),

        "rollout_status":
            "connected",
    }
