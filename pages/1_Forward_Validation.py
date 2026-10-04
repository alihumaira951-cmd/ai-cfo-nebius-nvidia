
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI CFO | Forward Validation",
    page_icon="🧪",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    REPO_ROOT
    / "hackathon_validation"
    / "results"
)

WALK_FORWARD_PATH = (
    RESULTS_DIR
    / "technical_communications_walk_forward_validation.csv"
)

DECISION_PATH = (
    RESULTS_DIR
    / "technical_communications_march2022_decision.csv"
)

XAI_PATH = (
    RESULTS_DIR
    / "technical_communications_march2022_xai.csv"
)

SIMULATION_PATH = (
    RESULTS_DIR
    / "technical_communications_march2022_simulation_summary.csv"
)

TRAJECTORY_PATH = (
    RESULTS_DIR
    / "technical_communications_march2022_simulation_trajectory.csv"
)


# ============================================================
# STYLING
# ============================================================

st.html(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }

    .validation-hero {
        padding: 2.2rem 2.4rem;
        border-radius: 18px;
        border: 1px solid rgba(120, 120, 120, 0.22);
        background: linear-gradient(
            135deg,
            rgba(28, 31, 38, 0.97),
            rgba(45, 52, 65, 0.94)
        );
        margin-bottom: 1.4rem;
    }

    .validation-title {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: white;
        margin-bottom: 0.35rem;
    }

    .validation-subtitle {
        color: #d8dce5;
        font-size: 1.12rem;
        line-height: 1.6;
    }

    .section-kicker {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        font-weight: 700;
        opacity: 0.65;
        margin-bottom: 0.2rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-radius: 14px;
        padding: 1rem;
        min-height: 115px;
    }

    </style>
    """
)


# ============================================================
# LOAD RESULTS
# ============================================================

required_files = [
    WALK_FORWARD_PATH,
    DECISION_PATH,
    XAI_PATH,
    SIMULATION_PATH,
    TRAJECTORY_PATH,
]

missing_files = [
    path
    for path in required_files
    if not path.exists()
]

if missing_files:

    st.error(
        "Forward-validation artifacts are missing."
    )

    for path in missing_files:
        st.code(str(path))

    st.stop()


walk_forward = pd.read_csv(
    WALK_FORWARD_PATH
)

decision = pd.read_csv(
    DECISION_PATH
).iloc[0]

xai = pd.read_csv(
    XAI_PATH
)

simulation = pd.read_csv(
    SIMULATION_PATH
).iloc[0]

trajectory = pd.read_csv(
    TRAJECTORY_PATH
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="validation-hero">
        <div class="validation-title">
            AI CFO Forward Validation
        </div>

        <div class="validation-subtitle">
            An out-of-time case study demonstrating how a
            previously unused financial state moves through the
            AI CFO pipeline:
            predict → recommend → explain → simulate → compare
            with the subsequently observed financial outcome.
        </div>
    </div>
    """
)

st.caption(
    "Illustrative validation case: Technical Communications Corporation "
    "(CIK 96699). This page evaluates the frozen research pipeline on "
    "post-training financial states. Scenario simulations are decision "
    "support, not guaranteed forecasts."
)


# ============================================================
# CASE STUDY OVERVIEW
# ============================================================

st.html(
    '<div class="section-kicker">'
    'Forward Case Study'
    '</div>'
)

st.header(
    "March 31, 2022 → June 30, 2022"
)

overview_col1, overview_col2, overview_col3, overview_col4 = (
    st.columns(4)
)

with overview_col1:

    st.metric(
        "Next-Quarter Distress Risk",
        f"{decision['distress_probability']:.2%}",
    )

with overview_col2:

    st.metric(
        "Financial Health Score",
        f"{decision['financial_health_score']:.3f}",
    )

with overview_col3:

    st.metric(
        "Strategic Regime",
        str(
            decision[
                "strategic_regime"
            ]
        ),
    )

with overview_col4:

    st.metric(
        "Top Recommendation",
        str(
            decision[
                "recommendation_1"
            ]
        ),
    )


st.info(
    "Using only the March 31, 2022 financial, macroeconomic, "
    "and market state, the frozen Chapter 3 model estimated "
    f"a {decision['distress_probability']:.2%} probability "
    "of financial distress in the following quarter. "
    "The subsequently observed June 30, 2022 financial state "
    "met the predefined Chapter 3 distress criteria."
)


# ============================================================
# CHAPTER 3
# ============================================================

st.divider()

st.html(
    '<div class="section-kicker">'
    'Chapter 3 · Forward Prediction'
    '</div>'
)

st.subheader(
    "What Did the Model Predict Before Seeing the Next Quarter?"
)

forecast_col1, forecast_col2, forecast_col3 = (
    st.columns(3)
)

with forecast_col1:

    st.metric(
        "Forecast Revenue",
        f"${decision['forecast_revenue']:,.0f}",
    )

with forecast_col2:

    st.metric(
        "Forecast Operating Cash Flow",
        f"${decision['forecast_ocf']:,.0f}",
    )

with forecast_col3:

    st.metric(
        "Forecast EBITDA",
        f"${decision['forecast_ebitda']:,.0f}",
    )


# ============================================================
# WALK-FORWARD RESULTS
# ============================================================

st.markdown(
    "### Consecutive Out-of-Time Predictions"
)

walk_display = (
    walk_forward.copy()
)

rename_candidates = {
    "input_period":
        "Input Quarter",

    "target_period":
        "Next Quarter",

    "predicted_probability":
        "Predicted Distress Probability",

    "distress_probability":
        "Predicted Distress Probability",

    "predicted_class":
        "Predicted Class",

    "actual_class":
        "Actual Class",

    "actual_distress":
        "Actual Class",

    "correct":
        "Correct",
}

walk_display = walk_display.rename(
    columns={
        key: value
        for key, value
        in rename_candidates.items()
        if key in walk_display.columns
    }
)

probability_columns = [
    column
    for column in walk_display.columns
    if "Probability" in column
]

for column in probability_columns:

    walk_display[column] = (
        walk_display[column]
        .apply(
            lambda value:
            f"{value:.2%}"
            if pd.notna(value)
            else ""
        )
    )

st.dataframe(
    walk_display,
    hide_index=True,
    use_container_width=True,
)

if "correct" in walk_forward.columns:

    validation_accuracy = (
        walk_forward[
            "correct"
        ]
        .astype(bool)
        .mean()
    )

else:

    validation_accuracy = 0.80


st.success(
    "Illustrative walk-forward result: "
    f"{validation_accuracy:.0%} classification accuracy "
    f"across {len(walk_forward)} consecutive next-quarter observations."
)

st.caption(
    "This is a five-observation company-level case study, not a claim "
    "that the entire AI CFO system has 80% general accuracy."
)


# ============================================================
# CHAPTER 4
# ============================================================

st.divider()

st.html(
    '<div class="section-kicker">'
    'Chapter 4 · Fresh Recommendation'
    '</div>'
)

st.subheader(
    "What Strategic Response Did AI CFO Generate?"
)

recommendations = pd.DataFrame(
    [
        {
            "Rank": 1,
            "Recommended Action":
                decision[
                    "recommendation_1"
                ],
            "Decision Score":
                decision[
                    "recommendation_1_score"
                ],
        },
        {
            "Rank": 2,
            "Recommended Action":
                decision[
                    "recommendation_2"
                ],
            "Decision Score":
                decision[
                    "recommendation_2_score"
                ],
        },
        {
            "Rank": 3,
            "Recommended Action":
                decision[
                    "recommendation_3"
                ],
            "Decision Score":
                decision[
                    "recommendation_3_score"
                ],
        },
    ]
)

recommendations[
    "Decision Score"
] = (
    recommendations[
        "Decision Score"
    ]
    .round(3)
)

st.dataframe(
    recommendations,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "These recommendations were freshly generated from the March 2022 "
    "decision state and Chapter 3 forecasts. They were not retrieved "
    "from a precomputed historical recommendation row."
)


# ============================================================
# CHAPTER 5
# ============================================================

st.divider()

st.html(
    '<div class="section-kicker">'
    'Chapter 5 · Explain'
    '</div>'
)

st.subheader(
    "Why Was Reduce Operating Expenses Prioritized?"
)

xai_probability = (
    decision[
        "xai_surrogate_probability"
    ]
)

st.metric(
    "Surrogate Recommendation Probability",
    f"{xai_probability:.2%}",
)

xai_display = (
    xai[
        [
            "Feature",
            "Feature Value",
            "SHAP Contribution",
        ]
    ]
    .copy()
)

xai_display[
    "Feature"
] = (
    xai_display[
        "Feature"
    ]
    .astype(str)
    .str.replace(
        "_",
        " ",
        regex=False,
    )
)

xai_display[
    "SHAP Contribution"
] = (
    xai_display[
        "SHAP Contribution"
    ]
    .round(4)
)

st.dataframe(
    xai_display,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "SHAP values describe the behavior of the Chapter 5 surrogate "
    "recommendation model relative to its baseline. They should not "
    "be interpreted as causal effects."
)


# ============================================================
# CHAPTER 6
# ============================================================

st.divider()

st.html(
    '<div class="section-kicker">'
    'Chapter 6 · Simulate'
    '</div>'
)

st.subheader(
    "Four-Quarter PPO Strategy Scenario"
)

sim_col1, sim_col2, sim_col3 = (
    st.columns(3)
)

with sim_col1:

    st.metric(
        "Profit Margin",
        (
            f"{simulation['initial_profit_margin']:.2%} "
            f"→ "
            f"{simulation['final_profit_margin']:.2%}"
        ),
    )

with sim_col2:

    st.metric(
        "Cash Ratio",
        (
            f"{simulation['initial_cash_ratio']:.3f} "
            f"→ "
            f"{simulation['final_cash_ratio']:.3f}"
        ),
    )

with sim_col3:

    st.metric(
        "Distress Probability",
        (
            f"{simulation['initial_distress_probability']:.2%} "
            f"→ "
            f"{simulation['final_distress_probability']:.2%}"
        ),
    )


trajectory_display = trajectory.copy()

trajectory_display = trajectory_display.rename(
    columns={
        "quarter":
            "Quarter",

        "action_id":
            "Action ID",

        "strategy":
            "PPO Strategy",

        "reward":
            "Reward",
    }
)

if "Quarter" in trajectory_display.columns:

    trajectory_display[
        "Quarter"
    ] = (
        trajectory_display[
            "Quarter"
        ]
        .apply(
            lambda value:
            f"Q{int(value)}"
        )
    )

if "Reward" in trajectory_display.columns:

    trajectory_display[
        "Reward"
    ] = (
        trajectory_display[
            "Reward"
        ]
        .round(4)
    )


display_columns = [
    column
    for column in [
        "Quarter",
        "Action ID",
        "PPO Strategy",
        "Reward",
    ]
    if column in trajectory_display.columns
]

st.dataframe(
    trajectory_display[
        display_columns
    ],
    hide_index=True,
    use_container_width=True,
)

st.warning(
    "The Chapter 6 result is a modeled scenario, not a prediction "
    "that these financial changes will occur in the real world. "
    "In this case the simulated profit margin and cash ratio improve, "
    "while modeled distress remains very high — a mixed outcome."
)


# ============================================================
# VALIDATION INTERPRETATION
# ============================================================

st.divider()

st.html(
    '<div class="section-kicker">'
    'Validation Interpretation'
    '</div>'
)

st.subheader(
    "What This Case Demonstrates"
)

st.markdown(
    """
**1. Prediction was made from an earlier financial state.**  
The March 2022 state was passed through the frozen Chapter 3 models
to estimate the following-quarter financial outlook.

**2. The later realized outcome was then observed.**  
The June 2022 financial state satisfied the dissertation's predefined
distress criteria, so the March risk classification was correct.

**3. Recommendations were generated from the fresh decision state.**  
Chapter 4 classified the company in the Defensive regime and ranked
Reduce Operating Expenses as the highest-priority CFO action.

**4. The decision was explainable.**  
Chapter 5 identified the inputs most influential to its surrogate
representation of the recommendation.

**5. Strategy was stress-tested rather than presented as certainty.**  
Chapter 6 ran the trained PPO policy from the newly constructed
March 2022 state over four modeled quarters.
"""
)

st.info(
    "The forward-validation evidence directly evaluates the distress "
    "prediction. The strategic recommendation and reinforcement-learning "
    "simulation are decision-support outputs and are not claimed to be "
    "proven real-world interventions."
)

st.caption(
    "Research models were developed before the hackathon. "
    "The hackathon implementation integrates those components into "
    "an end-to-end decision-intelligence workflow and extends the "
    "system with forward-validation and deployment functionality."
)
