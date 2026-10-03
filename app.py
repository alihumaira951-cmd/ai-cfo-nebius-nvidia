import os

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from agent.cfo_agent import run_cfo_analysis


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI CFO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VISUAL STYLING
# ============================================================

st.html(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }

    .hero-card {
        padding: 2.2rem 2.4rem;
        border-radius: 18px;
        border: 1px solid rgba(120, 120, 120, 0.22);
        background: linear-gradient(
            135deg,
            rgba(28, 31, 38, 0.97),
            rgba(45, 52, 65, 0.94)
        );
        margin-bottom: 1.3rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.04em;
        color: white;
    }

    .hero-subtitle {
        font-size: 1.25rem;
        color: #d8dce5;
        margin-bottom: 1.4rem;
    }

    .pipeline-text {
        display: inline-block;
        padding: 0.55rem 0.9rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.09);
        color: #e8eaf0;
        font-size: 0.92rem;
        font-weight: 500;
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

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(120, 120, 120, 0.15);
    }

    </style>
    """
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent

ARTIFACT_ROOT = (
    REPO_ROOT
    / "deployment_artifacts"
)


# ============================================================
# DEFAULT DEMO CASE
# ============================================================

DEFAULT_CIK = "96699"
DEFAULT_PERIOD = "2021-12-31"


# ============================================================
# LOAD RESEARCH DATA
# ============================================================

@st.cache_resource
def load_distress_model():

    model_path = (
        ARTIFACT_ROOT
        / "models"
        / "xgb_distress_h1.joblib"
    )

    return joblib.load(
        model_path
    )


@st.cache_data
def load_chapter3_panel():

    panel_path = (
        ARTIFACT_ROOT
        / "data"
        / "chapter3_deployment_panel.parquet"
    )

    panel = pd.read_parquet(
        panel_path
    )

    panel["period"] = pd.to_datetime(
        panel["period"]
    )

    panel["cik"] = (
        panel["cik"]
        .astype(str)
    )

    return panel


@st.cache_data
def load_chapter4_recommendations():

    recommendation_path = (
        ARTIFACT_ROOT
        / "tables"
        / "chapter4_recommendations.csv"
    )

    recommendations = pd.read_csv(
        recommendation_path
    )

    recommendations["period"] = pd.to_datetime(
        recommendations["period"]
    )

    recommendations["cik"] = (
        recommendations["cik"]
        .astype(str)
    )

    return recommendations


# ============================================================
# LOAD ARTIFACTS
# ============================================================

model = load_distress_model()

panel_df = load_chapter3_panel()

recommendation_df = (
    load_chapter4_recommendations()
)

model_features = list(
    model.feature_names_in_
)


# ============================================================
# HERO HEADER
# ============================================================

st.html(
    """
    <div class="hero-card">

        <div class="hero-title">
            AI CFO
        </div>

        <div class="hero-subtitle">
            Explainable Agentic Decision Intelligence
            for Financial Management
        </div>

        <div class="pipeline-text">
            Predict → Recommend → Explain → Simulate → Decide
        </div>

    </div>
    """
)

st.caption(
    "Research-driven financial decision support combining "
    "predictive modeling, governance-aware recommendations, "
    "explainable AI, reinforcement learning, and executive reasoning."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## AI CFO"
)

st.sidebar.caption(
    "Decision Intelligence Console"
)

st.sidebar.divider()

st.sidebar.markdown(
    "### Analysis Setup"
)


# ============================================================
# COMPANY SELECTION
# ============================================================

company_options = (
    recommendation_df[
        ["cik", "company_name"]
    ]
    .drop_duplicates()
    .sort_values(
        "company_name"
    )
    .reset_index(
        drop=True
    )
)

company_options["display_name"] = (
    company_options["company_name"]
    + "  |  CIK "
    + company_options["cik"]
)

company_display_list = (
    company_options[
        "display_name"
    ]
    .tolist()
)

default_company_index = 0

default_company_matches = (
    company_options.index[
        company_options["cik"]
        == DEFAULT_CIK
    ]
    .tolist()
)

if default_company_matches:

    default_company_index = (
        default_company_matches[0]
    )


selected_company_display = (
    st.sidebar.selectbox(
        "Select Company",
        company_display_list,
        index=default_company_index,
    )
)

selected_company_row = (
    company_options[
        company_options[
            "display_name"
        ]
        == selected_company_display
    ]
    .iloc[0]
)

selected_cik = str(
    selected_company_row[
        "cik"
    ]
)

selected_company_name = (
    selected_company_row[
        "company_name"
    ]
)


# ============================================================
# PERIOD SELECTION
# ============================================================

available_periods = (
    recommendation_df[
        recommendation_df[
            "cik"
        ]
        == selected_cik
    ]["period"]
    .drop_duplicates()
    .sort_values()
)

period_labels = [
    period.strftime(
        "%Y-%m-%d"
    )
    for period in available_periods
]

default_period_index = 0

if DEFAULT_PERIOD in period_labels:

    default_period_index = (
        period_labels.index(
            DEFAULT_PERIOD
        )
    )

elif period_labels:

    default_period_index = (
        len(period_labels) - 1
    )


selected_period_string = (
    st.sidebar.selectbox(
        "Select Quarter",
        period_labels,
        index=default_period_index,
    )
)

selected_period = pd.Timestamp(
    selected_period_string
)


# ============================================================
# ANALYSIS BUTTON
# ============================================================

st.sidebar.divider()

run_analysis = (
    st.sidebar.button(
        "Run AI CFO Analysis",
        type="primary",
        use_container_width=True,
    )
)

st.sidebar.caption(
    "Runs the full predictive, recommendation, "
    "explainability, and strategy-simulation pipeline."
)


# ============================================================
# RUNTIME STATUS
# ============================================================

st.sidebar.divider()

st.sidebar.markdown(
    "### System Status"
)

st.sidebar.success(
    "Research pipeline: Ready"
)

if os.environ.get(
    "NEBIUS_API_KEY"
):

    st.sidebar.success(
        "Nemotron executive layer: Connected"
    )

else:

    st.sidebar.info(
        "Nemotron executive layer: Local fallback"
    )

    st.sidebar.caption(
        "Nebius credentials have not been configured. "
        "No live Nemotron claim is made."
    )


# ============================================================
# ANALYSIS STATE
# ============================================================

analysis_key = (
    f"{selected_cik}|"
    f"{selected_period_string}"
)

stored_key = (
    st.session_state.get(
        "analysis_key"
    )
)

stored_result = (
    st.session_state.get(
        "analysis_result"
    )
)


# ============================================================
# RUN ANALYSIS WHEN REQUESTED
# ============================================================

if run_analysis:

    sample_match = panel_df[
        (
            panel_df["cik"]
            .astype(str)
            == selected_cik
        )
        & (
            panel_df["period"]
            == selected_period
        )
    ]

    if sample_match.empty:

        st.error(
            "This company-quarter was not found "
            "in the Chapter 3 modeling panel."
        )

        st.stop()


    sample_row = (
        sample_match.iloc[0]
    )


    company_data = {
        feature: sample_row[
            feature
        ]
        for feature in model_features
    }

    company_data["cik"] = (
        selected_cik
    )

    company_data["period"] = (
        selected_period.strftime(
            "%Y-%m-%d"
        )
    )


    with st.spinner(
        "AI CFO is analyzing risk, recommendations, "
        "explanations, and strategy scenarios..."
    ):

        result = run_cfo_analysis(
            company_data
        )


    st.session_state[
        "analysis_result"
    ] = result

    st.session_state[
        "analysis_key"
    ] = analysis_key


elif (
    stored_result is not None
    and stored_key == analysis_key
):

    result = stored_result


else:

    result = None


# ============================================================
# LANDING STATE
# ============================================================

if result is None:

    st.info(
        "The primary demonstration case is preselected. "
        "Click **Run AI CFO Analysis** to begin, or choose "
        "another company and reporting quarter."
    )

    st.markdown(
        "### Decision Intelligence Pipeline"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.markdown(
            "#### 01 · Predict"
        )

        st.write(
            "Estimate next-quarter "
            "financial distress risk."
        )

    with col2:

        st.markdown(
            "#### 02 · Recommend"
        )

        st.write(
            "Generate governance-aware "
            "strategic actions."
        )

    with col3:

        st.markdown(
            "#### 03 · Explain"
        )

        st.write(
            "Identify the model drivers "
            "behind the recommendation."
        )

    with col4:

        st.markdown(
            "#### 04 · Simulate"
        )

        st.write(
            "Stress-test strategy across "
            "four modeled quarters."
        )


    st.divider()

    st.markdown(
        "### From Prediction to Decision"
    )

    st.write(
        "Traditional financial AI often stops after predicting "
        "what may happen. AI CFO extends the workflow by connecting "
        "financial risk prediction to strategic recommendations, "
        "explainability, multi-quarter simulation, and executive "
        "decision support."
    )

    st.stop()


# ============================================================
# UNPACK RESULTS
# ============================================================

forecast = result[
    "forecast"
]

recommendations = result[
    "recommendations"
]

explanation = result[
    "explanation"
]

simulation = result[
    "simulation"
]

executive_brief = result[
    "executive_brief"
]


# ============================================================
# COMPANY HEADER
# ============================================================

st.html(
    '<div class="section-kicker">'
    'Company Analysis'
    '</div>'
)

st.header(
    selected_company_name
)

st.caption(
    f"Reporting period: "
    f"{selected_period_string} "
    f"• CIK {selected_cik}"
)


# ============================================================
# PREPARE EXECUTIVE SNAPSHOT
# ============================================================

distress = forecast.get(
    "financial_distress",
    {},
)

distress_probability = distress.get(
    "distress_probability"
)

strategic_regime = (
    recommendations.get(
        "strategic_regime",
        "Unavailable",
    )
)

recommendation_list = (
    recommendations.get(
        "recommendations",
        [],
    )
)

top_action = (
    recommendation_list[0].get(
        "action",
        "Unavailable",
    )
    if recommendation_list
    else "Unavailable"
)

simulation_interpretation = (
    executive_brief.get(
        "simulation_interpretation",
        {},
    )
)

scenario_outcome = (
    simulation_interpretation.get(
        "outcome"
    )
)

if scenario_outcome == "improving":

    scenario_label = (
        "Improving"
    )

elif scenario_outcome == "deteriorating":

    scenario_label = (
        "Deteriorating"
    )

elif scenario_outcome == "mixed":

    scenario_label = (
        "Mixed Outcome"
    )

else:

    scenario_label = (
        "See Simulation"
    )


# ============================================================
# EXECUTIVE SNAPSHOT
# ============================================================

st.markdown(
    "### Executive Snapshot"
)

snapshot_col1, snapshot_col2, snapshot_col3, snapshot_col4 = (
    st.columns(4)
)

with snapshot_col1:

    if distress_probability is not None:

        st.metric(
            "Distress Risk",
            f"{distress_probability:.2%}",
        )

    else:

        st.metric(
            "Distress Risk",
            "Unavailable",
        )


with snapshot_col2:

    st.metric(
        "Strategic Regime",
        strategic_regime,
    )


with snapshot_col3:

    st.metric(
        "Top Recommendation",
        top_action,
    )


with snapshot_col4:

    st.metric(
        "Scenario Assessment",
        scenario_label,
    )


st.divider()


# ============================================================
# CHAPTER 3 — PREDICT
# ============================================================

st.html(
    '<div class="section-kicker">'
    'Chapter 3 · Predict'
    '</div>'
)

st.subheader(
    "Financial Risk Outlook"
)

distress_threshold = distress.get(
    "classification_threshold"
)

predicted_distress = distress.get(
    "predicted_distress"
)

risk_col1, risk_col2, risk_col3 = (
    st.columns(3)
)

with risk_col1:

    if distress_probability is not None:

        st.metric(
            "Next-Quarter Distress Probability",
            f"{distress_probability:.2%}",
        )


with risk_col2:

    if distress_threshold is not None:

        st.metric(
            "Classification Threshold",
            f"{distress_threshold:.0%}",
        )


with risk_col3:

    risk_label = (
        "Distress Flag"
        if predicted_distress == 1
        else "No Distress Flag"
    )

    st.metric(
        "Model Classification",
        risk_label,
    )


if predicted_distress == 1:

    st.warning(
        "The modeled distress probability exceeds "
        "the classification threshold."
    )


# ============================================================
# CHAPTER 4 — RECOMMEND
# ============================================================

st.divider()

st.html(
    '<div class="section-kicker">'
    'Chapter 4 · Recommend'
    '</div>'
)

st.subheader(
    "Governance-Aware Recommendations"
)

if recommendations.get(
    "status"
) == "success":

    regime_col1, regime_col2 = (
        st.columns(2)
    )

    with regime_col1:

        st.metric(
            "Strategic Regime",
            recommendations[
                "strategic_regime"
            ],
        )


    with regime_col2:

        st.metric(
            "Financial Health Score",
            f"{recommendations['financial_health_score']:.3f}",
        )


    recommendation_table = (
        pd.DataFrame(
            recommendations[
                "recommendations"
            ]
        )
    )

    recommendation_table = (
        recommendation_table.rename(
            columns={
                "rank":
                    "Rank",

                "action":
                    "Recommended Action",

                "score":
                    "Decision Score",
            }
        )
    )

    if (
        "Decision Score"
        in recommendation_table.columns
    ):

        recommendation_table[
            "Decision Score"
        ] = (
            recommendation_table[
                "Decision Score"
            ]
            .round(3)
        )


    st.dataframe(
        recommendation_table,
        hide_index=True,
        use_container_width=True,
    )


else:

    st.warning(
        recommendations
    )


# ============================================================
# CHAPTER 5 — EXPLAIN
# ============================================================

st.divider()

st.html(
    '<div class="section-kicker">'
    'Chapter 5 · Explain'
    '</div>'
)

st.subheader(
    "Why This Recommendation?"
)

if explanation.get(
    "status"
) == "success":

    st.markdown(
        f"**Recommendation explained:** "
        f"{explanation['recommendation_explained']}"
    )

    st.markdown(
        f"**Surrogate recommendation probability:** "
        f"{explanation['surrogate_predicted_probability']:.2%}"
    )


    driver_table = pd.DataFrame(
        explanation[
            "drivers"
        ]
    )

    driver_table = (
        driver_table[
            [
                "feature",
                "feature_value",
                "shap_value",
                "direction",
            ]
        ]
        .rename(
            columns={
                "feature":
                    "Driver",

                "feature_value":
                    "Value",

                "shap_value":
                    "SHAP Contribution",

                "direction":
                    "Effect",
            }
        )
    )


    driver_table[
        "Driver"
    ] = (
        driver_table[
            "Driver"
        ]
        .astype(str)
        .str.replace(
            "_",
            " ",
            regex=False,
        )
    )


    driver_table[
        "SHAP Contribution"
    ] = (
        driver_table[
            "SHAP Contribution"
        ]
        .round(4)
    )


    st.dataframe(
        driver_table,
        hide_index=True,
        use_container_width=True,
    )


    strongest_driver = (
        explanation[
            "drivers"
        ][0]
    )

    strongest_driver_name = (
        strongest_driver[
            "feature"
        ]
        .replace(
            "_",
            " ",
        )
    )


    st.info(
        "Strongest explanatory driver: "
        f"**{strongest_driver_name}** "
        f"({strongest_driver['direction']})."
    )


    st.caption(
        explanation[
            "interpretation_note"
        ]
    )


else:

    st.warning(
        explanation
    )


# ============================================================
# CHAPTER 6 — SIMULATE
# ============================================================

st.divider()

st.html(
    '<div class="section-kicker">'
    'Chapter 6 · Simulate'
    '</div>'
)

st.subheader(
    "Four-Quarter Strategy Simulation"
)

if simulation.get(
    "status"
) == "success":

    first_action = simulation.get(
        "initial_selected_action"
    )

    if first_action:

        st.markdown(
            "### PPO-Selected Strategy"
        )

        st.success(
            first_action[
                "strategy"
            ]
        )

        st.caption(
            "Governance profile: "
            f"{first_action['governance_profile']}"
        )


    sim_summary = simulation.get(
        "summary",
        {},
    )


    initial_profit_margin = sim_summary.get(
        "initial_profit_margin",
        0,
    )

    final_profit_margin = sim_summary.get(
        "final_profit_margin",
        0,
    )

    initial_cash_ratio = sim_summary.get(
        "initial_cash_ratio",
        0,
    )

    final_cash_ratio = sim_summary.get(
        "final_cash_ratio",
        0,
    )

    initial_debt_ratio = sim_summary.get(
        "initial_debt_ratio"
    )

    final_debt_ratio = sim_summary.get(
        "final_debt_ratio"
    )

    initial_distress_probability = sim_summary.get(
        "initial_distress_probability",
        0,
    )

    final_distress_probability = sim_summary.get(
        "final_distress_probability",
        0,
    )


    sim_col1, sim_col2, sim_col3 = (
        st.columns(3)
    )

    with sim_col1:

        st.metric(
            "Profit Margin",
            (
                f"{initial_profit_margin:.2%} "
                f"→ "
                f"{final_profit_margin:.2%}"
            ),
        )


    with sim_col2:

        st.metric(
            "Cash Ratio",
            (
                f"{initial_cash_ratio:.3f} "
                f"→ "
                f"{final_cash_ratio:.3f}"
            ),
        )


    with sim_col3:

        st.metric(
            "Distress Probability",
            (
                f"{initial_distress_probability:.3%} "
                f"→ "
                f"{final_distress_probability:.3%}"
            ),
        )


    st.caption(
        "Starting financial state → simulated position "
        "after four quarters."
    )


    trajectory_df = pd.DataFrame(
        simulation[
            "trajectory"
        ]
    )


    # --------------------------------------------------------
    # SCENARIO TRAJECTORY CHART
    # --------------------------------------------------------

    st.markdown(
        "### Scenario Trajectory"
    )


    initial_chart_row = {
        "Quarter": "Start",
        "Profit Margin":
            initial_profit_margin,
        "Cash Ratio":
            initial_cash_ratio,
        "Distress Probability":
            initial_distress_probability,
    }

    if initial_debt_ratio is not None:

        initial_chart_row[
            "Debt Ratio"
        ] = initial_debt_ratio


    chart_rows = [
        initial_chart_row
    ]


    for _, row in trajectory_df.iterrows():

        chart_row = {
            "Quarter":
                f"Q{int(row['quarter'])}",

            "Profit Margin":
                row[
                    "profit_margin_after"
                ],

            "Cash Ratio":
                row[
                    "cash_ratio_after"
                ],

            "Distress Probability":
                row[
                    "distress_probability_after"
                ],
        }

        if (
            "debt_ratio_after"
            in row.index
        ):

            chart_row[
                "Debt Ratio"
            ] = row[
                "debt_ratio_after"
            ]

        chart_rows.append(
            chart_row
        )


    chart_df = (
        pd.DataFrame(
            chart_rows
        )
        .set_index(
            "Quarter"
        )
    )


    st.line_chart(
        chart_df,
        use_container_width=True,
        height=320,
    )

    st.caption(
        "Values are shown on their native ratio/proportion scale. "
        "The chart represents the modeled Chapter 6 scenario, "
        "not a guaranteed forecast."
    )


    # --------------------------------------------------------
    # EXECUTIVE-FRIENDLY STRATEGY TABLE
    # --------------------------------------------------------

    st.markdown(
        "### Strategy Path"
    )


    trajectory_display = (
        trajectory_df[
            [
                "quarter",
                "strategy",
                "profit_margin_after",
                "cash_ratio_after",
                "debt_ratio_after",
                "distress_probability_after",
            ]
        ]
        .copy()
        .rename(
            columns={
                "quarter":
                    "Quarter",

                "strategy":
                    "PPO Strategy",

                "profit_margin_after":
                    "Profit Margin",

                "cash_ratio_after":
                    "Cash Ratio",

                "debt_ratio_after":
                    "Debt Ratio",

                "distress_probability_after":
                    "Distress Probability",
            }
        )
    )


    trajectory_display[
        "Quarter"
    ] = (
        trajectory_display[
            "Quarter"
        ]
        .apply(
            lambda x:
            f"Q{int(x)}"
        )
    )


    trajectory_display[
        "Profit Margin"
    ] = (
        trajectory_display[
            "Profit Margin"
        ]
        .apply(
            lambda x:
            f"{x:.2%}"
        )
    )


    trajectory_display[
        "Cash Ratio"
    ] = (
        trajectory_display[
            "Cash Ratio"
        ]
        .apply(
            lambda x:
            f"{x:.3f}"
        )
    )


    trajectory_display[
        "Debt Ratio"
    ] = (
        trajectory_display[
            "Debt Ratio"
        ]
        .apply(
            lambda x:
            f"{x:.3f}"
        )
    )


    trajectory_display[
        "Distress Probability"
    ] = (
        trajectory_display[
            "Distress Probability"
        ]
        .apply(
            lambda x:
            f"{x:.2%}"
        )
    )


    st.dataframe(
        trajectory_display,
        hide_index=True,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # TECHNICAL DETAILS
    # --------------------------------------------------------

    with st.expander(
        "Technical simulation details"
    ):

        technical_table = (
            trajectory_df[
                [
                    "quarter",
                    "action_id",
                    "reward",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "quarter":
                        "Quarter",

                    "action_id":
                        "Action ID",

                    "reward":
                        "Reward",
                }
            )
        )

        technical_table[
            "Reward"
        ] = (
            technical_table[
                "Reward"
            ]
            .round(4)
        )

        st.dataframe(
            technical_table,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            simulation[
                "interpretation_note"
            ]
        )


else:

    st.warning(
        simulation
    )


# ============================================================
# EXECUTIVE DECISION BRIEF
# ============================================================

st.divider()

st.html(
    '<div class="section-kicker">'
    'Executive Decision Support'
    '</div>'
)

st.subheader(
    "AI CFO Executive Decision Brief"
)


if executive_brief.get(
    "status"
) == "success":

    st.markdown(
        executive_brief[
            "content"
        ]
    )

    st.caption(
        "Executive reasoning generated by "
        f"{executive_brief.get('model')}"
    )


else:

    st.info(
        "NVIDIA Nemotron through Nebius Token Factory "
        "is not connected yet. The structured AI CFO "
        "decision summary is shown below."
    )


    risk_outlook = executive_brief.get(
        "risk_outlook",
        {},
    )

    top_recommendation = (
        executive_brief.get(
            "top_recommendation"
        )
    )

    strongest_driver = (
        executive_brief.get(
            "strongest_explanation_driver"
        )
    )

    local_simulation = (
        executive_brief.get(
            "simulation_summary",
            {},
        )
    )

    simulation_interpretation = (
        executive_brief.get(
            "simulation_interpretation",
            {},
        )
    )


    st.markdown(
        "#### Risk Outlook"
    )

    if risk_outlook.get(
        "distress_probability"
    ) is not None:

        st.write(
            "The Chapter 3 model estimates a "
            f"**{risk_outlook['distress_probability']:.2%}** "
            "next-quarter financial distress probability."
        )


    if top_recommendation:

        st.markdown(
            "#### Strategic Recommendation"
        )

        st.write(
            "The highest-ranked Chapter 4 action is "
            f"**{top_recommendation['action']}** "
            f"with a decision score of "
            f"**{top_recommendation['score']:.3f}**."
        )


    if strongest_driver:

        executive_driver_name = (
            strongest_driver[
                "feature"
            ]
            .replace(
                "_",
                " ",
            )
        )

        st.markdown(
            "#### Explainability"
        )

        st.write(
            "The strongest Chapter 5 SHAP driver is "
            f"**{executive_driver_name}**, which "
            f"**{strongest_driver['direction']}**."
        )


    if local_simulation:

        st.markdown(
            "#### Strategy Simulation"
        )

        st.write(
            "Across the four-quarter Chapter 6 PPO "
            "simulation, profit margin moves from "
            f"**{local_simulation.get('initial_profit_margin', 0):.2%}** "
            "to "
            f"**{local_simulation.get('final_profit_margin', 0):.2%}**, "
            "while the cash ratio moves from "
            f"**{local_simulation.get('initial_cash_ratio', 0):.3f}** "
            "to "
            f"**{local_simulation.get('final_cash_ratio', 0):.3f}** "
            "and modeled distress moves from "
            f"**{local_simulation.get('initial_distress_probability', 0):.2%}** "
            "to "
            f"**{local_simulation.get('final_distress_probability', 0):.2%}**."
        )


    if simulation_interpretation:

        st.markdown(
            "#### Executive Interpretation"
        )

        simulation_outcome = (
            simulation_interpretation.get(
                "outcome",
                "mixed",
            )
        )

        simulation_message = (
            simulation_interpretation.get(
                "message",
                "",
            )
        )


        if simulation_outcome == "improving":

            st.success(
                f"**Scenario Assessment: Improving**\n\n"
                f"{simulation_message}"
            )


        elif simulation_outcome == "deteriorating":

            st.error(
                f"**Scenario Assessment: Deteriorating**\n\n"
                f"{simulation_message}"
            )


        elif simulation_outcome == "mixed":

            st.warning(
                f"**Scenario Assessment: Mixed Outcome**\n\n"
                f"{simulation_message}"
            )


        else:

            st.info(
                simulation_message
            )


    st.caption(
        executive_brief.get(
            "note",
            ""
        )
    )


# ============================================================
# MODEL GOVERNANCE
# ============================================================

st.divider()

st.markdown(
    "#### Model Governance"
)

st.caption(
    "AI CFO is a research-based decision-support system. "
    "Predictive outputs, SHAP explanations, governance-aware "
    "recommendations, and PPO simulations should be interpreted "
    "within their respective model assumptions. "
    "Simulation outputs are scenario-based modeled outcomes, "
    "not guaranteed business results or financial advice."
)