import os

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

st.markdown(
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
            rgba(28, 31, 38, 0.96),
            rgba(45, 52, 65, 0.92)
        );
        margin-bottom: 1.5rem;
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

    .company-card {
        padding: 1rem 1.25rem;
        border-radius: 14px;
        border: 1px solid rgba(120, 120, 120, 0.2);
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-radius: 14px;
        padding: 1rem;
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
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

THESIS_ROOT = (
    "/Users/humairaali/"
    "Revised_AI_CFO_Thesis"
)

os.environ["AI_CFO_THESIS_ROOT"] = THESIS_ROOT


# ============================================================
# LOAD RESEARCH DATA
# ============================================================

@st.cache_resource
def load_distress_model():

    model_path = (
        f"{THESIS_ROOT}/"
        "models/"
        "xgb_distress_h1.joblib"
    )

    return joblib.load(
        model_path
    )


@st.cache_data
def load_chapter3_panel():

    panel_path = (
        f"{THESIS_ROOT}/"
        "data/processed/"
        "chapter3_model_panel_final_v1.parquet"
    )

    panel = pd.read_parquet(
        panel_path
    )

    panel["sic"] = (
        panel["sic"]
        .astype(str)
    )

    panel["sic_code"] = (
        pd.factorize(
            panel["sic"]
        )[0]
    )

    panel["period"] = pd.to_datetime(
        panel["period"]
    )

    return panel


@st.cache_data
def load_chapter4_recommendations():

    recommendation_path = (
        f"{THESIS_ROOT}/"
        "results/tables/chapter4/"
        "chapter4_final_recommendations.csv"
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

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">AI CFO</div>

        <div class="hero-subtitle">
            Explainable Agentic Decision Intelligence
            for Financial Management
        </div>

        <div class="pipeline-text">
            Predict → Recommend → Explain → Simulate → Decide
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Research-driven financial decision support combining "
    "predictive modeling, governance-aware recommendations, "
    "explainable AI, reinforcement learning, and executive reasoning."
)


# ============================================================
# SIDEBAR — COMPANY SELECTION
# ============================================================

st.sidebar.markdown(
    "## AI CFO"
)

st.sidebar.caption(
    "Decision Intelligence Console"
)

st.sidebar.divider()

st.sidebar.header(
    "Analysis Setup"
)

company_options = (
    recommendation_df[
        ["cik", "company_name"]
    ]
    .drop_duplicates()
    .sort_values(
        "company_name"
    )
)

company_options["display_name"] = (
    company_options["company_name"]
    + "  |  CIK "
    + company_options["cik"]
)

selected_company_display = (
    st.sidebar.selectbox(
        "Select Company",
        company_options[
            "display_name"
        ].tolist(),
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

selected_period_string = (
    st.sidebar.selectbox(
        "Select Quarter",
        period_labels,
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
    "Runs the complete Chapter 3–6 "
    "decision intelligence pipeline."
)


# ============================================================
# INTRO PANEL
# ============================================================

if not run_analysis:

    st.info(
        "Select a company and reporting quarter from the "
        "sidebar, then run the AI CFO analysis."
    )

    st.markdown(
        "### Decision Intelligence Pipeline"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.markdown(
            "#### 01 · Predict"
        )

        st.write(
            "Estimate next-quarter financial distress risk."
        )

    with col2:

        st.markdown(
            "#### 02 · Recommend"
        )

        st.write(
            "Generate governance-aware strategic actions."
        )

    with col3:

        st.markdown(
            "#### 03 · Explain"
        )

        st.write(
            "Identify the drivers behind the recommendation."
        )

    with col4:

        st.markdown(
            "#### 04 · Simulate"
        )

        st.write(
            "Stress-test strategy across four modeled quarters."
        )

    st.divider()

    st.markdown(
        "### From Prediction to Decision"
    )

    st.write(
        "Traditional financial AI often stops after predicting "
        "what may happen. AI CFO extends that workflow by connecting "
        "risk prediction to strategic recommendations, explainability, "
        "multi-quarter simulation, and executive decision support."
    )

    st.stop()


# ============================================================
# FIND COMPANY-QUARTER IN CHAPTER 3 PANEL
# ============================================================

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
        "This company-quarter was not found in the "
        "Chapter 3 modeling panel."
    )

    st.stop()


sample_row = (
    sample_match.iloc[0]
)


# ============================================================
# BUILD MODEL INPUT
# ============================================================

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


# ============================================================
# RUN COMPLETE AI CFO PIPELINE
# ============================================================

with st.spinner(
    "AI CFO is analyzing risk, recommendations, "
    "explanations, and strategy scenarios..."
):

    result = run_cfo_analysis(
        company_data
    )


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

st.markdown(
    '<div class="section-kicker">Company Analysis</div>',
    unsafe_allow_html=True,
)

st.header(
    selected_company_name
)

st.caption(
    f"Reporting period: {selected_period_string} "
    f"• CIK {selected_cik}"
)

st.divider()


# ============================================================
# CHAPTER 3 — RISK OUTLOOK
# ============================================================

st.markdown(
    '<div class="section-kicker">Chapter 3 · Predict</div>',
    unsafe_allow_html=True,
)

st.subheader(
    "Financial Risk Outlook"
)

distress = forecast.get(
    "financial_distress",
    {},
)

distress_probability = distress.get(
    "distress_probability"
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


# ============================================================
# CHAPTER 4 — RECOMMENDATIONS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-kicker">Chapter 4 · Recommend</div>',
    unsafe_allow_html=True,
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
# CHAPTER 5 — EXPLAINABILITY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-kicker">Chapter 5 · Explain</div>',
    unsafe_allow_html=True,
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
        f"**Chapter 5 surrogate probability:** "
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

    st.info(
        "Strongest explanatory driver: "
        f"**{strongest_driver['feature']}** "
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
# CHAPTER 6 — PPO STRATEGY SIMULATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-kicker">Chapter 6 · Simulate</div>',
    unsafe_allow_html=True,
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

    trajectory_display = (
        trajectory_df[
            [
                "quarter",
                "action_id",
                "strategy",
                "profit_margin_after",
                "cash_ratio_after",
                "debt_ratio_after",
                "distress_probability_after",
                "reward",
            ]
        ]
        .rename(
            columns={
                "quarter":
                    "Quarter",

                "action_id":
                    "Action ID",

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

                "reward":
                    "Reward",
            }
        )
    )

    st.markdown(
        "### Strategy Path"
    )

    st.dataframe(
        trajectory_display,
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

st.markdown(
    '<div class="section-kicker">Executive Decision Support</div>',
    unsafe_allow_html=True,
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

        st.markdown(
            "#### Explainability"
        )

        st.write(
            "The strongest Chapter 5 SHAP driver is "
            f"**{strongest_driver['feature']}**, which "
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
# MODEL GOVERNANCE NOTICE
# ============================================================

st.divider()

st.markdown(
    "#### Model Governance"
)

st.caption(
    "AI CFO is a research-based decision-support system. "
    "Predictive outputs, SHAP explanations, governance-aware "
    "recommendations, and PPO simulations should be interpreted "
    "within their respective model assumptions. Simulated outcomes "
    "are scenarios, not guaranteed business results."
)

   
    

