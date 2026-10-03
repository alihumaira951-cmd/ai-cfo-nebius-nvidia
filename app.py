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

    # Recreate Chapter 3 SIC encoding
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
# HEADER
# ============================================================

st.title(
    "AI CFO"
)

st.subheader(
    "Explainable Agentic Decision Intelligence "
    "for Financial Management"
)

st.caption(
    "Prediction → Governance-Aware Recommendation → "
    "Explainability → Strategy Simulation → Executive Decision Support"
)

st.divider()


# ============================================================
# SIDEBAR — COMPANY SELECTION
# ============================================================

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

run_analysis = (
    st.sidebar.button(
        "Run AI CFO Analysis",
        type="primary",
        use_container_width=True,
    )
)


# ============================================================
# INTRO PANEL
# ============================================================

if not run_analysis:

    st.info(
        "Choose a company and quarter from the sidebar, "
        "then click **Run AI CFO Analysis**."
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:
        st.markdown(
            "### Chapter 3"
        )
        st.write(
            "Predictive financial risk modeling"
        )

    with col2:
        st.markdown(
            "### Chapter 4"
        )
        st.write(
            "Governance-aware strategic recommendations"
        )

    with col3:
        st.markdown(
            "### Chapter 5"
        )
        st.write(
            "SHAP-based recommendation explanation"
        )

    with col4:
        st.markdown(
            "### Chapter 6"
        )
        st.write(
            "PPO strategy simulation"
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
    "AI CFO is analyzing the company..."
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

st.header(
    selected_company_name
)

st.caption(
    f"Analysis period: "
    f"{selected_period_string}"
)

st.divider()


# ============================================================
# CHAPTER 3 — RISK OUTLOOK
# ============================================================

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
            "### Initial PPO Strategy"
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

    sim_col1, sim_col2, sim_col3 = (
        st.columns(3)
    )

    with sim_col1:

        st.metric(
            "Profit Margin",
            f"{sim_summary.get('final_profit_margin', 0):.2%}",
            delta=(
                f"{sim_summary.get('final_profit_margin', 0) - sim_summary.get('initial_profit_margin', 0):.2%}"
            ),
        )

    with sim_col2:

        st.metric(
            "Cash Ratio",
            f"{sim_summary.get('final_cash_ratio', 0):.3f}",
            delta=(
                f"{sim_summary.get('final_cash_ratio', 0) - sim_summary.get('initial_cash_ratio', 0):.3f}"
            ),
        )

    with sim_col3:

        st.metric(
            "Distress Probability",
            f"{sim_summary.get('final_distress_probability', 0):.2%}",
            delta=(
                f"{sim_summary.get('final_distress_probability', 0) - sim_summary.get('initial_distress_probability', 0):.2%}"
            ),
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
            f"**{local_simulation.get('final_cash_ratio', 0):.3f}**."
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

st.caption(
    "AI CFO is a research-based decision-support system. "
    "Predictive outputs, SHAP explanations, governance-aware "
    "recommendations, and PPO simulations should be interpreted "
    "within their respective model assumptions. Simulated outcomes "
    "are scenarios, not guaranteed business results."
)
