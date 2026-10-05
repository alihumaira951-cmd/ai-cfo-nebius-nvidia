import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from services.chapter4 import (
    classify_regime,
    compute_financial_health_score,
    rank_recommendations,
)

from services.chapter5 import explain_recommendation
from services.chapter6 import run_custom_company_simulation


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="New Company Analysis",
    page_icon="🏢",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "deployment_artifacts/models/xgb_distress_h1.joblib"
REVENUE_MODEL_PATH = ROOT / "deployment_artifacts/models/xgb_Revenue_H1.joblib"
OCF_MODEL_PATH = ROOT / "deployment_artifacts/models/xgb_Operating Cash Flow_H1.joblib"
EBITDA_MODEL_PATH = ROOT / "deployment_artifacts/models/xgb_EBITDA_H1.joblib"

PANEL_PATH = ROOT / "deployment_artifacts/data/chapter3_deployment_panel.parquet"
SIC_PATH = ROOT / "deployment_artifacts/data/sic_code_mapping.csv"

DISTRESS_THRESHOLD = 0.77


# ============================================================
# VERIFIED SEC EXAMPLE
# ============================================================

AXOGEN_SEC_EXAMPLE = {
    "company_name": "AXOGEN, INC.",
    "year": 2022,
    "quarter": 3,

    # Official standalone Q3 2022 SEC financial data
    "revenue": 36_959_000.0,
    "cogs": 6_176_000.0,
    "gross_profit": 30_783_000.0,
    "operating_expenses": 35_638_000.0,
    "operating_income": -4_855_000.0,
    "net_income": -4_318_000.0,
    "operating_cash_flow": -689_000.0,
    "capex": 4_370_000.0,
    "rd_expense": 7_050_000.0,
    "sga_expense": 28_588_000.0,

    "cash": 14_318_000.0,
    "accounts_receivable": 21_363_000.0,
    "inventory": 19_116_000.0,
    "current_assets": 102_454_000.0,
    "total_assets": 195_520_000.0,
    "equity": 101_445_000.0,
    "accounts_payable": 22_017_000.0,
    "current_liabilities": 23_547_000.0,
    "total_liabilities": 94_075_000.0,
    "total_debt": 45_487_000.0,
    "shares_outstanding": 42_272_223.0,
    "sic": 3845,

    # Official Q2 2022 SEC values used for growth features
    "previous_revenue": 34_454_000.0,
    "previous_total_assets": 196_084_000.0,
    "previous_cash": 11_822_000.0,

    # Actual standalone Q4 2022 outcome for comparison
    "actual_q4_revenue": 36_164_000.0,
    "actual_q4_cogs": 6_141_000.0,
    "actual_q4_gross_profit": 30_023_000.0,
    "actual_q4_operating_expenses": 35_631_000.0,
    "actual_q4_operating_income": -5_608_000.0,
    "actual_q4_net_income": -5_415_000.0,
    "actual_q4_operating_cash_flow": 1_362_000.0,
    "actual_q4_capex": 6_622_000.0,
}


# ============================================================
# LOAD DEPLOYMENT ARTIFACTS
# ============================================================

@st.cache_resource
def load_models():
    return {
        "distress": joblib.load(MODEL_PATH),
        "revenue": joblib.load(REVENUE_MODEL_PATH),
        "ocf": joblib.load(OCF_MODEL_PATH),
        "ebitda": joblib.load(EBITDA_MODEL_PATH),
    }


@st.cache_data
def load_panel():
    return pd.read_parquet(PANEL_PATH)


@st.cache_data
def load_sic_mapping():
    mapping = pd.read_csv(SIC_PATH)
    mapping["sic"] = mapping["sic"].astype(str)
    return mapping


models = load_models()
deployment_panel = load_panel()
sic_mapping = load_sic_mapping()

model = models["distress"]
revenue_model = models["revenue"]
ocf_model = models["ocf"]
ebitda_model = models["ebitda"]

model_features = model.get_booster().feature_names


# ============================================================
# MACRO / MARKET FEATURES
# ============================================================

MACRO_MARKET_COLS = [
    "real_gdp",
    "real_gdp_growth_rate",
    "industrial_production",
    "cpi_all_items",
    "cpi_core",
    "ppi",
    "fed_funds_rate",
    "prime_rate",
    "baa_credit_spread",
    "yield_curve_spread",
    "unemployment_rate",
    "nonfarm_payrolls",
    "avg_hourly_earnings",
    "consumer_sentiment",
    "retail_sales",
    "cpi_all_items_yoy",
    "retail_sales_yoy",
    "nonfarm_payrolls_yoy",
    "avg_hourly_earnings_yoy",
    "industrial_production_yoy",
    "fed_funds_rate_mom",
    "baa_credit_spread_mom",
    "consumer_sentiment_mom",
    "yield_curve_inverted",
    "financial_conditions_proxy",
    "sp500_etf",
    "vix_volatility",
    "ten_year_treasury_yield",
    "nasdaq_etf",
    "russell2000_etf",
    "financial_sector_etf",
    "sp500_etf_return_1m",
    "sp500_etf_return_3m",
    "sp500_etf_volatility_6m",
    "vix_volatility_return_1m",
    "vix_volatility_return_3m",
    "vix_volatility_volatility_6m",
    "ten_year_treasury_yield_return_1m",
    "ten_year_treasury_yield_return_3m",
    "ten_year_treasury_yield_volatility_6m",
    "nasdaq_etf_return_1m",
    "nasdaq_etf_return_3m",
    "nasdaq_etf_volatility_6m",
    "russell2000_etf_return_1m",
    "russell2000_etf_return_3m",
    "russell2000_etf_volatility_6m",
    "financial_sector_etf_return_1m",
    "financial_sector_etf_return_3m",
    "financial_sector_etf_volatility_6m",
]


# ============================================================
# HELPERS
# ============================================================

def safe_divide(a, b):
    if b is None or b == 0:
        return 0.0
    return a / b


def growth_rate(current, previous):
    if previous is None or previous == 0:
        return 0.0
    return (current - previous) / abs(previous)


# ============================================================
# HEADER
# ============================================================

st.title("🏢 New Company Analysis")

st.caption(
    "Enter your company's financial information and let AI CFO build "
    "a model-ready financial profile and estimate next-quarter financial distress risk."
)

st.info(
    "Custom analysis currently supports reporting quarters from 2021–2022 "
    "because those are the macroeconomic and market periods packaged with "
    "the deployed Chapter 3 model."
)


# ============================================================
# INPUT FORM
# ============================================================

if "load_axogen_sec_example" not in st.session_state:
    st.session_state["load_axogen_sec_example"] = False


if st.button(
    "Load SEC Example: Axogen 2022 Q3",
    use_container_width=True,
):
    st.session_state["load_axogen_sec_example"] = True

    for key, value in AXOGEN_SEC_EXAMPLE.items():
        if not key.startswith("actual_q4_"):
            st.session_state[key] = value

    st.rerun()


if st.session_state.get("load_axogen_sec_example"):
    st.success(
        "Loaded verified Axogen 2022 Q3 SEC financial data. "
        "You can review or edit any field before running the analysis."
    )


with st.form("company_input_form"):

    st.subheader("1. Company Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        company_name = st.text_input(
            "Company name",
            value=st.session_state.get("company_name", ""),
            placeholder="Example: ABC Manufacturing"
        )

    with col2:
        year = st.selectbox(
            "Reporting year",
            [2021, 2022],
            index=[2021, 2022].index(
                st.session_state.get("year", 2021)
            )
        )

    with col3:
        quarter = st.selectbox(
            "Reporting quarter",
            [1, 2, 3, 4],
            index=[1, 2, 3, 4].index(
                st.session_state.get("quarter", 1)
            ),
            format_func=lambda x: f"Q{x}"
        )

    st.divider()

    st.subheader("2. Revenue & Operations")

    col1, col2, col3 = st.columns(3)

    with col1:
        revenue = st.number_input(
            "Revenue ($)",
            value=st.session_state.get("revenue", 0),
            min_value=0.0,
            format="%.2f"
        )

        cogs = st.number_input(
            "Cost of goods sold ($)",
            value=st.session_state.get("cogs", 0),
            min_value=0.0,
            format="%.2f"
        )

        gross_profit = st.number_input(
            "Gross profit ($)",
            value=st.session_state.get("gross_profit", 0),
            format="%.2f"
        )

    with col2:
        operating_expenses = st.number_input(
            "Operating expenses ($)",
            value=st.session_state.get("operating_expenses", 0),
            min_value=0.0,
            format="%.2f"
        )

        operating_income = st.number_input(
            "Operating income ($)",
            value=st.session_state.get("operating_income", 0),
            format="%.2f"
        )

        net_income = st.number_input(
            "Net income ($)",
            value=st.session_state.get("net_income", 0),
            format="%.2f"
        )

    with col3:
        operating_cash_flow = st.number_input(
            "Operating cash flow ($)",
            value=st.session_state.get("operating_cash_flow", 0),
            format="%.2f"
        )

        capex = st.number_input(
            "Capital expenditures ($)",
            value=st.session_state.get("capex", 0),
            min_value=0.0,
            format="%.2f"
        )

        rd_expense = st.number_input(
            "R&D expense ($)",
            value=st.session_state.get("rd_expense", 0),
            min_value=0.0,
            format="%.2f"
        )

        sga_expense = st.number_input(
            "SG&A expense ($)",
            value=st.session_state.get("sga_expense", 0),
            min_value=0.0,
            format="%.2f"
        )

    st.divider()

    st.subheader("3. Balance Sheet")

    col1, col2, col3 = st.columns(3)

    with col1:
        cash = st.number_input(
            "Cash ($)",
            value=st.session_state.get("cash", 0),
            min_value=0.0,
            format="%.2f"
        )

        accounts_receivable = st.number_input(
            "Accounts receivable ($)",
            value=st.session_state.get("accounts_receivable", 0),
            min_value=0.0,
            format="%.2f"
        )

        inventory = st.number_input(
            "Inventory ($)",
            value=st.session_state.get("inventory", 0),
            min_value=0.0,
            format="%.2f"
        )

    with col2:
        current_assets = st.number_input(
            "Current assets ($)",
            value=st.session_state.get("current_assets", 0),
            min_value=0.0,
            format="%.2f"
        )

        total_assets = st.number_input(
            "Total assets ($)",
            value=st.session_state.get("total_assets", 0),
            min_value=0.0,
            format="%.2f"
        )

        equity = st.number_input(
            "Shareholders' equity ($)",
            value=st.session_state.get("equity", 0),
            format="%.2f"
        )

    with col3:
        accounts_payable = st.number_input(
            "Accounts payable ($)",
            value=st.session_state.get("accounts_payable", 0),
            min_value=0.0,
            format="%.2f"
        )

        current_liabilities = st.number_input(
            "Current liabilities ($)",
            value=st.session_state.get("current_liabilities", 0),
            min_value=0.0,
            format="%.2f"
        )

        total_liabilities = st.number_input(
            "Total liabilities ($)",
            value=st.session_state.get("total_liabilities", 0),
            min_value=0.0,
            format="%.2f"
        )

        total_debt = st.number_input(
            "Total debt ($)",
            value=st.session_state.get("total_debt", 0),
            min_value=0.0,
            format="%.2f"
        )

    st.divider()

    st.subheader("4. Additional Company Data")

    col1, col2 = st.columns(2)

    with col1:
        shares_outstanding = st.number_input(
            "Shares outstanding",
            value=st.session_state.get("shares_outstanding", 0),
            min_value=0.0,
            format="%.0f"
        )

    with col2:
        sic = st.number_input(
            "SIC code",
            value=st.session_state.get("sic", 0),
            min_value=0,
            max_value=9999,
            step=1,
            help="Enter the company's standard 4-digit SIC code."
        )

    st.divider()

    st.subheader("5. Prior-Quarter Information")

    st.caption(
        "These values allow AI CFO to calculate company growth features automatically."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        previous_revenue = st.number_input(
            "Previous-quarter revenue ($)",
            value=st.session_state.get("previous_revenue", 0),
            min_value=0.0,
            format="%.2f"
        )

    with col2:
        previous_total_assets = st.number_input(
            "Previous-quarter total assets ($)",
            value=st.session_state.get("previous_total_assets", 0),
            min_value=0.0,
            format="%.2f"
        )

    with col3:
        previous_cash = st.number_input(
            "Previous-quarter cash ($)",
            value=st.session_state.get("previous_cash", 0),
            min_value=0.0,
            format="%.2f"
        )

    submitted = st.form_submit_button(
        "Run AI CFO Analysis",
        use_container_width=True
    )


# ============================================================
# ANALYSIS
# ============================================================

if submitted:

    entered_sic = str(int(sic))

    sic_match = sic_mapping[
        sic_mapping["sic"] == entered_sic
    ]

    quarter_rows = deployment_panel[
        (deployment_panel["year"] == year)
        & (deployment_panel["quarter"] == quarter)
    ]

    if not company_name.strip():

        st.error("Please enter a company name.")

    elif revenue <= 0 or total_assets <= 0 or current_liabilities <= 0:

        st.error(
            "Revenue, total assets, and current liabilities must be greater than zero."
        )

    elif sic_match.empty:

        st.error(
            "This SIC code was not represented in the model development data. "
            "Please enter a SIC code supported by the deployed model."
        )

    elif quarter_rows.empty:

        st.error(
            "Macroeconomic and market data are unavailable for the selected quarter."
        )

    else:

        # ----------------------------------------------------
        # Industry encoding
        # ----------------------------------------------------

        sic_code = int(
            sic_match["sic_code"].iloc[0]
        )

        # ----------------------------------------------------
        # Derived company features
        # ----------------------------------------------------

        current_ratio = safe_divide(
            current_assets,
            current_liabilities
        )

        cash_ratio = safe_divide(
            cash,
            current_liabilities
        )

        debt_to_assets = safe_divide(
            total_debt,
            total_assets
        )

        debt_to_equity = safe_divide(
            total_debt,
            equity
        )

        net_margin = safe_divide(
            net_income,
            revenue
        )

        operating_margin = safe_divide(
            operating_income,
            revenue
        )

        return_on_assets = safe_divide(
            net_income,
            total_assets
        )

        return_on_equity = safe_divide(
            net_income,
            equity
        )

        asset_turnover = safe_divide(
            revenue,
            total_assets
        )

        inventory_turnover = safe_divide(
            cogs,
            inventory
        )

        ocf_to_debt = safe_divide(
            operating_cash_flow,
            total_debt
        )

        ocf_to_assets = safe_divide(
            operating_cash_flow,
            total_assets
        )

        revenue_growth = growth_rate(
            revenue,
            previous_revenue
        )

        asset_growth = growth_rate(
            total_assets,
            previous_total_assets
        )

        cash_growth = growth_rate(
            cash,
            previous_cash
        )

        # ----------------------------------------------------
        # Quarter-specific macro / market state
        # ----------------------------------------------------

        macro_row = quarter_rows.iloc[0]

        # ----------------------------------------------------
        # Build exact Chapter 3 feature row
        # ----------------------------------------------------

        feature_values = {
            "accounts_payable": accounts_payable,
            "accounts_receivable": accounts_receivable,
            "capex": capex,
            "cash": cash,
            "cogs": cogs,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "equity": equity,
            "gross_profit": gross_profit,
            "inventory": inventory,
            "net_income": net_income,
            "operating_cash_flow": operating_cash_flow,
            "operating_expenses": operating_expenses,
            "operating_income": operating_income,
            "rd_expense": rd_expense,
            "revenue": revenue,
            "sga_expense": sga_expense,
            "shares_outstanding": shares_outstanding,
            "total_assets": total_assets,
            "total_debt": total_debt,
            "total_liabilities": total_liabilities,
            "year": year,
            "quarter": quarter,
            "current_ratio": current_ratio,
            "cash_ratio": cash_ratio,
            "debt_to_assets": debt_to_assets,
            "debt_to_equity": debt_to_equity,
            "net_margin": net_margin,
            "operating_margin": operating_margin,
            "return_on_assets": return_on_assets,
            "return_on_equity": return_on_equity,
            "asset_turnover": asset_turnover,
            "inventory_turnover": inventory_turnover,
            "ocf_to_debt": ocf_to_debt,
            "ocf_to_assets": ocf_to_assets,
            "revenue_growth": revenue_growth,
            "asset_growth": asset_growth,
            "cash_growth": cash_growth,
            "sic_code": sic_code,
        }

        for col in MACRO_MARKET_COLS:
            feature_values[col] = macro_row[col]

        model_input = pd.DataFrame(
            [[feature_values[col] for col in model_features]],
            columns=model_features
        )

        model_input = model_input.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # ----------------------------------------------------
        # Model validation
        # ----------------------------------------------------

        missing_features = [
            col
            for col in model_features
            if col not in model_input.columns
        ]

        if missing_features:

            st.error(
                "The model-ready row is missing required features: "
                + ", ".join(missing_features)
            )

        elif model_input.isna().any().any():

            missing_values = (
                model_input.columns[
                    model_input.isna().any()
                ]
                .tolist()
            )

            st.error(
                "Some model features could not be created: "
                + ", ".join(missing_values)
            )

        else:

            # ------------------------------------------------
            # Frozen Chapter 3 distress prediction
            # ------------------------------------------------

            distress_probability = float(
                model.predict_proba(model_input)[0, 1]
            )

            distress_class = int(
                distress_probability >= DISTRESS_THRESHOLD
            )

            # ------------------------------------------------
            # Chapter 4 forecast models
            # ------------------------------------------------

            revenue_features = list(revenue_model.feature_names_in_)
            ocf_features = list(ocf_model.feature_names_in_)
            ebitda_features = list(ebitda_model.feature_names_in_)

            forecast_revenue = float(
                np.expm1(
                    revenue_model.predict(
                        model_input[revenue_features]
                    )[0]
                )
            )

            forecast_ocf = float(
                ocf_model.predict(
                    model_input[ocf_features]
                )[0]
            )

            forecast_ebitda = float(
                ebitda_model.predict(
                    model_input[ebitda_features]
                )[0]
            )

            # ------------------------------------------------
            # Chapter 4 financial health
            # ------------------------------------------------

            health_input = {
                "current_ratio": current_ratio,
                "cash_ratio": cash_ratio,
                "ocf_to_assets": ocf_to_assets,
                "debt_to_assets": debt_to_assets,
                "debt_to_equity": debt_to_equity,
                "ocf_to_debt": ocf_to_debt,
                "net_margin": net_margin,
                "operating_margin": operating_margin,
                "return_on_assets": return_on_assets,
                "return_on_equity": return_on_equity,
                "revenue_growth": revenue_growth,
                "asset_growth": asset_growth,
                "cash_growth": cash_growth,
                "asset_turnover": asset_turnover,
                "inventory_turnover": inventory_turnover,
            }

            health_scores = compute_financial_health_score(
                health_input
            )

            financial_health = health_scores[
                "Financial_Health_Score"
            ]

            strategic_regime = classify_regime(
                distress_probability,
                financial_health,
            )

            decision_row = {
                **health_scores,
                "Distress_Probability": distress_probability,
                "Forecast_OCF": forecast_ocf,
                "Forecast_EBITDA": forecast_ebitda,
                "Strategic_Regime": strategic_regime,
            }

            recommendations = rank_recommendations(
                decision_row,
                top_n=3,
            )

            # ------------------------------------------------
            # OUTPUT
            # ------------------------------------------------

            st.success(
                f"AI CFO analysis completed for "
                f"{company_name} — {year} Q{quarter}."
            )

            st.caption(
                f"SIC {entered_sic} mapped internally to "
                f"model industry code {sic_code}."
            )

            st.subheader("Next-Quarter Distress Prediction")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Distress Probability",
                f"{distress_probability:.2%}"
            )

            col2.metric(
                "Model Threshold",
                f"{DISTRESS_THRESHOLD:.0%}"
            )

            col3.metric(
                "Risk Classification",
                "DISTRESS" if distress_class == 1 else "NON-DISTRESS"
            )

            if distress_class == 1:

                st.warning(
                    "The frozen Chapter 3 model classifies this company "
                    "state as elevated next-quarter financial distress risk."
                )

            else:

                st.success(
                    "The frozen Chapter 3 model does not classify this company "
                    "state as next-quarter financial distress at the selected threshold."
                )

            st.divider()

            st.subheader("Strategic Decision Profile")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Financial Health Score",
                f"{financial_health:.3f}"
            )

            col2.metric(
                "Strategic Regime",
                strategic_regime
            )

            col3.metric(
                "Forecast OCF",
                f"${forecast_ocf:,.0f}"
            )

            col4.metric(
                "Forecast EBITDA",
                f"${forecast_ebitda:,.0f}"
            )

            st.caption(
                f"Next-quarter revenue forecast: ${forecast_revenue:,.0f}"
            )

            # ------------------------------------------------
            # Verified SEC next-quarter comparison
            # ------------------------------------------------

            is_verified_axogen_example = (
                company_name.strip().upper()
                == AXOGEN_SEC_EXAMPLE["company_name"]
                and year == AXOGEN_SEC_EXAMPLE["year"]
                and quarter == AXOGEN_SEC_EXAMPLE["quarter"]
                and np.isclose(
                    revenue,
                    AXOGEN_SEC_EXAMPLE["revenue"]
                )
                and np.isclose(
                    operating_cash_flow,
                    AXOGEN_SEC_EXAMPLE["operating_cash_flow"]
                )
                and np.isclose(
                    total_assets,
                    AXOGEN_SEC_EXAMPLE["total_assets"]
                )
                and int(sic) == AXOGEN_SEC_EXAMPLE["sic"]
            )

            if is_verified_axogen_example:

                st.subheader(
                    "SEC Actual Next-Quarter Comparison"
                )

                st.caption(
                    "AI CFO uses Axogen's official 2022 Q3 SEC financial "
                    "state as input. The comparison below shows the model's "
                    "next-quarter forecasts alongside Axogen's subsequently "
                    "reported standalone Q4 2022 results."
                )

                comparison_df = pd.DataFrame({
                    "Metric": [
                        "Revenue",
                        "Operating Cash Flow",
                    ],
                    "AI CFO Forecast": [
                        forecast_revenue,
                        forecast_ocf,
                    ],
                    "Actual SEC Q4": [
                        AXOGEN_SEC_EXAMPLE[
                            "actual_q4_revenue"
                        ],
                        AXOGEN_SEC_EXAMPLE[
                            "actual_q4_operating_cash_flow"
                        ],
                    ],
                })

                comparison_df["Forecast Error"] = (
                    comparison_df["AI CFO Forecast"]
                    - comparison_df["Actual SEC Q4"]
                )

                comparison_df["Absolute % Error"] = (
                    (
                        comparison_df["Forecast Error"].abs()
                        / comparison_df[
                            "Actual SEC Q4"
                        ].abs()
                    )
                    * 100
                )

                formatted_comparison = (
                    comparison_df.copy()
                )

                for col in [
                    "AI CFO Forecast",
                    "Actual SEC Q4",
                    "Forecast Error",
                ]:
                    formatted_comparison[col] = (
                        formatted_comparison[col]
                        .map(
                            lambda x: f"${x:,.0f}"
                        )
                    )

                formatted_comparison[
                    "Absolute % Error"
                ] = (
                    formatted_comparison[
                        "Absolute % Error"
                    ]
                    .map(
                        lambda x: f"{x:.1f}%"
                    )
                )

                st.dataframe(
                    formatted_comparison,
                    width="stretch",
                    hide_index=True,
                )

                st.markdown(
                    "**Additional actual Q4 SEC outcomes:**  "
                    f"Operating income: "
                    f"${AXOGEN_SEC_EXAMPLE['actual_q4_operating_income']:,.0f}  |  "
                    f"Net income: "
                    f"${AXOGEN_SEC_EXAMPLE['actual_q4_net_income']:,.0f}  |  "
                    f"Capital expenditures: "
                    f"${AXOGEN_SEC_EXAMPLE['actual_q4_capex']:,.0f}"
                )

                st.caption(
                    "The SEC comparison is an illustrative historical case study. "
                    "Forecast errors are shown transparently and should not be "
                    "interpreted as general model performance."
                )

            st.subheader("Top Recommended Actions")

            for rank, (action, score) in enumerate(
                recommendations,
                start=1
            ):
                st.write(
                    f"**{rank}. {action}** — priority score: {score:.3f}"
                )

            st.caption(
                "Recommendation scores represent relative model-based "
                "decision priority, not the probability that an action will succeed."
            )

            # ------------------------------------------------
            # Chapter 5 explainability for top recommendation
            # ------------------------------------------------

            top_action = recommendations[0][0]

            xai_result = explain_recommendation(
                action=top_action,
                financial_health=financial_health,
                distress_probability=distress_probability,
                strategic_regime=strategic_regime,
            )

            st.subheader("Why This Recommendation?")

            st.metric(
                "Chapter 5 Surrogate Probability",
                f"{xai_result['Surrogate_Probability']:.2%}"
            )

            st.caption(
                "This probability reflects the Chapter 5 surrogate model's "
                "alignment with the recommended action. It is not the probability "
                "that the action will succeed."
            )

            contributions = xai_result["Contributions"].copy()

            st.dataframe(
                contributions[
                    ["Feature", "Value", "SHAP_Contribution"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "SHAP values show predictive contribution within the surrogate model. "
                "They should not be interpreted as causal effects."
            )

            # ------------------------------------------------
            # Chapter 6 scenario simulation
            # ------------------------------------------------

            quarter_end_dates = {
                1: f"{year}-03-31",
                2: f"{year}-06-30",
                3: f"{year}-09-30",
                4: f"{year}-12-31",
            }

            simulation_result = run_custom_company_simulation(
                company_name=company_name,
                period=quarter_end_dates[int(quarter)],
                revenue=revenue,
                gross_profit=gross_profit,
                operating_expenses=operating_expenses,
                operating_income=operating_income,
                net_income=net_income,
                operating_cash_flow=operating_cash_flow,
                sga_expense=sga_expense,
                cash=cash,
                current_assets=current_assets,
                current_liabilities=current_liabilities,
                total_assets=total_assets,
                total_debt=total_debt,
                equity=equity,
                revenue_growth=revenue_growth,
                current_ratio=current_ratio,
                cash_ratio=cash_ratio,
                debt_to_assets=debt_to_assets,
                debt_to_equity=debt_to_equity,
                return_on_assets=return_on_assets,
                asset_turnover=asset_turnover,
                operating_margin=operating_margin,
                distress_probability=distress_probability,
                fed_funds_rate=float(
                    macro_row["fed_funds_rate"]
                ),
                baa_credit_spread=float(
                    macro_row["baa_credit_spread"]
                ),
                consumer_sentiment=float(
                    macro_row["consumer_sentiment"]
                ),
                yield_curve_inverted=float(
                    macro_row["yield_curve_inverted"]
                ),
                quarter=int(quarter),
                sic=int(sic),
            )

            if simulation_result.get("status") == "success":

                st.subheader(
                    "Chapter 6 Strategic Scenario Simulation"
                )

                summary = simulation_result["summary"]

                sim_col1, sim_col2, sim_col3 = st.columns(3)

                sim_col1.metric(
                    "Initial Distress",
                    f"{summary['initial_distress_probability']:.2%}"
                )

                sim_col2.metric(
                    "Simulated Final Distress",
                    f"{summary['final_distress_probability']:.2%}"
                )

                sim_col3.metric(
                    "Economic Regime",
                    simulation_result[
                        "initial_economic_regime"
                    ].replace("_", " ").title()
                )

                trajectory_df = pd.DataFrame(
                    simulation_result["trajectory"]
                )

                if not trajectory_df.empty:

                    display_trajectory = trajectory_df[
                        [
                            "quarter",
                            "strategy",
                            "distress_probability_before",
                            "distress_probability_after",
                            "reward",
                        ]
                    ].copy()

                    display_trajectory.columns = [
                        "Simulation Quarter",
                        "PPO Strategy",
                        "Distress Before",
                        "Distress After",
                        "Reward",
                    ]

                    display_trajectory[
                        "Distress Before"
                    ] = display_trajectory[
                        "Distress Before"
                    ].map(
                        lambda x: f"{x:.2%}"
                    )

                    display_trajectory[
                        "Distress After"
                    ] = display_trajectory[
                        "Distress After"
                    ].map(
                        lambda x: f"{x:.2%}"
                    )

                    st.dataframe(
                        display_trajectory,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.caption(
                    "This four-quarter trajectory is generated by the "
                    "Chapter 6 PPO policy inside the SMBGym simulated "
                    "financial environment. It is scenario-based decision "
                    "support, not a guaranteed forecast of real-world outcomes."
                )

            else:
                st.warning(
                    "Chapter 6 scenario simulation could not be generated."
                )

                st.error(
                    f"Status: {simulation_result.get('status', 'unknown')} | "
                    f"Message: {simulation_result.get('message', 'No message returned')}"
                )

            st.divider()

            st.subheader("Derived Financial Profile")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Current Ratio",
                f"{current_ratio:.2f}"
            )

            col2.metric(
                "Cash Ratio",
                f"{cash_ratio:.2f}"
            )

            col3.metric(
                "Debt / Assets",
                f"{debt_to_assets:.2%}"
            )

            col4.metric(
                "Net Margin",
                f"{net_margin:.2%}"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Operating Margin",
                f"{operating_margin:.2%}"
            )

            col2.metric(
                "Return on Assets",
                f"{return_on_assets:.2%}"
            )

            col3.metric(
                "Asset Turnover",
                f"{asset_turnover:.2f}"
            )

            col4.metric(
                "OCF / Assets",
                f"{ocf_to_assets:.2%}"
            )

            st.subheader("Growth Indicators")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Revenue Growth",
                f"{revenue_growth:.2%}"
            )

            col2.metric(
                "Asset Growth",
                f"{asset_growth:.2%}"
            )

            col3.metric(
                "Cash Growth",
                f"{cash_growth:.2%}"
            )

            with st.expander("Model Input Audit"):

                st.write(
                    f"All {len(model_features)} frozen Chapter 3 "
                    "model features were successfully constructed."
                )

                st.dataframe(
                    model_input.T.rename(
                        columns={0: "Value"}
                    ),
                    use_container_width=True
                )

            st.info(
                "The analysis combines the frozen Chapter 3 predictive model, "
                "Chapter 4 forecasting and recommendation logic, Chapter 5 "
                "surrogate explainability, and Chapter 6 PPO scenario simulation."
            )
