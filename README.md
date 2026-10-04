# AI CFO

### Explainable Agentic Decision Intelligence for Financial Management

**Predict → Recommend → Explain → Simulate → Decide**

🌐 **Live Demo:**  
(https://ai-cfo-nebius-nvidia-bgkhxcxxkfcu2qucsmwbgw.streamlit.app/)

---

## Overview

AI CFO is an explainable decision-intelligence system designed to help move financial AI beyond prediction alone.

Traditional financial models can estimate what may happen. AI CFO extends that workflow by connecting financial distress prediction with strategic recommendations, explainable AI, multi-quarter reinforcement-learning simulation, and executive decision support.

The system combines several analytical layers into one decision pipeline:

1. **Predict** financial distress risk.
2. **Recommend** governance-aware strategic actions.
3. **Explain** why a recommendation was generated.
4. **Simulate** the potential trajectory of the strategy across multiple quarters.
5. **Decide** by synthesizing the evidence into an executive-level decision brief.

---

## Why AI CFO?

Financial decision-makers rarely need only a probability.

They need to understand:

- What risk is emerging?
- What should we do about it?
- Why is that action being recommended?
- What could happen if we follow that strategy?
- What tradeoffs or governance concerns should management consider?

AI CFO brings these questions into one integrated workflow.

---

## System Architecture

### Chapter 3 — Predict

An XGBoost financial-distress model estimates next-quarter financial distress probability.

The prediction layer produces:

- Financial distress probability
- Distress classification
- Decision threshold
- Model feature information

---

### Chapter 4 — Recommend

The recommendation layer converts the company's financial condition into a strategic regime and ranks candidate management actions.

Examples include:

- Refinance Debt
- Improve Working Capital Efficiency
- Reduce Operating Expenses
- Increase Cash Reserves
- Delay Capital Expenditures
- Hold Strategy

Recommendations incorporate financial health, distress risk, and governance-aware scoring.

---

### Chapter 5 — Explain

SHAP-based surrogate models explain the factors influencing the recommended action.

The explainability layer identifies:

- Important recommendation drivers
- Direction of model contribution
- Financial health effects
- Distress-risk effects
- Strategic-regime effects

SHAP values are treated as **predictive explanations**, not causal effects.

---

### Chapter 6 — Simulate

A PPO reinforcement-learning policy operates within a custom SMBGym financial environment.

The simulation evaluates a strategy across four modeled quarters and tracks outcomes such as:

- Profit margin
- Cash ratio
- Current ratio
- Debt ratio
- Revenue growth
- Financial distress probability

The simulation represents **scenario-based decision support** based on modeled transition assumptions. It should not be interpreted as a guaranteed real-world financial forecast.

---

## Forward Validation

The hackathon implementation includes a dedicated **Forward Validation** workflow that evaluates the frozen research pipeline on later, post-training financial states.

### Technical Communications Corporation case study

- **Input state:** March 31, 2022
- **Next-quarter distress probability:** 94.39%
- **Observed next quarter:** June 30, 2022
- **Observed distress outcome:** Distress criteria met
- **Financial Health Score:** 0.230
- **Strategic Regime:** Defensive
- **Top recommendation:** Reduce Operating Expenses
- **Chapter 5 surrogate probability:** 64.30%

The trained PPO policy was also initialized from the newly constructed March 2022 Chapter 6 state for a four-quarter scenario simulation.

In that modeled scenario:

- Profit margin moved from **-21.72% to 11.67%**
- Cash ratio moved from **0.080 to 0.804**
- Distress probability remained elevated at approximately **94.68%**

The simulation is interpreted as a **mixed scenario**, not as a guaranteed real-world outcome.

### Walk-forward evidence

Across five consecutive next-quarter observations for Technical Communications Corporation, the frozen distress classifier correctly classified **4 of 5 outcomes (80%)**.

This is an illustrative company-level, out-of-time case study and should not be interpreted as general model accuracy or as validation of the recommendation and simulation outputs.

Use the **Forward Validation** page in the Streamlit sidebar to view the live case study.


## Executive Decision Layer

The final layer brings together evidence from prediction, recommendation, explainability, and simulation.

The executive interface is designed to distinguish between:

- Predictive model outputs
- Recommendation scores
- SHAP explanations
- Simulated scenarios
- Executive interpretation

This separation is important because AI CFO is intended to support human decision-making rather than hide analytical uncertainty behind a single automated answer.

---

## NVIDIA Nemotron + Nebius Integration

AI CFO contains an executive reasoning layer designed to use **NVIDIA Nemotron through Nebius Token Factory**.

The reasoning layer receives structured evidence produced by the underlying financial tools and is instructed to:

- Use only evidence generated by the analytical pipeline
- Avoid inventing financial values
- Distinguish prediction from simulation
- Avoid treating SHAP explanations as causal evidence
- Highlight mixed or conflicting outcomes
- Communicate uncertainty and governance considerations

### Current public demo status

The public Streamlit deployment currently operates in **local fallback mode** because live Nebius credentials have not yet been configured.

The application explicitly displays this status and does **not** claim that live Nemotron inference is occurring when credentials are unavailable.

The Nemotron/Nebius service integration is contained in:

```text
services/nebius.py
