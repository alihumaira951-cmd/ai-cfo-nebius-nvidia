# AI CFO

# AI CFO

### Explainable Agentic Decision Intelligence for Financial Management

**Predict → Recommend → Explain → Simulate → Decide**

AI CFO is a multi-stage financial decision-intelligence system that combines predictive modeling, governance-aware recommendations, SHAP explainability, reinforcement-learning simulation, and **live NVIDIA Nemotron executive reasoning through Nebius Token Factory**.

### Hackathon Architecture

**Financial Data → XGBoost Distress Prediction → Strategic Recommendation → SHAP Explanation → PPO Scenario Simulation → NVIDIA Nemotron on Nebius Token Factory → Executive Decision Brief**

🌐 **Live Demo:**  
https://ai-cfo-nebius-nvidia-bgkhxcxxkfcu2qucsmwbgw.streamlit.app/

🧠 **Executive Reasoning:** NVIDIA Nemotron-3-Super-120B-A12B  
☁️ **Inference Platform:** Nebius Token Factory

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

---

## New Company Analysis

The hackathon implementation now includes a business-facing **New Company Analysis** workflow.

Instead of requiring users to construct the full model feature vector manually, users enter familiar company financial information such as:

- Revenue and expenses
- Net income
- Operating cash flow
- Cash and working-capital accounts
- Assets and liabilities
- Debt and equity
- Shares outstanding
- SIC industry code
- Prior-quarter revenue, assets, and cash

AI CFO then automatically:

1. Derives financial ratios and growth measures.
2. Maps the company's SIC to the model industry encoding.
3. Adds the quarter-specific macroeconomic and market state.
4. Constructs the Chapter 3 model input.
5. Estimates next-quarter distress risk.
6. Produces Chapter 4 forecasts and strategic recommendations.
7. Explains the top recommendation using the Chapter 5 surrogate model and SHAP.
8. Constructs the 24-variable Chapter 6 state.
9. Runs the trained PPO policy through a four-quarter SMBGym scenario simulation.

This turns the research pipeline into a workflow that a business user can interact with directly rather than requiring notebook or coding access.

### Verified SEC Example — Axogen 2022 Q3

The page includes a one-click **Load SEC Example: Axogen 2022 Q3** option using financial information reconstructed from Axogen's official SEC filings.

For this case, AI CFO produces:

- **Distress probability:** 0.58%
- **Financial Health Score:** 0.392
- **Strategic Regime:** Growth
- **Top recommendation:** Improve Working Capital Efficiency
- **Chapter 5 surrogate probability:** 96.66%
- **Chapter 6 simulated distress:** approximately 0.58% to 0.61%

The application also compares selected AI CFO forecasts with Axogen's subsequently reported standalone Q4 2022 results:

| Metric | AI CFO Forecast | Actual SEC Q4 |
| --- | ---: | ---: |
| Revenue | $56.95M | $36.16M |
| Operating Cash Flow | -$1.33M | $1.36M |

The forecast misses are intentionally shown rather than hidden. This SEC example is an illustrative historical case study and should not be interpreted as general forecast-performance evidence.

---

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
```

---

## Hackathon Contribution

The underlying predictive, recommendation, explainability, and reinforcement-learning models originate from the AI CFO doctoral research project.

The hackathon work focuses on turning those research components into an integrated, usable decision-intelligence product.

Hackathon development includes:

- Streamlit deployment
- Historical decision dashboard
- Multi-layer model orchestration
- Out-of-time forward-validation workflow
- Five-quarter walk-forward validation extension
- New-company business input workflow
- Automatic model feature construction
- Verified SEC example workflow
- Actual next-quarter SEC comparison
- Chapter 5 explainability integration
- Chapter 6 custom-state construction for new companies
- PPO scenario-simulation integration
- Nebius/Nemotron service layer
- Executive decision-interface design

## Important Interpretation Boundaries

AI CFO is a research-based decision-support prototype.

- Financial distress outputs are **model predictions**, not certainties.
- Recommendation scores are **relative priorities**, not probabilities of success.
- SHAP values explain predictive model behavior and are **not causal estimates**.
- Reinforcement-learning trajectories are **simulated scenarios**, not guaranteed forecasts.
- The Technical Communications walk-forward result is an **illustrative company-level case study**, not overall model accuracy.
- The Axogen SEC comparison is a **historical example**, not general forecast-performance evidence.
- Custom company analysis currently supports **2021–2022 reporting quarters** because those macroeconomic and market states are packaged with the deployed model.
- The public deployment currently uses a **local executive-reasoning fallback** rather than live Nemotron inference.

