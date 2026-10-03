import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SMBGymEnv(gym.Env):
    """
    Base governance-aware sequential financial decision environment.

    Observation:
        24-dimensional normalized financial state vector.

    Action:
        Discrete action index from the Chapter 6 action registry.

    Reward:
        Composite reward balancing profitability, liquidity,
        growth, distress risk, leverage risk, and governance penalties.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data,
        state_vars,
        action_registry,
        max_steps=20,
        stochastic_noise=0.03,
        seed=42,
        render_mode=None,
        scaler_means=None,
        scaler_stds=None,
    ):
        super().__init__()

        self.data = data.copy().reset_index(drop=True)
        self.state_vars = list(state_vars)
        self.action_registry = action_registry.copy().reset_index(drop=True)

        self.max_steps = max_steps
        self.stochastic_noise = stochastic_noise
        self.seed_value = seed
        self.render_mode = render_mode

        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(
            len(self.action_registry)
        )

        self.observation_space = spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(len(self.state_vars),),
            dtype=np.float32,
        )

        if (
            scaler_means is not None
            and scaler_stds is not None
        ):
            self.means = {
                var: float(
                    scaler_means[var]
                )
                for var in self.state_vars
            }

            self.stds = {
                var: float(
                    max(
                        scaler_stds[var],
                        1e-6,
                    )
                )
                for var in self.state_vars
            }

        else:
            self._compute_scalers()

        self.current_raw_state = None
        self.current_obs = None
        self.step_count = 0
        self.initial_distress = None
        self.current_info = {}

    def _compute_scalers(self):
        self.means = {}
        self.stds = {}

        for var in self.state_vars:
            values = (
                self.data[var]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.data[var].median())
            )

            self.means[var] = float(
                values.mean()
            )

            self.stds[var] = float(
                max(
                    values.std(),
                    1e-6,
                )
            )

    def _normalize_state(self, raw_state):
        obs = []

        for var in self.state_vars:
            val = raw_state.get(
                var,
                self.means[var],
            )

            norm_val = (
                val - self.means[var]
            ) / self.stds[var]

            obs.append(
                norm_val
            )

        obs = np.array(
            obs,
            dtype=np.float32,
        )

        return np.clip(
            obs,
            -3.0,
            3.0,
        )

    def _sample_initial_state(self):
        idx = int(
            self.rng.integers(
                0,
                len(self.data),
            )
        )

        row = self.data.iloc[
            idx
        ].to_dict()

        raw_state = {
            var: float(
                row.get(
                    var,
                    self.means[var],
                )
            )
            for var in self.state_vars
        }

        info = {
            "row_index": idx,
            "cik": row.get(
                "cik"
            ),
            "company_name": row.get(
                "company_name"
            ),
            "economic_regime": row.get(
                "economic_regime"
            ),
            "economic_regime_code": row.get(
                "economic_regime_code"
            ),
        }

        return (
            raw_state,
            info,
        )

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        if seed is not None:
            self.rng = np.random.default_rng(
                seed
            )

        self.current_raw_state, self.current_info = (
            self._sample_initial_state()
        )

        self.step_count = 0

        self.initial_distress = (
            self.current_raw_state.get(
                "distress_prob_h1",
                0.5,
            )
        )

        self.current_obs = (
            self._normalize_state(
                self.current_raw_state
            )
        )

        return (
            self.current_obs,
            self.current_info,
        )

    def _compute_reward(
        self,
        state,
        action_row,
    ):
        pm = state.get(
            "profit_margin",
            0.0,
        )

        cr = state.get(
            "cash_ratio",
            0.0,
        )

        growth = state.get(
            "revenue_growth_qoq",
            0.0,
        )

        debt = state.get(
            "debt_ratio",
            0.0,
        )

        distress = state.get(
            "distress_prob_h1",
            0.5,
        )

        reward = 0.0

        reward += 1.00 * pm
        reward += 0.50 * min(
            cr / 2.0,
            1.0,
        )
        reward += 0.30 * np.tanh(
            growth
        )

        reward -= 0.75 * distress
        reward -= 0.20 * debt

        reward -= float(
            action_row.get(
                "governance_penalty",
                0.0,
            )
        )

        if cr < 0.10:
            reward -= 5.0

        if debt > 1.50:
            reward -= 2.0

        if (
            growth > 0.15
            and cr < 0.30
        ):
            reward -= 1.5

        if (
            action_row["cost_control"] == 1
            and pm > 0.10
        ):
            reward -= 0.5

        if (
            distress < 0.30
            and self.initial_distress is not None
            and self.initial_distress > 0.50
        ):
            reward += 1.0

        return float(
            reward
        )

    def step(
        self,
        action,
    ):
        action = int(
            action
        )

        action_row = (
            self.action_registry.iloc[
                action
            ]
        )

        next_state = (
            self._apply_action_transition(
                action_row
            )
        )

        reward = (
            self._compute_reward(
                next_state,
                action_row,
            )
        )

        self.current_raw_state = (
            next_state
        )

        self.current_obs = (
            self._normalize_state(
                next_state
            )
        )

        self.step_count += 1

        cash_ratio = next_state.get(
            "cash_ratio",
            1.0,
        )

        debt_ratio = next_state.get(
            "debt_ratio",
            0.0,
        )

        distress_prob = next_state.get(
            "distress_prob_h1",
            0.5,
        )

        bankrupt = bool(
            cash_ratio < 0.05
            and debt_ratio > 2.0
        )

        recovered = bool(
            distress_prob < 0.30
            and self.initial_distress is not None
            and self.initial_distress > 0.50
        )

        terminated = bankrupt

        truncated = (
            self.step_count
            >= self.max_steps
        )

        if bankrupt:
            reward -= 20.0

        info = {
            "step":
                self.step_count,

            "action_id":
                action,

            "action_description":
                action_row[
                    "action_description"
                ],

            "governance_profile":
                action_row[
                    "governance_profile"
                ],

            "economic_regime":
                self.current_info.get(
                    "economic_regime"
                ),

            "profit_margin":
                next_state.get(
                    "profit_margin",
                    np.nan,
                ),

            "cash_ratio":
                cash_ratio,

            "current_ratio":
                next_state.get(
                    "current_ratio",
                    np.nan,
                ),

            "debt_ratio":
                debt_ratio,

            "revenue_growth_qoq":
                next_state.get(
                    "revenue_growth_qoq",
                    np.nan,
                ),

            "distress_prob_h1":
                distress_prob,

            "bankrupt":
                bankrupt,

            "recovered":
                recovered,

            "reward":
                reward,
        }

        return (
            self.current_obs,
            reward,
            terminated,
            truncated,
            info,
        )


class SMBGymRealDistressEnv(
    SMBGymEnv
):
    """
    Final Chapter 6 environment using actual Chapter 3
    distress probabilities.

    Distress evolves on the log-odds scale with persistence,
    anchoring to the initial Chapter 3 prediction,
    financial risk adjustments, macro effects,
    stochastic variation, and bounded quarterly movement.
    """

    @staticmethod
    def _probability_to_logit(
        probability,
        epsilon=1e-6,
    ):
        probability = float(
            np.clip(
                probability,
                epsilon,
                1.0 - epsilon,
            )
        )

        return float(
            np.log(
                probability
                / (
                    1.0
                    - probability
                )
            )
        )

    @staticmethod
    def _logit_to_probability(
        logit_value,
    ):
        logit_value = float(
            np.clip(
                logit_value,
                -20.0,
                20.0,
            )
        )

        return float(
            1.0
            / (
                1.0
                + np.exp(
                    -logit_value
                )
            )
        )

    def reset(
        self,
        seed=None,
        options=None,
    ):
        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        self.baseline_distress_probability = float(
            np.clip(
                self.current_raw_state.get(
                    "distress_prob_h1",
                    0.5,
                ),
                1e-6,
                1.0 - 1e-6,
            )
        )

        self.baseline_distress_logit = (
            self._probability_to_logit(
                self.baseline_distress_probability
            )
        )

        return (
            obs,
            info,
        )

    def _apply_action_transition(
        self,
        action_row,
    ):
        state = (
            self.current_raw_state.copy()
        )

        pm = float(
            state.get(
                "profit_margin",
                0.0,
            )
        )

        cr = float(
            state.get(
                "cash_ratio",
                1.0,
            )
        )

        cur = float(
            state.get(
                "current_ratio",
                1.0,
            )
        )

        debt = float(
            state.get(
                "debt_ratio",
                0.5,
            )
        )

        growth = float(
            state.get(
                "revenue_growth_qoq",
                0.0,
            )
        )

        ocf = float(
            state.get(
                "ocf_margin",
                0.0,
            )
        )

        distress = float(
            state.get(
                "distress_prob_h1",
                0.5,
            )
        )

        cost = float(
            action_row[
                "cost_control"
            ]
        )

        price = float(
            action_row[
                "pricing_strategy"
            ]
        )

        hiring = float(
            action_row[
                "hiring_strategy"
            ]
        )

        debt_action = float(
            action_row[
                "debt_strategy"
            ]
        )

        investment = float(
            action_row[
                "investment_strategy"
            ]
        )

        # ----------------------------------------------------
        # Financial transition effects
        # ----------------------------------------------------

        pm += 0.035 * cost
        cr += 0.120 * cost
        growth -= 0.025 * cost

        pm += 0.025 * price
        growth -= 0.015 * price

        growth += 0.035 * hiring
        pm -= 0.020 * hiring
        cr -= 0.060 * hiring

        cr += 0.200 * debt_action
        cur += 0.120 * debt_action
        debt += 0.150 * debt_action

        growth += 0.040 * investment
        cr -= 0.080 * investment
        pm -= 0.015 * investment

        # ----------------------------------------------------
        # Macroeconomic regime effects
        # ----------------------------------------------------

        regime = int(
            state.get(
                "economic_regime_code",
                0,
            )
        )

        macro_logit_effect = 0.0

        if regime == 4:
            cr -= 0.040
            debt += 0.040
            growth -= 0.015
            macro_logit_effect = 0.025

        elif regime == 3:
            growth -= 0.025
            macro_logit_effect = 0.020

        elif regime == 2:
            debt += 0.015
            growth -= 0.010
            macro_logit_effect = 0.012

        elif regime == 1:
            growth -= 0.020
            macro_logit_effect = 0.010

        # ----------------------------------------------------
        # Stochastic transition effects
        # ----------------------------------------------------

        transition_noise = (
            self.rng.normal(
                0,
                self.stochastic_noise,
                size=6,
            )
        )

        pm += transition_noise[0]
        cr += transition_noise[1]
        cur += transition_noise[2]
        debt += transition_noise[3]
        growth += transition_noise[4]
        ocf += transition_noise[5]

        # ----------------------------------------------------
        # Derived performance measures
        # ----------------------------------------------------

        roa = (
            0.60 * pm
            + 0.20 * growth
            + self.rng.normal(
                0,
                self.stochastic_noise,
            )
        )

        operating_margin = (
            0.85 * pm
            + self.rng.normal(
                0,
                self.stochastic_noise,
            )
        )

        # ----------------------------------------------------
        # Calibrated distress transition
        # ----------------------------------------------------

        current_distress_logit = (
            self._probability_to_logit(
                distress
            )
        )

        baseline_logit = getattr(
            self,
            "baseline_distress_logit",
            current_distress_logit,
        )

        leverage_risk = max(
            debt - 1.0,
            0.0,
        )

        cash_shortfall = max(
            0.50 - cr,
            0.0,
        )

        current_ratio_shortfall = max(
            0.80 - cur,
            0.0,
        )

        margin_stress = min(
            max(
                -pm,
                0.0,
            ),
            2.0,
        )

        negative_ocf = min(
            max(
                -ocf,
                0.0,
            ),
            2.0,
        )

        unsupported_growth = (
            max(
                growth - 0.15,
                0.0,
            )
            if cr < 0.50
            else 0.0
        )

        positive_margin = min(
            max(
                pm,
                0.0,
            ),
            2.0,
        )

        cash_strength = min(
            max(
                cr,
                0.0,
            ) / 2.0,
            1.0,
        )

        current_ratio_strength = min(
            max(
                cur,
                0.0,
            ) / 2.0,
            1.0,
        )

        positive_ocf = min(
            max(
                ocf,
                0.0,
            ),
            2.0,
        )

        risk_logit_effect = (
            0.020 * leverage_risk
            + 0.030 * cash_shortfall
            + 0.025 * current_ratio_shortfall
            + 0.018 * margin_stress
            + 0.015 * negative_ocf
            + 0.012 * unsupported_growth
        )

        health_logit_effect = (
            0.012 * positive_margin
            + 0.008 * cash_strength
            + 0.008 * current_ratio_strength
            + 0.008 * positive_ocf
        )

        distress_noise = (
            self.rng.normal(
                0,
                0.008,
            )
        )

        raw_logit_adjustment = (
            risk_logit_effect
            - health_logit_effect
            + macro_logit_effect
            + distress_noise
        )

        quarterly_logit_adjustment = float(
            np.clip(
                raw_logit_adjustment,
                -0.08,
                0.08,
            )
        )

        next_distress_logit = (
            0.94 * current_distress_logit
            + 0.06 * baseline_logit
            + quarterly_logit_adjustment
        )

        distress = (
            self._logit_to_probability(
                next_distress_logit
            )
        )

        distress = float(
            np.clip(
                distress,
                1e-6,
                1.0 - 1e-6,
            )
        )

        # ----------------------------------------------------
        # RL-safe state updates
        # ----------------------------------------------------

        state["profit_margin"] = float(
            np.clip(
                pm,
                -2,
                2,
            )
        )

        state["cash_ratio"] = float(
            np.clip(
                cr,
                0,
                20,
            )
        )

        state["current_ratio"] = float(
            np.clip(
                cur,
                0,
                20,
            )
        )

        state["debt_ratio"] = float(
            np.clip(
                debt,
                0,
                5,
            )
        )

        state["revenue_growth_qoq"] = float(
            np.clip(
                growth,
                -1,
                5,
            )
        )

        prior_yoy_growth = float(
            state.get(
                "revenue_growth_yoy",
                growth,
            )
        )

        state["revenue_growth_yoy"] = float(
            np.clip(
                0.70 * prior_yoy_growth
                + 0.30 * growth,
                -1,
                5,
            )
        )

        state["return_on_assets"] = float(
            np.clip(
                roa,
                -2,
                2,
            )
        )

        state["operating_margin"] = float(
            np.clip(
                operating_margin,
                -2,
                2,
            )
        )

        state["ocf_margin"] = float(
            np.clip(
                ocf,
                -2,
                2,
            )
        )

        state[
            "distress_prob_h1"
        ] = distress

        state["cash_runway_quarters"] = float(
            np.clip(
                cr * 1.20,
                0,
                20,
            )
        )

        state["revenue_volatility_4q"] = float(
            np.clip(
                abs(
                    growth
                ),
                0,
                5,
            )
        )

        return state