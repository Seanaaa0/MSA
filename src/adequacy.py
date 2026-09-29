from dataclasses import dataclass

from src.planner import AStarPlanner


# ======================================================
# Prediction result
# ======================================================

@dataclass(frozen=True)
class AdequacyPrediction:
    """
    Agent-side adequacy prediction.

    IMPORTANT:
        This object contains only information available
        from the agent's current belief.

        It does NOT use:
            - true state
            - true map
            - oracle action
            - actual regret
            - counterfactual query value
    """

    predicted_adequate: bool

    candidate_action: object

    # --------------------------------------------------
    # Immediate decision sensitivity
    # --------------------------------------------------

    stability: float

    hypotheses_tested: int
    action_changes: int

    # --------------------------------------------------
    # Current optimistic plan
    # --------------------------------------------------

    path_length: int

    path_unknown_count: int
    path_unknown_ratio: float

    # --------------------------------------------------
    # Near-term uncertainty
    # --------------------------------------------------

    near_horizon: int

    near_unknown_count: int
    near_unknown_ratio: float

    # --------------------------------------------------
    # Temporal uncertainty
    # --------------------------------------------------

    high_uncertainty_streak: int

    persistent_uncertainty: bool

    # --------------------------------------------------
    # Explanation
    # --------------------------------------------------

    reason: str


# ======================================================
# Monitor
# ======================================================

class DecisionStabilityMonitor:
    """
    Level-2 Adequacy Monitor V2.

    Main idea:

        Uncertainty alone should NOT automatically
        trigger QueryReality.

    Instead, ask:

        1. Is the immediate action sensitive?

        2. Does the planned path rely heavily
           on unknown information?

        3. Is that uncertainty PERSISTING even after
           the agent receives new passive observations?

    ----------------------------------------------------

    V1 behavior:

        lots of unknown cells
            ->
        Query immediately

    Problem:

        high recall
        but many false-positive queries.

    ----------------------------------------------------

    V2 behavior:

        high uncertainty once
            ->
        wait / continue observing

        high uncertainty persists
            ->
        QueryReality

    This represents:

        "I am uncertain,
         but can normal observation resolve it naturally?"

    ----------------------------------------------------

    This monitor remains heuristic and symbolic.

    It is NOT the final learned MSA monitor.
    """

    def __init__(
        self,
        sensitivity_radius=4,
        stability_threshold=1.0,

        persistent_path_threshold=0.78,
        persistence_steps=2,

        action_path_gate=0.50,

        near_horizon=4,
    ):
        """
        Args:
            sensitivity_radius:

                Radius used for local action
                sensitivity testing.

            stability_threshold:

                Minimum action stability considered safe.

            persistent_path_threshold:

                Path unknown ratio considered
                "high uncertainty".

                Example:
                    0.78 means >= 78% of the currently
                    relevant optimistic route is unknown.

            persistence_steps:

                Number of consecutive decisions that
                high uncertainty must survive before
                it triggers QueryReality.

            action_path_gate:

                Action sensitivity alone is not enough.

                The action-sensitive trigger is enabled
                only when the planned route itself also
                contains meaningful uncertainty.

            near_horizon:

                Number of future steps used for
                near-term diagnostic uncertainty.
        """

        if sensitivity_radius < 0:
            raise ValueError(
                "sensitivity_radius "
                "must be non-negative."
            )

        if not (
            0.0
            <= stability_threshold
            <= 1.0
        ):
            raise ValueError(
                "stability_threshold "
                "must be in [0, 1]."
            )

        if not (
            0.0
            <= persistent_path_threshold
            <= 1.0
        ):
            raise ValueError(
                "persistent_path_threshold "
                "must be in [0, 1]."
            )

        if persistence_steps <= 0:
            raise ValueError(
                "persistence_steps "
                "must be positive."
            )

        if not (
            0.0
            <= action_path_gate
            <= 1.0
        ):
            raise ValueError(
                "action_path_gate "
                "must be in [0, 1]."
            )

        if near_horizon <= 0:
            raise ValueError(
                "near_horizon "
                "must be positive."
            )

        self.sensitivity_radius = (
            sensitivity_radius
        )

        self.stability_threshold = (
            stability_threshold
        )

        self.persistent_path_threshold = (
            persistent_path_threshold
        )

        self.persistence_steps = (
            persistence_steps
        )

        self.action_path_gate = (
            action_path_gate
        )

        self.near_horizon = (
            near_horizon
        )

        # ------------------------------------------------
        # Temporal state
        # ------------------------------------------------

        self._high_uncertainty_streak = 0

    # ==================================================
    # State management
    # ==================================================

    def reset(self):
        """
        Reset temporal monitor memory.

        Call this if the SAME monitor object is reused
        for a completely new episode.

        Current experiments create a new monitor per
        episode, so this is mostly future-proofing.
        """

        self._high_uncertainty_streak = 0

    # ==================================================
    # Planning
    # ==================================================

    def _optimistic_plan(
        self,
        belief,
        goal,
    ):
        """
        Compute the agent's current optimistic plan.

        UNKNOWN cells are treated as free.

        This uses the exact same belief assumption
        as the Level-2 planner.
        """

        if (
            belief.estimated_position
            is None
        ):
            return []

        grid = (
            belief.planning_grid(
                unknown_as_free=True
            )
        )

        planner = AStarPlanner(
            grid
        )

        try:

            path = planner.plan(
                start=
                    belief.estimated_position,

                goal=goal,
            )

        except (
            ValueError,
            RuntimeError,
        ):

            return []

        return path

    # ==================================================
    # Path uncertainty
    # ==================================================

    def _path_uncertainty(
        self,
        belief,
        path,
        goal,
    ):
        """
        Measure uncertainty on the currently
        planned optimistic path.

        Current position is excluded.

        Goal is also excluded because the task
        definition already gives the goal location.
        """

        if not path:

            return (
                0,
                0,
                0.0,
                0,
                0.0,
            )

        future_path = (
            path[1:]
        )

        path_length = len(
            future_path
        )

        # ----------------------------------------------
        # Whole path
        # ----------------------------------------------

        relevant_path = [
            position
            for position in future_path
            if position != goal
        ]

        path_unknown_count = sum(
            belief.is_unknown(
                position
            )
            for position in relevant_path
        )

        if relevant_path:

            path_unknown_ratio = (
                path_unknown_count
                / len(relevant_path)
            )

        else:

            path_unknown_ratio = 0.0

        # ----------------------------------------------
        # Near-term path
        # ----------------------------------------------

        near_path = (
            future_path[
                :self.near_horizon
            ]
        )

        relevant_near = [
            position
            for position in near_path
            if position != goal
        ]

        near_unknown_count = sum(
            belief.is_unknown(
                position
            )
            for position in relevant_near
        )

        if relevant_near:

            near_unknown_ratio = (
                near_unknown_count
                / len(relevant_near)
            )

        else:

            near_unknown_ratio = 0.0

        return (
            path_length,
            path_unknown_count,
            path_unknown_ratio,
            near_unknown_count,
            near_unknown_ratio,
        )

    # ==================================================
    # Action sensitivity
    # ==================================================

    def _action_stability(
        self,
        belief,
        candidate_action,
        goal,
    ):
        """
        Stress-test the immediate action.

        For every nearby UNKNOWN cell:

            hypothesis:
                "what if this cell is actually blocked?"

        Then replan.

        The true map is NEVER used.
        """

        current = (
            belief.estimated_position
        )

        unknowns = (
            belief.unknown_positions(
                center=current,
                radius=
                    self.sensitivity_radius,
            )
        )

        tested = 0
        changed = 0

        for position in unknowns:

            if (
                position == current
                or
                position == goal
            ):
                continue

            grid = (
                belief.planning_grid(
                    unknown_as_free=True
                )
            )

            x, y = position

            grid[y][x] = "#"

            planner = AStarPlanner(
                grid
            )

            tested += 1

            try:

                alternative_action = (
                    planner.next_action(
                        start=current,
                        goal=goal,
                    )
                )

            except (
                ValueError,
                RuntimeError,
            ):

                changed += 1
                continue

            if (
                alternative_action
                != candidate_action
            ):
                changed += 1

        if tested == 0:

            stability = 1.0

        else:

            stability = (
                (tested - changed)
                / tested
            )

        return (
            stability,
            tested,
            changed,
        )

    # ==================================================
    # Temporal uncertainty
    # ==================================================

    def _update_uncertainty_streak(
        self,
        path_unknown_ratio,
    ):
        """
        Track whether uncertainty is naturally
        resolving through passive observation.

        Example:

            step 1:
                path uncertainty = 0.90
                streak = 1

            step 2:
                path uncertainty = 0.65
                streak = 0

        Interpretation:

            passive sensing resolved the uncertainty
            → no active Query needed.

        But:

            step 1 = 0.90
            step 2 = 0.85
            step 3 = 0.82

        means uncertainty persists despite moving
        and observing.

        That is stronger evidence that QueryReality
        may be useful.
        """

        high_uncertainty = (
            path_unknown_ratio
            >= self.persistent_path_threshold
        )

        if high_uncertainty:

            self._high_uncertainty_streak += 1

        else:

            self._high_uncertainty_streak = 0

        persistent = (
            self._high_uncertainty_streak
            >= self.persistence_steps
        )

        return (
            self._high_uncertainty_streak,
            persistent,
        )

    # ==================================================
    # Main assessment
    # ==================================================

    def evaluate(
        self,
        belief,
        candidate_action,
        goal,
    ) -> AdequacyPrediction:
        """
        Predict whether the current information
        is adequate for the current decision.

        Query is triggered by:

        A.
            persistent unsupported planning

        OR

        B.
            strong immediate action sensitivity
            combined with meaningful path uncertainty

        Raw uncertainty by itself is NOT enough.
        """

        # ----------------------------------------------
        # No candidate
        # ----------------------------------------------

        if candidate_action is None:

            self._high_uncertainty_streak = 0

            return AdequacyPrediction(
                predicted_adequate=False,

                candidate_action=None,

                stability=0.0,

                hypotheses_tested=0,
                action_changes=0,

                path_length=0,

                path_unknown_count=0,
                path_unknown_ratio=0.0,

                near_horizon=
                    self.near_horizon,

                near_unknown_count=0,
                near_unknown_ratio=0.0,

                high_uncertainty_streak=0,

                persistent_uncertainty=False,

                reason=
                    "NO_CANDIDATE_ACTION",
            )

        # ----------------------------------------------
        # Current optimistic plan
        # ----------------------------------------------

        path = (
            self._optimistic_plan(
                belief=belief,
                goal=goal,
            )
        )

        if not path:

            self._high_uncertainty_streak = 0

            return AdequacyPrediction(
                predicted_adequate=False,

                candidate_action=
                    candidate_action,

                stability=0.0,

                hypotheses_tested=0,
                action_changes=0,

                path_length=0,

                path_unknown_count=0,
                path_unknown_ratio=0.0,

                near_horizon=
                    self.near_horizon,

                near_unknown_count=0,
                near_unknown_ratio=0.0,

                high_uncertainty_streak=0,

                persistent_uncertainty=False,

                reason=
                    "NO_OPTIMISTIC_PATH",
            )

        # ==============================================
        # Metric 1:
        # immediate action stability
        # ==============================================

        (
            stability,
            tested,
            changed,
        ) = self._action_stability(
            belief=belief,

            candidate_action=
                candidate_action,

            goal=goal,
        )

        # ==============================================
        # Metric 2:
        # path uncertainty
        # ==============================================

        (
            path_length,
            path_unknown_count,
            path_unknown_ratio,
            near_unknown_count,
            near_unknown_ratio,
        ) = self._path_uncertainty(
            belief=belief,
            path=path,
            goal=goal,
        )

        # ==============================================
        # Metric 3:
        # temporal persistence
        # ==============================================

        (
            uncertainty_streak,
            persistent_uncertainty,
        ) = self._update_uncertainty_streak(
            path_unknown_ratio
        )

        # ==============================================
        # Decision logic
        # ==============================================

        reasons = []

        # ----------------------------------------------
        # A. Persistent route uncertainty
        # ----------------------------------------------

        if persistent_uncertainty:

            reasons.append(
                "PERSISTENT_PATH_UNCERTAINTY"
            )

        # ----------------------------------------------
        # B. Immediate action sensitivity
        #
        # Important:
        # action instability alone no longer causes
        # QueryReality.
        #
        # It must occur while the route still contains
        # meaningful unsupported assumptions.
        # ----------------------------------------------

        action_sensitive = (
            stability
            < self.stability_threshold
        )

        meaningful_path_uncertainty = (
            path_unknown_ratio
            >= self.action_path_gate
        )

        if (
            action_sensitive
            and
            meaningful_path_uncertainty
        ):

            reasons.append(
                "ACTION_SENSITIVE"
            )

        # ----------------------------------------------
        # Final decision
        # ----------------------------------------------

        predicted_adequate = (
            len(reasons)
            == 0
        )

        if predicted_adequate:

            if (
                path_unknown_ratio
                >= self.persistent_path_threshold
            ):

                reason = (
                    "WAIT_FOR_PASSIVE_OBSERVATION"
                )

            else:

                reason = (
                    "DECISION_ADEQUATE"
                )

        else:

            reason = (
                "+".join(
                    reasons
                )
            )

        return AdequacyPrediction(
            predicted_adequate=
                predicted_adequate,

            candidate_action=
                candidate_action,

            # ------------------------------------------
            # Action stability
            # ------------------------------------------

            stability=
                stability,

            hypotheses_tested=
                tested,

            action_changes=
                changed,

            # ------------------------------------------
            # Path uncertainty
            # ------------------------------------------

            path_length=
                path_length,

            path_unknown_count=
                path_unknown_count,

            path_unknown_ratio=
                path_unknown_ratio,

            # ------------------------------------------
            # Near-term uncertainty
            # ------------------------------------------

            near_horizon=
                self.near_horizon,

            near_unknown_count=
                near_unknown_count,

            near_unknown_ratio=
                near_unknown_ratio,

            # ------------------------------------------
            # Temporal state
            # ------------------------------------------

            high_uncertainty_streak=
                uncertainty_streak,

            persistent_uncertainty=
                persistent_uncertainty,

            # ------------------------------------------
            # Diagnostic reason
            # ------------------------------------------

            reason=
                reason,
        )