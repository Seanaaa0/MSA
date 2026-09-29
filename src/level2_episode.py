import copy

from src.env import GridWorld
from src.planner import AStarPlanner
from src.belief import OccupancyBelief
from src.evaluator import DecisionEvaluator


class Level2EpisodeRunner:
    """
    Level-2 partial-observation MSA episode.

    Real execution policies:
        - always_trust
        - always_query
        - monitor

    In addition, the EXPERIMENT SIDE performs
    counterfactual rollouts at every decision point:

        TRUST NOW
            vs
        QUERY NOW

    This lets us evaluate whether querying at the
    current moment would actually improve future
    task consequence.

    IMPORTANT:
        Counterfactual results are NEVER given to
        the Adequacy Monitor.
    """

    VALID_POLICIES = {
        "always_trust",
        "always_query",
        "monitor",
    }

    def __init__(
        self,
        env,
        observer,
        monitor,
        query_reality,
        policy="monitor",
        max_steps=300,
        counterfactual_max_steps=300,
    ):

        if policy not in self.VALID_POLICIES:
            raise ValueError(
                f"Unknown policy: {policy}"
            )

        self.env = env
        self.observer = observer
        self.monitor = monitor

        self.query_reality = (
            query_reality
        )

        self.policy = policy

        self.max_steps = max_steps

        self.counterfactual_max_steps = (
            counterfactual_max_steps
        )

    # ==================================================
    # Agent-side planning
    # ==================================================

    def _plan(
        self,
        belief,
    ):
        """
        Plan using the current partial world belief.

        UNKNOWN cells are treated optimistically
        as free.
        """

        planner = AStarPlanner(
            belief.planning_grid(
                unknown_as_free=True
            )
        )

        return planner.next_action(
            start=
                belief.estimated_position,

            goal=
                belief.goal_position,
        )

    # ==================================================
    # Counterfactual utilities
    # ==================================================

    def _clone_env_at(
        self,
        position,
    ):
        """
        Create an isolated copy of the true environment.

        Used ONLY for experiment-side counterfactual
        simulation.

        Does not alter the real episode.
        """

        grid_map = [
            "".join(row)
            for row in self.env.grid
        ]

        sim_env = GridWorld(
            grid_map=grid_map
        )

        sim_env.agent_position = (
            position
        )

        sim_env.steps = 0

        sim_env.done = (
            position
            == sim_env.goal_position
        )

        return sim_env

    def _rollout_without_future_queries(
        self,
        start_position,
        start_belief,
        first_action,
        query_now=False,
    ):
        """
        Counterfactual rollout.

        Branch A:
            query_now=False

            Execute the existing candidate action,
            then continue using only normal observations.

        Branch B:
            query_now=True

            QueryReality ONCE now,
            replan,
            then continue using only normal observations.

        No additional active queries are allowed after
        this initial choice.

        This isolates the value of querying NOW.
        """

        sim_env = (
            self._clone_env_at(
                start_position
            )
        )

        sim_belief = (
            copy.deepcopy(
                start_belief
            )
        )

        # ----------------------------------------------
        # Optional single QueryReality
        # ----------------------------------------------

        if query_now:

            self.query_reality.execute(
                env=sim_env,
                belief=sim_belief,
            )

            try:

                action = (
                    self._plan(
                        sim_belief
                    )
                )

            except (
                ValueError,
                RuntimeError,
            ):

                action = None

        else:

            action = (
                first_action
            )

        steps = 0
        collisions = 0

        # ----------------------------------------------
        # Roll forward to task completion
        # ----------------------------------------------

        while (
            not sim_env.done
            and
            steps
            < self.counterfactual_max_steps
        ):

            if action is None:
                break

            result = (
                sim_env.step(
                    action
                )
            )

            steps += 1

            if result["collision"]:
                collisions += 1

            if sim_env.done:
                break

            # ------------------------------------------
            # Free/default passive observation
            # ------------------------------------------

            observation = (
                self.observer.observe(
                    sim_env
                )
            )

            sim_belief.update(
                observation
            )

            # ------------------------------------------
            # Replan using new partial knowledge
            # ------------------------------------------

            try:

                action = (
                    self._plan(
                        sim_belief
                    )
                )

            except (
                ValueError,
                RuntimeError,
            ):

                action = None

        success = (
            sim_env.agent_position
            == sim_env.goal_position
        )

        timeout = (
            not success
            and
            steps
            >= self.counterfactual_max_steps
        )

        return {
            "success":
                success,

            "steps":
                steps,

            "collisions":
                collisions,

            "timeout":
                timeout,
        }

    @staticmethod
    def _compare_counterfactuals(
        trust_result,
        query_result,
    ):
        """
        Determine whether QueryReality NOW is beneficial.

        Comparison order:

        1. Task success
        2. Fewer collisions
        3. Fewer task steps

        Query cost is intentionally NOT mixed into the
        task metric here.

        We first ask:

            "Did QueryReality improve the task?"

        Query cost is tracked separately.
        """

        # ----------------------------------------------
        # Success dominates everything
        # ----------------------------------------------

        if (
            trust_result["success"]
            != query_result["success"]
        ):

            query_beneficial = (
                query_result["success"]
                and
                not trust_result["success"]
            )

            query_harmful = (
                trust_result["success"]
                and
                not query_result["success"]
            )

        # ----------------------------------------------
        # Then safety / collisions
        # ----------------------------------------------

        elif (
            trust_result["collisions"]
            != query_result["collisions"]
        ):

            query_beneficial = (
                query_result["collisions"]
                <
                trust_result["collisions"]
            )

            query_harmful = (
                query_result["collisions"]
                >
                trust_result["collisions"]
            )

        # ----------------------------------------------
        # Then path efficiency
        # ----------------------------------------------

        else:

            query_beneficial = (
                query_result["steps"]
                <
                trust_result["steps"]
            )

            query_harmful = (
                query_result["steps"]
                >
                trust_result["steps"]
            )

        # ----------------------------------------------
        # Step gain is meaningful when both complete.
        # ----------------------------------------------

        if (
            trust_result["success"]
            and
            query_result["success"]
        ):

            step_gain = (
                trust_result["steps"]
                -
                query_result["steps"]
            )

        else:

            step_gain = None

        return {
            "query_beneficial":
                query_beneficial,

            "query_harmful":
                query_harmful,

            "step_gain":
                step_gain,
        }

    def _evaluate_query_opportunity(
        self,
        true_position,
        belief,
        candidate_action,
    ):
        """
        Experiment-side label:

            Would querying NOW improve future
            task consequence?

        The agent never sees this answer.
        """

        trust_result = (
            self._rollout_without_future_queries(
                start_position=
                    true_position,

                start_belief=
                    belief,

                first_action=
                    candidate_action,

                query_now=False,
            )
        )

        query_result = (
            self._rollout_without_future_queries(
                start_position=
                    true_position,

                start_belief=
                    belief,

                first_action=
                    candidate_action,

                query_now=True,
            )
        )

        comparison = (
            self._compare_counterfactuals(
                trust_result=
                    trust_result,

                query_result=
                    query_result,
            )
        )

        return {
            "trust":
                trust_result,

            "query":
                query_result,

            **comparison,
        }

    # ==================================================
    # Episode
    # ==================================================

    def run(
        self,
        verbose=False,
    ):

        self.env.reset()

        # ----------------------------------------------
        # Agent belief
        # ----------------------------------------------

        belief = OccupancyBelief(
            width=
                self.env.width,

            height=
                self.env.height,

            goal_position=
                self.env.goal_position,
        )

        # ----------------------------------------------
        # Experiment-side oracle
        # ----------------------------------------------

        true_planner = (
            AStarPlanner(
                self.env.grid
            )
        )

        evaluator = (
            DecisionEvaluator(
                planner=
                    true_planner,

                adequacy_tolerance=0,
            )
        )

        # ----------------------------------------------
        # Episode metrics
        # ----------------------------------------------

        queries = 0

        query_cost = 0.0

        collisions = 0

        decisions = 0

        # One-step ground truth.
        inadequate_candidates = 0

        monitor_accepts = 0
        monitor_rejects = 0

        # ----------------------------------------------
        # Horizon / query-value ground truth
        # ----------------------------------------------

        beneficial_query_opportunities = 0

        harmful_query_opportunities = 0

        neutral_query_opportunities = 0

        positive_step_gain_total = 0

        # ----------------------------------------------
        # Confusion matrix
        #
        # Positive:
        #   Query would be beneficial.
        #
        # Prediction positive:
        #   policy decides to Query.
        # ----------------------------------------------

        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        # ----------------------------------------------
        # Initial normal observation
        # ----------------------------------------------

        belief.update(
            self.observer.observe(
                self.env
            )
        )

        # ==============================================
        # Main loop
        # ==============================================

        while (
            not self.env.done
            and
            self.env.steps
            < self.max_steps
        ):

            # ------------------------------------------
            # Passive/default sensing
            # ------------------------------------------

            belief.update(
                self.observer.observe(
                    self.env
                )
            )

            # ------------------------------------------
            # Candidate decision
            # ------------------------------------------

            try:

                candidate_action = (
                    self._plan(
                        belief
                    )
                )

            except (
                ValueError,
                RuntimeError,
            ):

                candidate_action = None

            if candidate_action is None:
                break

            decisions += 1

            true_position = (
                self.env.agent_position
            )

            # ==========================================
            # Immediate one-step ground truth
            # ==========================================

            immediate_eval = (
                evaluator.evaluate_action(
                    true_state=
                        true_position,

                    agent_action=
                        candidate_action,

                    goal=
                        self.env.goal_position,
                )
            )

            if (
                not immediate_eval
                .decision_adequate
            ):
                inadequate_candidates += 1

            # ==========================================
            # FUTURE query-value ground truth
            #
            # Experiment side ONLY.
            # ==========================================

            query_value = (
                self._evaluate_query_opportunity(
                    true_position=
                        true_position,

                    belief=
                        belief,

                    candidate_action=
                        candidate_action,
                )
            )

            query_beneficial = (
                query_value[
                    "query_beneficial"
                ]
            )

            query_harmful = (
                query_value[
                    "query_harmful"
                ]
            )

            if query_beneficial:

                beneficial_query_opportunities += 1

            elif query_harmful:

                harmful_query_opportunities += 1

            else:

                neutral_query_opportunities += 1

            step_gain = (
                query_value[
                    "step_gain"
                ]
            )

            if (
                step_gain is not None
                and
                step_gain > 0
            ):

                positive_step_gain_total += (
                    step_gain
                )

            # ==========================================
            # Agent-side adequacy prediction
            # ==========================================

            prediction = (
                self.monitor.evaluate(
                    belief=belief,

                    candidate_action=
                        candidate_action,

                    goal=
                        self.env.goal_position,
                )
            )

            # ------------------------------------------
            # Real policy decision
            # ------------------------------------------

            if (
                self.policy
                == "always_query"
            ):

                should_query = True

            elif (
                self.policy
                == "always_trust"
            ):

                should_query = False

            else:

                should_query = (
                    not prediction
                    .predicted_adequate
                )

            # ==========================================
            # Query-value confusion matrix
            # ==========================================

            if should_query:

                if query_beneficial:

                    true_positive += 1

                else:

                    false_positive += 1

            else:

                if query_beneficial:

                    false_negative += 1

                else:

                    true_negative += 1

            # ==========================================
            # Actual intervention
            # ==========================================

            if should_query:

                monitor_rejects += 1

                query_result = (
                    self.query_reality.execute(
                        env=self.env,
                        belief=belief,
                    )
                )

                queries += 1

                query_cost += (
                    query_result[
                        "cost"
                    ]
                )

                # Replan after querying.
                try:

                    candidate_action = (
                        self._plan(
                            belief
                        )
                    )

                except (
                    ValueError,
                    RuntimeError,
                ):

                    candidate_action = None

                if candidate_action is None:
                    break

            else:

                monitor_accepts += 1

            # ==========================================
            # Execute in reality
            # ==========================================

            result = (
                self.env.step(
                    candidate_action
                )
            )

            if result["collision"]:
                collisions += 1

            # ------------------------------------------
            # Debug output
            # ------------------------------------------

            if verbose:

                print(
                    f"step={result['steps']} "
                    f"action="
                    f"{candidate_action.value} "
                    f"query={should_query} "
                    f"beneficial="
                    f"{query_beneficial} "
                    f"step_gain="
                    f"{step_gain} "
                    f"stability="
                    f"{prediction.stability:.2f} "
                    f"position="
                    f"{result['position']}"
                )

        # ==============================================
        # Final result
        # ==============================================

        success = (
            self.env.agent_position
            == self.env.goal_position
        )

        timeout = (
            not success
            and
            self.env.steps
            >= self.max_steps
        )

        # ----------------------------------------------
        # Query decision metrics
        # ----------------------------------------------

        predicted_queries = (
            true_positive
            + false_positive
        )

        actual_beneficial = (
            true_positive
            + false_negative
        )

        precision = (
            true_positive
            / predicted_queries
            if predicted_queries
            else 1.0
        )

        recall = (
            true_positive
            / actual_beneficial
            if actual_beneficial
            else 1.0
        )

        return {
            "policy":
                self.policy,

            # ------------------------------------------
            # Actual episode result
            # ------------------------------------------

            "success":
                success,

            "steps":
                self.env.steps,

            "queries":
                queries,

            "query_cost":
                query_cost,

            "collisions":
                collisions,

            "timeout":
                timeout,

            "decisions":
                decisions,

            # ------------------------------------------
            # Old one-step label
            # ------------------------------------------

            "inadequate_candidates":
                inadequate_candidates,

            # ------------------------------------------
            # Policy behavior
            # ------------------------------------------

            "monitor_accepts":
                monitor_accepts,

            "monitor_rejects":
                monitor_rejects,

            # ------------------------------------------
            # Counterfactual query-value labels
            # ------------------------------------------

            "beneficial_query_opportunities":
                beneficial_query_opportunities,

            "neutral_query_opportunities":
                neutral_query_opportunities,

            "harmful_query_opportunities":
                harmful_query_opportunities,

            "positive_step_gain_total":
                positive_step_gain_total,

            # ------------------------------------------
            # Confusion matrix
            # ------------------------------------------

            "true_positive":
                true_positive,

            "false_positive":
                false_positive,

            "true_negative":
                true_negative,

            "false_negative":
                false_negative,

            "query_precision":
                precision,

            "query_recall":
                recall,
        }


# ======================================================
# Manual run
# ======================================================

def main():

    from src.observation import (
        LocalObserver,
    )

    from src.adequacy import (
        DecisionStabilityMonitor,
    )

    from src.query import (
        QueryReality,
    )

    env = GridWorld(
        map_name="detour"
    )

    observer = LocalObserver(
        normal_radius=1,
        query_radius=4,
    )

    monitor = (
        DecisionStabilityMonitor(
            sensitivity_radius=4,
            stability_threshold=1.0,
        )
    )

    query = QueryReality(
        observer=observer,
        cost=1.0,
    )

    runner = Level2EpisodeRunner(
        env=env,

        observer=observer,

        monitor=monitor,

        query_reality=query,

        policy="monitor",

        max_steps=300,

        counterfactual_max_steps=300,
    )

    result = runner.run(
        verbose=True
    )

    print()

    print(
        "=== Level 2 Result ==="
    )

    for key, value in (
        result.items()
    ):

        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()