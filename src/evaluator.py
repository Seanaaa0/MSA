from dataclasses import dataclass
from typing import Optional

from src.env import (
    Action,
    GridWorld,
)
from src.planner import AStarPlanner
from src.belief import BeliefState
from src.perturbation import (
    BeliefPerturbation,
    PositionOffset,
)


# ======================================================
# Evaluation result types
# ======================================================

@dataclass(frozen=True)
class DecisionEvaluation:
    """
    Ground-truth evaluation for the old V1/V2
    belief-position experiments.

    Experiment side only.

    The deployed agent / Adequacy Monitor must NOT see:
        - true_state
        - oracle_action
        - actual regret
    """

    true_state: tuple[int, int]
    belief_state: tuple[int, int]

    error_manhattan: int

    oracle_action: Optional[Action]
    agent_action: Optional[Action]

    decision_diverged: bool
    belief_plannable: bool

    optimal_cost: Optional[int]
    agent_action_cost: Optional[int]

    decision_regret: Optional[int]
    normalized_regret: Optional[float]

    action_collision: bool
    false_terminal: bool

    decision_adequate: bool


@dataclass(frozen=True)
class ActionEvaluation:
    """
    Ground-truth consequence evaluation for an
    externally supplied candidate action.

    This is used by Level 2.

    The action may come from:
        - a partial-observation planner
        - an Adequacy pipeline
        - a future learned policy

    Experiment side only.
    """

    true_state: tuple[int, int]

    oracle_action: Optional[Action]
    agent_action: Optional[Action]

    decision_diverged: bool

    optimal_cost: Optional[int]
    agent_action_cost: Optional[int]

    decision_regret: Optional[int]
    normalized_regret: Optional[float]

    action_collision: bool

    decision_adequate: bool


# ======================================================
# Evaluator
# ======================================================

class DecisionEvaluator:
    """
    Consequence-based ground-truth evaluator.

    IMPORTANT:
        This belongs to the EXPERIMENT SIDE.

    It is allowed to know:
        - true state
        - true map
        - oracle decision
        - actual task consequence

    A deployed Adequacy Monitor must NOT receive
    those values.

    Two interfaces are provided:

    1. evaluate(...)

        Legacy V1/V2 interface.

        true_state
        +
        BeliefState(estimated_position)
        ↓
        evaluator generates the belief-based action
        and evaluates its consequence.

    2. evaluate_action(...)

        Level-2 interface.

        true_state
        +
        externally supplied candidate action
        ↓
        evaluator only evaluates the consequence.

        This is necessary because Level 2 generates
        actions from a partial world belief rather than
        from a single perturbed coordinate.
    """

    def __init__(
        self,
        planner: AStarPlanner,
        adequacy_tolerance: int = 0,
    ):

        if adequacy_tolerance < 0:
            raise ValueError(
                "adequacy_tolerance "
                "must be non-negative."
            )

        self.planner = planner

        self.adequacy_tolerance = (
            adequacy_tolerance
        )

    # ==================================================
    # Basic metrics
    # ==================================================

    @staticmethod
    def _manhattan_error(
        true_state,
        belief_state,
    ):
        """
        Manhattan distance between the true position
        and the estimated position.
        """

        tx, ty = true_state
        bx, by = belief_state

        return (
            abs(tx - bx)
            + abs(ty - by)
        )

    # ==================================================
    # True-world helpers
    # ==================================================

    def _is_walkable(
        self,
        position,
    ):
        """
        Ground-truth walkability check.

        Uses the evaluator's full true map.
        """

        x, y = position

        if (
            x < 0
            or x >= self.planner.width
        ):
            return False

        if (
            y < 0
            or y >= self.planner.height
        ):
            return False

        return (
            self.planner.grid[y][x]
            != "#"
        )

    def _apply_action(
        self,
        state,
        action,
    ):
        """
        Simulate one candidate action from the
        TRUE state.

        This does NOT modify the real GridWorld.

        Returns:
            next_state
            collision
        """

        if action not in (
            self.planner.ACTION_DELTAS
        ):
            raise ValueError(
                f"Unknown action: {action}"
            )

        dx, dy = (
            self.planner
            .ACTION_DELTAS[action]
        )

        candidate = (
            state[0] + dx,
            state[1] + dy,
        )

        if not self._is_walkable(
            candidate
        ):
            # GridWorld collision behavior:
            # stay in the same state.
            return (
                state,
                True,
            )

        return (
            candidate,
            False,
        )

    # ==================================================
    # Cost helpers
    # ==================================================

    def _optimal_cost(
        self,
        start,
        goal,
    ):
        """
        Ground-truth shortest-path cost.

        Unit-cost grid:
            cost = number of actions
        """

        try:

            path = self.planner.plan(
                start=start,
                goal=goal,
            )

        except ValueError:

            return None

        if not path:
            return None

        return (
            len(path)
            - 1
        )

    def _evaluate_action_cost(
        self,
        true_state,
        action,
        goal,
    ):
        """
        Evaluate one candidate action from reality.

        Computes:

            candidate action
                +
            optimal remaining path

        Therefore:

            agent_action_cost
                =
            1 + V*(next_state)

        Returns:
            (
                agent_action_cost,
                collision
            )
        """

        # Already at goal.
        if (
            true_state == goal
            and action is None
        ):
            return (
                0,
                False,
            )

        # No action while reality still requires movement.
        if action is None:
            return (
                None,
                False,
            )

        (
            next_state,
            collision,
        ) = self._apply_action(
            state=true_state,
            action=action,
        )

        remaining_cost = (
            self._optimal_cost(
                start=next_state,
                goal=goal,
            )
        )

        if remaining_cost is None:
            return (
                None,
                collision,
            )

        agent_action_cost = (
            1
            + remaining_cost
        )

        return (
            agent_action_cost,
            collision,
        )

    # ==================================================
    # Shared consequence calculation
    # ==================================================

    def _calculate_consequence(
        self,
        true_state,
        agent_action,
        goal,
    ):
        """
        Shared consequence computation.

        Returns:
            optimal_cost
            agent_action_cost
            decision_regret
            normalized_regret
            action_collision
            decision_adequate
        """

        optimal_cost = (
            self._optimal_cost(
                start=true_state,
                goal=goal,
            )
        )

        if optimal_cost is None:
            raise RuntimeError(
                f"True state {true_state} "
                f"cannot reach goal {goal}."
            )

        (
            agent_action_cost,
            action_collision,
        ) = self._evaluate_action_cost(
            true_state=true_state,
            action=agent_action,
            goal=goal,
        )

        # No meaningful candidate consequence.
        if agent_action_cost is None:

            decision_regret = None
            normalized_regret = None
            decision_adequate = False

            return (
                optimal_cost,
                agent_action_cost,
                decision_regret,
                normalized_regret,
                action_collision,
                decision_adequate,
            )

        decision_regret = (
            agent_action_cost
            - optimal_cost
        )

        # With a correct optimal planner,
        # negative regret should be impossible.
        if decision_regret < 0:
            raise RuntimeError(
                "Negative decision regret detected. "
                "Check planner/evaluator consistency."
            )

        normalized_regret = (
            decision_regret
            / max(
                optimal_cost,
                1,
            )
        )

        # Hard failure:
        # collision is never adequate.
        #
        # Soft performance:
        # otherwise allow task tolerance.
        decision_adequate = (
            not action_collision
            and decision_regret
            <= self.adequacy_tolerance
        )

        return (
            optimal_cost,
            agent_action_cost,
            decision_regret,
            normalized_regret,
            action_collision,
            decision_adequate,
        )

    # ==================================================
    # V1 / V2 interface
    # ==================================================

    def evaluate(
        self,
        true_state,
        belief: BeliefState,
        goal,
    ) -> DecisionEvaluation:
        """
        Legacy V1/V2 evaluator.

        Agent decision is generated from:

            belief.estimated_position

        This remains for:
            experiments/run_sweep.py
            controlled perturbation experiments
        """

        belief_state = (
            belief.estimated_position
        )

        # ----------------------------------------------
        # Actual localization error
        #
        # Experiment side ONLY.
        # ----------------------------------------------

        error = (
            self._manhattan_error(
                true_state=true_state,
                belief_state=belief_state,
            )
        )

        # ----------------------------------------------
        # Oracle action
        # ----------------------------------------------

        oracle_action = (
            self.planner.next_action(
                start=true_state,
                goal=goal,
            )
        )

        # ----------------------------------------------
        # Agent action from belief
        # ----------------------------------------------

        belief_plannable = True

        try:

            agent_action = (
                self.planner.next_action(
                    start=belief_state,
                    goal=goal,
                )
            )

        except (
            ValueError,
            RuntimeError,
        ):

            agent_action = None

            belief_plannable = False

        # ----------------------------------------------
        # Old V0 diagnostic
        # ----------------------------------------------

        decision_diverged = (
            not belief_plannable
            or oracle_action
            != agent_action
        )

        # ----------------------------------------------
        # False terminal
        #
        # Agent believes it is at the goal,
        # while reality says otherwise.
        # ----------------------------------------------

        false_terminal = (
            belief_plannable
            and agent_action is None
            and true_state != goal
        )

        # ----------------------------------------------
        # Invalid belief
        # ----------------------------------------------

        if not belief_plannable:

            optimal_cost = (
                self._optimal_cost(
                    start=true_state,
                    goal=goal,
                )
            )

            if optimal_cost is None:
                raise RuntimeError(
                    f"True state {true_state} "
                    f"cannot reach goal {goal}."
                )

            agent_action_cost = None
            decision_regret = None
            normalized_regret = None

            action_collision = False

            decision_adequate = False

        # ----------------------------------------------
        # False terminal
        # ----------------------------------------------

        elif false_terminal:

            optimal_cost = (
                self._optimal_cost(
                    start=true_state,
                    goal=goal,
                )
            )

            if optimal_cost is None:
                raise RuntimeError(
                    f"True state {true_state} "
                    f"cannot reach goal {goal}."
                )

            agent_action_cost = None
            decision_regret = None
            normalized_regret = None

            action_collision = False

            decision_adequate = False

        # ----------------------------------------------
        # Normal consequence evaluation
        # ----------------------------------------------

        else:

            (
                optimal_cost,
                agent_action_cost,
                decision_regret,
                normalized_regret,
                action_collision,
                decision_adequate,
            ) = self._calculate_consequence(
                true_state=true_state,
                agent_action=agent_action,
                goal=goal,
            )

        return DecisionEvaluation(
            true_state=true_state,

            belief_state=belief_state,

            error_manhattan=error,

            oracle_action=oracle_action,

            agent_action=agent_action,

            decision_diverged=
                decision_diverged,

            belief_plannable=
                belief_plannable,

            optimal_cost=
                optimal_cost,

            agent_action_cost=
                agent_action_cost,

            decision_regret=
                decision_regret,

            normalized_regret=
                normalized_regret,

            action_collision=
                action_collision,

            false_terminal=
                false_terminal,

            decision_adequate=
                decision_adequate,
        )

    # ==================================================
    # Level-2 interface
    # ==================================================

    def evaluate_action(
        self,
        true_state,
        agent_action: Optional[Action],
        goal,
    ) -> ActionEvaluation:
        """
        Evaluate a candidate action generated elsewhere.

        This is the Level-2 interface.

        Example:

            partial observation
                  ↓
            OccupancyBelief
                  ↓
            partial-map planner
                  ↓
            candidate_action
                  ↓
            evaluate_action(...)

        IMPORTANT:

            evaluate_action() is EXPERIMENT SIDE.

            The candidate action is evaluated using
            ground-truth reality only to produce labels
            and metrics.

            The Adequacy Monitor itself must never
            receive these ground-truth values.
        """

        # ----------------------------------------------
        # Oracle action
        # ----------------------------------------------

        oracle_action = (
            self.planner.next_action(
                start=true_state,
                goal=goal,
            )
        )

        # ----------------------------------------------
        # Diagnostic only
        #
        # Different does NOT automatically mean bad.
        # ----------------------------------------------

        decision_diverged = (
            oracle_action
            != agent_action
        )

        # ----------------------------------------------
        # Actual consequence
        # ----------------------------------------------

        (
            optimal_cost,
            agent_action_cost,
            decision_regret,
            normalized_regret,
            action_collision,
            decision_adequate,
        ) = self._calculate_consequence(
            true_state=true_state,
            agent_action=agent_action,
            goal=goal,
        )

        return ActionEvaluation(
            true_state=true_state,

            oracle_action=
                oracle_action,

            agent_action=
                agent_action,

            decision_diverged=
                decision_diverged,

            optimal_cost=
                optimal_cost,

            agent_action_cost=
                agent_action_cost,

            decision_regret=
                decision_regret,

            normalized_regret=
                normalized_regret,

            action_collision=
                action_collision,

            decision_adequate=
                decision_adequate,
        )


# ======================================================
# Manual tests
# ======================================================

def main():

    env = GridWorld()

    planner = AStarPlanner(
        env.grid
    )

    evaluator = DecisionEvaluator(
        planner=planner,
        adequacy_tolerance=0,
    )

    # ==================================================
    # Test A:
    # old V2 belief-position interface
    # ==================================================

    true_state = (
        4,
        3,
    )

    clean_belief = (
        BeliefState(
            estimated_position=
                true_state
        )
    )

    perturbation = (
        BeliefPerturbation(
            PositionOffset(
                dx=2,
                dy=0,
            )
        )
    )

    perturbed_belief = (
        perturbation.apply(
            clean_belief
        )
    )

    result = evaluator.evaluate(
        true_state=true_state,
        belief=perturbed_belief,
        goal=env.goal_position,
    )

    print(
        "=== Evaluator V2 "
        "Belief Test ==="
    )

    print(
        f"True state:         "
        f"{result.true_state}"
    )

    print(
        f"Belief state:       "
        f"{result.belief_state}"
    )

    print(
        f"Error magnitude:    "
        f"{result.error_manhattan}"
    )

    oracle_name = (
        result.oracle_action.value
        if result.oracle_action
        else "NONE"
    )

    agent_name = (
        result.agent_action.value
        if result.agent_action
        else "NONE"
    )

    print(
        f"Oracle action:      "
        f"{oracle_name}"
    )

    print(
        f"Agent action:       "
        f"{agent_name}"
    )

    print(
        f"Decision diverged:  "
        f"{result.decision_diverged}"
    )

    print(
        f"Optimal cost:       "
        f"{result.optimal_cost}"
    )

    print(
        f"Agent action cost:  "
        f"{result.agent_action_cost}"
    )

    print(
        f"Decision regret:    "
        f"{result.decision_regret}"
    )

    print(
        f"Normalized regret:  "
        f"{result.normalized_regret}"
    )

    print(
        f"Collision:          "
        f"{result.action_collision}"
    )

    print(
        f"False terminal:     "
        f"{result.false_terminal}"
    )

    print(
        f"Decision adequate:  "
        f"{result.decision_adequate}"
    )

    # ==================================================
    # Test B:
    # new Level-2 externally supplied action
    # ==================================================

    print()

    print(
        "=== Evaluator Level-2 "
        "Action Test ==="
    )

    candidate_action = (
        result.agent_action
    )

    action_result = (
        evaluator.evaluate_action(
            true_state=true_state,
            agent_action=
                candidate_action,
            goal=env.goal_position,
        )
    )

    oracle_name = (
        action_result
        .oracle_action.value
        if action_result
        .oracle_action
        else "NONE"
    )

    candidate_name = (
        action_result
        .agent_action.value
        if action_result
        .agent_action
        else "NONE"
    )

    print(
        f"True state:         "
        f"{action_result.true_state}"
    )

    print(
        f"Oracle action:      "
        f"{oracle_name}"
    )

    print(
        f"Candidate action:   "
        f"{candidate_name}"
    )

    print(
        f"Decision diverged:  "
        f"{action_result.decision_diverged}"
    )

    print(
        f"Optimal cost:       "
        f"{action_result.optimal_cost}"
    )

    print(
        f"Action cost:        "
        f"{action_result.agent_action_cost}"
    )

    print(
        f"Decision regret:    "
        f"{action_result.decision_regret}"
    )

    print(
        f"Normalized regret:  "
        f"{action_result.normalized_regret}"
    )

    print(
        f"Collision:          "
        f"{action_result.action_collision}"
    )

    print(
        f"Decision adequate:  "
        f"{action_result.decision_adequate}"
    )


if __name__ == "__main__":
    main()