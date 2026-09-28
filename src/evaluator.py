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


@dataclass(frozen=True)
class DecisionEvaluation:
    """
    Experiment-side ground-truth
    evaluation of one decision.
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


class DecisionEvaluator:
    """
    Consequence-based evaluator.

    Important:
        This belongs to the EXPERIMENT SIDE.

        A deployed Adequacy Monitor must not
        receive true_state, oracle_action,
        or actual regret.
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

    @staticmethod
    def _manhattan_error(
        true_state,
        belief_state,
    ):

        tx, ty = true_state
        bx, by = belief_state

        return (
            abs(tx - bx)
            + abs(ty - by)
        )

    def _is_walkable(
        self,
        position,
    ):

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
        Simulate one candidate action
        from the TRUE state.

        Does not modify the environment.
        """

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

            return (
                state,
                True,
            )

        return (
            candidate,
            False,
        )

    def _optimal_cost(
        self,
        start,
        goal,
    ):

        try:

            path = self.planner.plan(
                start=start,
                goal=goal,
            )

        except ValueError:

            return None

        if not path:
            return None

        return len(path) - 1

    def _evaluate_action_cost(
        self,
        true_state,
        action,
        goal,
    ):
        """
        Compute:

            one candidate action
            +
            optimal remaining path
        """

        if (
            true_state == goal
            and action is None
        ):

            return (
                0,
                False,
            )

        if action is None:

            return (
                None,
                False,
            )

        (
            next_state,
            collision,
        ) = self._apply_action(
            true_state,
            action,
        )

        remaining_cost = (
            self._optimal_cost(
                next_state,
                goal,
            )
        )

        if remaining_cost is None:

            return (
                None,
                collision,
            )

        return (
            1 + remaining_cost,
            collision,
        )

    def evaluate(
        self,
        true_state,
        belief: BeliefState,
        goal,
    ) -> DecisionEvaluation:

        belief_state = (
            belief.estimated_position
        )

        error = (
            self._manhattan_error(
                true_state,
                belief_state,
            )
        )

        # ----------------------------------------------
        # Ground-truth optimal decision
        # ----------------------------------------------

        oracle_action = (
            self.planner.next_action(
                start=true_state,
                goal=goal,
            )
        )

        optimal_cost = (
            self._optimal_cost(
                true_state,
                goal,
            )
        )

        if optimal_cost is None:

            raise RuntimeError(
                f"True state "
                f"{true_state} "
                f"cannot reach goal "
                f"{goal}."
            )

        # ----------------------------------------------
        # Agent decision based on belief
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
        # Old diagnostic:
        # action divergence
        # ----------------------------------------------

        decision_diverged = (
            not belief_plannable
            or oracle_action
            != agent_action
        )

        # ----------------------------------------------
        # False terminal
        # ----------------------------------------------

        false_terminal = (
            belief_plannable
            and agent_action is None
            and true_state != goal
        )

        # ----------------------------------------------
        # Consequence evaluation
        # ----------------------------------------------

        if not belief_plannable:

            agent_action_cost = None

            decision_regret = None

            normalized_regret = None

            action_collision = False

            decision_adequate = False

        elif false_terminal:

            agent_action_cost = None

            decision_regret = None

            normalized_regret = None

            action_collision = False

            decision_adequate = False

        else:

            (
                agent_action_cost,
                action_collision,
            ) = (
                self._evaluate_action_cost(
                    true_state,
                    agent_action,
                    goal,
                )
            )

            if agent_action_cost is None:

                decision_regret = None

                normalized_regret = None

                decision_adequate = False

            else:

                decision_regret = (
                    agent_action_cost
                    - optimal_cost
                )

                if decision_regret < 0:

                    raise RuntimeError(
                        "Negative decision regret "
                        "detected. Check planner "
                        "and evaluator consistency."
                    )

                normalized_regret = (
                    decision_regret
                    / max(
                        optimal_cost,
                        1,
                    )
                )

                # Collision is treated as
                # a hard inadequacy.
                #
                # Otherwise allow a configurable
                # task tolerance.
                decision_adequate = (
                    not action_collision
                    and decision_regret
                    <= self.adequacy_tolerance
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


def main():

    env = GridWorld()

    planner = AStarPlanner(
        env.grid
    )

    evaluator = DecisionEvaluator(
        planner=planner,
        adequacy_tolerance=0,
    )

    true_state = (4, 3)

    clean_belief = BeliefState(
        estimated_position=true_state
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
        "=== Decision Evaluation V2 ==="
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

    print()

    print(
        f"Oracle action:      "
        f"{result.oracle_action.value}"
    )

    agent_name = (
        result.agent_action.value
        if result.agent_action
        else "NONE"
    )

    print(
        f"Agent action:       "
        f"{agent_name}"
    )

    print(
        f"Decision diverged:  "
        f"{result.decision_diverged}"
    )

    print()

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

    print()

    print(
        f"Decision adequate:  "
        f"{result.decision_adequate}"
    )


if __name__ == "__main__":
    main()