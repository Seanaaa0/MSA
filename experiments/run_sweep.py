import csv
from pathlib import Path

from src.env import GridWorld
from src.maps import available_maps
from src.planner import AStarPlanner
from src.belief import BeliefState
from src.perturbation import (
    BeliefPerturbation,
    PositionOffset,
)
from src.evaluator import DecisionEvaluator


# ======================================================
# Experiment configuration
# ======================================================

ERROR_MAGNITUDES = [
    0,
    1,
    2,
    3,
    4,
]

MAP_NAMES = list(
    available_maps()
)

ADEQUACY_TOLERANCE = 0


RESULTS_DIR = Path(
    "results"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "sweep_v2_multi_map.csv"
)


# ======================================================
# Utilities
# ======================================================

def generate_manhattan_offsets(
    magnitude,
):

    if magnitude < 0:

        raise ValueError(
            "Error magnitude "
            "must be non-negative."
        )

    offsets = []

    for dx in range(
        -magnitude,
        magnitude + 1,
    ):

        for dy in range(
            -magnitude,
            magnitude + 1,
        ):

            if (
                abs(dx)
                + abs(dy)
                == magnitude
            ):

                offsets.append(
                    (dx, dy)
                )

    return offsets


def get_walkable_positions(
    env,
):

    positions = []

    for y, row in enumerate(
        env.grid
    ):

        for x, cell in enumerate(
            row
        ):

            position = (
                x,
                y,
            )

            if cell == "#":
                continue

            if (
                position
                == env.goal_position
            ):
                continue

            positions.append(
                position
            )

    return positions


def action_name(
    action,
):

    if action is None:
        return "NONE"

    return action.value


def case_type(
    result,
):
    """
    Mutually-exclusive consequence classes.
    """

    if not result.belief_plannable:

        return "INVALID_BELIEF"

    if result.false_terminal:

        return "FALSE_TERMINAL"

    if result.action_collision:

        return "COLLISION"

    if result.decision_adequate:

        return "ADEQUATE"

    # Important:
    #
    # valid belief
    # no collision
    # no false terminal
    # BUT decision causes regret
    return "INADEQUATE"


def summarize(
    rows,
):

    total = len(rows)

    invalid = sum(
        row["case_type"]
        == "INVALID_BELIEF"
        for row in rows
    )

    valid = (
        total
        - invalid
    )

    false_terminal = sum(
        row["case_type"]
        == "FALSE_TERMINAL"
        for row in rows
    )

    collisions = sum(
        row["case_type"]
        == "COLLISION"
        for row in rows
    )

    adequate = sum(
        row["case_type"]
        == "ADEQUATE"
        for row in rows
    )

    noncollision_inadequate = sum(
        row["case_type"]
        == "INADEQUATE"
        for row in rows
    )

    inadequate_valid = (
        false_terminal
        + collisions
        + noncollision_inadequate
    )

    regrets = [
        row["decision_regret"]
        for row in rows
        if (
            row["decision_regret"]
            is not None
        )
    ]

    positive_regrets = [
        regret
        for regret in regrets
        if regret > 0
    ]

    return {
        "total":
            total,

        "valid":
            valid,

        "invalid":
            invalid,

        "adequate":
            adequate,

        "inadequate_valid":
            inadequate_valid,

        "false_terminal":
            false_terminal,

        "collisions":
            collisions,

        "noncollision_inadequate":
            noncollision_inadequate,

        "adequate_rate_valid":
            (
                adequate / valid
                if valid
                else 0.0
            ),

        "mean_regret":
            (
                sum(regrets)
                / len(regrets)
                if regrets
                else 0.0
            ),

        "max_regret":
            (
                max(regrets)
                if regrets
                else 0
            ),

        "positive_regret_cases":
            len(
                positive_regrets
            ),
    }


# ======================================================
# Main experiment
# ======================================================

def run_sweep():

    results = []

    # --------------------------------------------------
    # Map sweep
    # --------------------------------------------------

    for map_name in MAP_NAMES:

        env = GridWorld(
            map_name=map_name
        )

        planner = AStarPlanner(
            env.grid
        )

        evaluator = DecisionEvaluator(
            planner=planner,
            adequacy_tolerance=
                ADEQUACY_TOLERANCE,
        )

        true_states = (
            get_walkable_positions(
                env
            )
        )

        # ----------------------------------------------
        # Error-magnitude sweep
        # ----------------------------------------------

        for error_magnitude in (
            ERROR_MAGNITUDES
        ):

            offsets = (
                generate_manhattan_offsets(
                    error_magnitude
                )
            )

            # ------------------------------------------
            # State sweep
            # ------------------------------------------

            for true_state in (
                true_states
            ):

                # Ground truth must itself
                # have a valid route.
                try:

                    planner.next_action(
                        start=true_state,
                        goal=env.goal_position,
                    )

                except (
                    ValueError,
                    RuntimeError,
                ):

                    continue

                clean_belief = (
                    BeliefState(
                        estimated_position=
                            true_state
                    )
                )

                # --------------------------------------
                # Direction sweep
                # --------------------------------------

                for dx, dy in offsets:

                    perturbation = (
                        BeliefPerturbation(
                            PositionOffset(
                                dx=dx,
                                dy=dy,
                            )
                        )
                    )

                    perturbed_belief = (
                        perturbation.apply(
                            clean_belief
                        )
                    )

                    result = (
                        evaluator.evaluate(
                            true_state=
                                true_state,

                            belief=
                                perturbed_belief,

                            goal=
                                env.goal_position,
                        )
                    )

                    results.append(
                        {
                            "map_name":
                                map_name,

                            "error_magnitude":
                                error_magnitude,

                            "true_x":
                                result.true_state[0],

                            "true_y":
                                result.true_state[1],

                            "belief_x":
                                result.belief_state[0],

                            "belief_y":
                                result.belief_state[1],

                            "dx":
                                dx,

                            "dy":
                                dy,

                            "error_manhattan":
                                result.error_manhattan,

                            "oracle_action":
                                action_name(
                                    result.oracle_action
                                ),

                            "agent_action":
                                action_name(
                                    result.agent_action
                                ),

                            "belief_plannable":
                                result.belief_plannable,

                            "decision_diverged":
                                result.decision_diverged,

                            "optimal_cost":
                                result.optimal_cost,

                            "agent_action_cost":
                                result.agent_action_cost,

                            "decision_regret":
                                result.decision_regret,

                            "normalized_regret":
                                result.normalized_regret,

                            "action_collision":
                                result.action_collision,

                            "false_terminal":
                                result.false_terminal,

                            "decision_adequate":
                                result.decision_adequate,

                            "adequacy_tolerance":
                                ADEQUACY_TOLERANCE,

                            "case_type":
                                case_type(
                                    result
                                ),
                        }
                    )

    # ==================================================
    # Save data
    # ==================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if results:

        with OUTPUT_FILE.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=
                    results[0].keys(),
            )

            writer.writeheader()

            writer.writerows(
                results
            )

    # ==================================================
    # Console report
    # ==================================================

    print(
        "=== MSA Adequacy Sweep V2 ==="
    )

    print(
        f"Maps: "
        f"{', '.join(MAP_NAMES)}"
    )

    print(
        f"Error magnitudes: "
        f"{ERROR_MAGNITUDES}"
    )

    print(
        f"Adequacy tolerance: "
        f"{ADEQUACY_TOLERANCE}"
    )

    print()

    print(
        "map         err   "
        "valid   adequate   "
        "inadequate   collision   "
        "noncoll-inad   false-term"
    )

    print(
        "-" * 90
    )

    # --------------------------------------------------
    # Per-map / per-error results
    # --------------------------------------------------

    for map_name in MAP_NAMES:

        for magnitude in (
            ERROR_MAGNITUDES
        ):

            group = [
                row
                for row in results
                if (
                    row["map_name"]
                    == map_name
                    and
                    row["error_magnitude"]
                    == magnitude
                )
            ]

            summary = summarize(
                group
            )

            print(
                f"{map_name:<12} "
                f"{magnitude:>3} "

                f"{summary['valid']:>7} "
                f"{summary['adequate']:>10} "
                f"{summary['inadequate_valid']:>12} "

                f"{summary['collisions']:>11} "
                f"{summary['noncollision_inadequate']:>14} "
                f"{summary['false_terminal']:>12}"
            )

    # ==================================================
    # Overall result
    # ==================================================

    overall = summarize(
        results
    )

    print()

    print(
        "=== Overall ==="
    )

    print(
        f"Total cases:                  "
        f"{overall['total']}"
    )

    print(
        f"Valid beliefs:                "
        f"{overall['valid']}"
    )

    print(
        f"Invalid beliefs:              "
        f"{overall['invalid']}"
    )

    print(
        f"Adequate valid decisions:     "
        f"{overall['adequate']}"
    )

    print(
        f"Inadequate valid decisions:   "
        f"{overall['inadequate_valid']}"
    )

    print(
        f"Non-collision inadequacies:   "
        f"{overall['noncollision_inadequate']}"
    )

    print(
        f"Collisions:                   "
        f"{overall['collisions']}"
    )

    print(
        f"False terminals:              "
        f"{overall['false_terminal']}"
    )

    print(
        f"Mean regret:                  "
        f"{overall['mean_regret']:.3f}"
    )

    print(
        f"Maximum regret:               "
        f"{overall['max_regret']}"
    )

    # ==================================================
    # Most interesting examples
    # ==================================================

    examples = [
        row
        for row in results
        if (
            row["case_type"]
            == "INADEQUATE"

            and

            row["decision_regret"]
            is not None

            and

            row["decision_regret"]
            > 0
        )
    ][:8]

    print()

    print(
        "=== Non-Collision "
        "Inadequate Examples ==="
    )

    if not examples:

        print(
            "No examples found."
        )

    for row in examples:

        print(
            f"map={row['map_name']} "
            f"err={row['error_magnitude']} "

            f"true="
            f"({row['true_x']},"
            f"{row['true_y']}) "

            f"belief="
            f"({row['belief_x']},"
            f"{row['belief_y']}) "

            f"oracle="
            f"{row['oracle_action']} "

            f"agent="
            f"{row['agent_action']} "

            f"regret="
            f"{row['decision_regret']}"
        )

    print()

    print(
        f"Results saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    run_sweep()