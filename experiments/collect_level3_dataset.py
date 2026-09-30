import argparse
import csv
from pathlib import Path

from src.env import GridWorld
from src.planner import AStarPlanner

from src.belief import (
    OccupancyBelief,
)

from src.observation import (
    LocalObserver,
)

from src.evaluator import (
    DecisionEvaluator,
)

from src.generated_maps import (
    generate_random_map,
    obstacle_probability_for_seed,
)

from src.level3_features import (
    Level3FeatureExtractor,
    FULL_FEATURES,
)


RESULTS_DIR = Path(
    "results"
)


TRAIN_OUTPUT = (
    RESULTS_DIR
    / "level3_train.csv"
)


VAL_OUTPUT = (
    RESULTS_DIR
    / "level3_val.csv"
)


TEST_OUTPUT = (
    RESULTS_DIR
    / "level3_test.csv"
)


# ======================================================
# IMPORTANT SPLIT POLICY
# ======================================================

# Training maps.
TRAIN_SEEDS = range(
    0,
    120,
)

# Validation maps.
#
# Hyperparameter / architecture decisions may use these.
VAL_SEEDS = range(
    1000,
    1040,
)

# Final unseen test maps.
#
# Do NOT repeatedly tune against these.
TEST_SEEDS = range(
    10000,
    10040,
)


def plan_from_belief(
    belief,
):
    """
    Agent-side optimistic planner.

    Same assumption as Level 2:
        UNKNOWN -> free
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


def collect_episode(
    seed: int,
    split: str,
    max_steps: int = 300,
):
    """
    Collect one passive-observation episode.

    No QueryReality is used here.

    Purpose:
        collect agent-visible decision contexts
        and evaluator-side adequacy labels.

    Level-3 learning target:

        1 = INADEQUATE
        0 = ADEQUATE

    The learned model should therefore estimate:

        P(Inadequate | agent-visible context)
    """

    obstacle_probability = (
        obstacle_probability_for_seed(
            seed
        )
    )

    generated = (
        generate_random_map(
            seed=seed,

            width=15,
            height=9,

            obstacle_probability=
                obstacle_probability,

            min_detour_extra=2,
        )
    )

    env = GridWorld(
        grid_map=
            generated.grid
    )

    observer = LocalObserver(
        normal_radius=1,
        query_radius=4,
    )

    belief = OccupancyBelief(
        width=
            env.width,

        height=
            env.height,

        goal_position=
            env.goal_position,
    )

    feature_extractor = (
        Level3FeatureExtractor(
            sensitivity_radius=4,
            near_horizon=4,
            local_radius=3,
        )
    )

    # ==================================================
    # EXPERIMENT-SIDE evaluator
    #
    # Ground truth lives here only.
    # ==================================================

    true_planner = (
        AStarPlanner(
            env.grid
        )
    )

    evaluator = (
        DecisionEvaluator(
            planner=
                true_planner,

            adequacy_tolerance=0,
        )
    )

    rows = []

    feature_extractor.reset()

    env.reset()

    while (
        not env.done
        and
        env.steps
        < max_steps
    ):

        # ----------------------------------------------
        # Agent receives passive observation
        # ----------------------------------------------

        observation = (
            observer.observe(
                env
            )
        )

        belief.update(
            observation
        )

        # ----------------------------------------------
        # Agent chooses candidate action
        # ----------------------------------------------

        try:

            candidate_action = (
                plan_from_belief(
                    belief
                )
            )

        except (
            ValueError,
            RuntimeError,
        ):

            break

        if candidate_action is None:
            break

        # ==============================================
        # AGENT-SIDE INPUT X
        # ==============================================

        feature_vector = (
            feature_extractor.extract(
                belief=belief,

                candidate_action=
                    candidate_action,

                goal=
                    env.goal_position,
            )
        )

        # ==============================================
        # EXPERIMENT-SIDE LABEL y
        #
        # NEVER give these fields to the model.
        # ==============================================

        evaluation = (
            evaluator.evaluate_action(
                true_state=
                    env.agent_position,

                agent_action=
                    candidate_action,

                goal=
                    env.goal_position,
            )
        )

        label_inadequate = int(
            not evaluation
            .decision_adequate
        )

        row = {
            # ------------------------------------------
            # Dataset identity
            # ------------------------------------------

            "split":
                split,

            "map_seed":
                seed,

            "step":
                env.steps,

            "obstacle_probability":
                generated
                .obstacle_probability,

            "true_shortest_path_length":
                generated
                .shortest_path_length,

            "map_detour_extra":
                generated
                .detour_extra,

            # ==========================================
            # MODEL INPUTS
            # ==========================================

            **feature_vector.as_dict(),

            # ==========================================
            # LABEL
            # ==========================================

            "label_inadequate":
                label_inadequate,

            # ==========================================
            # EXPERIMENT-ONLY METADATA
            #
            # DO NOT use as model input.
            # ==========================================

            "meta_decision_regret":
                (
                    evaluation
                    .decision_regret
                ),

            "meta_normalized_regret":
                (
                    evaluation
                    .normalized_regret
                ),

            "meta_collision":
                int(
                    evaluation
                    .action_collision
                ),

            "meta_decision_diverged":
                int(
                    evaluation
                    .decision_diverged
                ),
        }

        rows.append(
            row
        )

        # ----------------------------------------------
        # Execute candidate action in reality
        # ----------------------------------------------

        env.step(
            candidate_action
        )

    return rows


def collect_split(
    name,
    seeds,
    output_path,
):

    rows = []

    successful_maps = 0

    for index, seed in enumerate(
        seeds,
        start=1,
    ):

        episode_rows = (
            collect_episode(
                seed=seed,
                split=name,
            )
        )

        if episode_rows:
            successful_maps += 1

        rows.extend(
            episode_rows
        )

        if (
            index % 20
            == 0
        ):

            print(
                f"{name}: "
                f"{index} maps processed"
            )

    if not rows:

        raise RuntimeError(
            f"No samples collected "
            f"for split={name}."
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = (
        list(
            rows[0].keys()
        )
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    inadequate = sum(
        row[
            "label_inadequate"
        ]
        for row in rows
    )

    adequate = (
        len(rows)
        - inadequate
    )

    inadequate_rate = (
        inadequate
        / len(rows)
    )

    print()

    print(
        f"=== {name.upper()} ==="
    )

    print(
        f"Maps:             "
        f"{successful_maps}"
    )

    print(
        f"Samples:          "
        f"{len(rows)}"
    )

    print(
        f"Adequate:         "
        f"{adequate}"
    )

    print(
        f"Inadequate:       "
        f"{inadequate}"
    )

    print(
        f"Inadequate rate:  "
        f"{inadequate_rate:.3f}"
    )

    print(
        f"Saved to:         "
        f"{output_path}"
    )

    return {
        "maps":
            successful_maps,

        "samples":
            len(rows),

        "adequate":
            adequate,

        "inadequate":
            inadequate,

        "inadequate_rate":
            inadequate_rate,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--include-test",
        action="store_true",

        help=(
            "Also collect the reserved final "
            "test split. Do not use this "
            "during normal model tuning."
        ),
    )

    args = parser.parse_args()

    print(
        "=== MSA Level 3 Dataset Collection ==="
    )

    print()

    print(
        "Model feature count: "
        f"{len(FULL_FEATURES)}"
    )

    print()

    collect_split(
        name="train",
        seeds=TRAIN_SEEDS,
        output_path=
            TRAIN_OUTPUT,
    )

    print()

    collect_split(
        name="val",
        seeds=VAL_SEEDS,
        output_path=
            VAL_OUTPUT,
    )

    if args.include_test:

        print()

        print(
            "WARNING:"
        )

        print(
            "Collecting reserved TEST split."
        )

        print(
            "Do not tune model decisions "
            "against this split."
        )

        print()

        collect_split(
            name="test",
            seeds=TEST_SEEDS,
            output_path=
                TEST_OUTPUT,
        )


if __name__ == "__main__":
    main()