import csv
from pathlib import Path

import numpy as np

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

from src.level3_pathcentric import (
    PathCentricRepresentation,
)

from src.learned_adequacy import (
    LogisticAdequacyModel,
)

from experiments.collect_level3_dataset import (
    TRAIN_SEEDS,
    VAL_SEEDS,
)

from experiments.train_level3_mlp import (
    evaluate_model,
    choose_for_recall,
    choose_best_f1,
)


# ======================================================
# Configuration
# ======================================================

RESULTS_DIR = Path(
    "results"
)

MODEL_DIR = Path(
    "models"
)


OLD_TRAIN_FILE = (
    RESULTS_DIR
    / "level3_train.csv"
)

OLD_VAL_FILE = (
    RESULTS_DIR
    / "level3_val.csv"
)


SCALAR_MODEL_FILE = (
    MODEL_DIR
    / "level3_logistic_full.npz"
)


PATH_MODEL_FILE = (
    MODEL_DIR
    / "level3_logistic_path.npz"
)


HYBRID_MODEL_FILE = (
    MODEL_DIR
    / "level3_logistic_scalar_path.npz"
)


PATH_TRAIN_FILE = (
    RESULTS_DIR
    / "level3_path_train.npz"
)


PATH_VAL_FILE = (
    RESULTS_DIR
    / "level3_path_val.npz"
)


COMPARISON_FILE = (
    RESULTS_DIR
    / "level3_scalar_vs_path.csv"
)


PATH_HORIZON = 16
PATH_LOCAL_RADIUS = 1


# ======================================================
# Planner
# ======================================================

def plan_from_belief(
    belief,
):
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


# ======================================================
# Collect episode
# ======================================================

def collect_episode(
    seed,
    split,
    path_encoder,
    max_steps=300,
):
    obstacle_probability = (
        obstacle_probability_for_seed(
            seed
        )
    )

    generated = generate_random_map(
        seed=seed,
        width=15,
        height=9,
        obstacle_probability=
            obstacle_probability,
        min_detour_extra=2,
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
        width=env.width,
        height=env.height,
        goal_position=
            env.goal_position,
    )

    scalar_extractor = (
        Level3FeatureExtractor(
            sensitivity_radius=4,
            near_horizon=4,
            local_radius=3,
        )
    )

    # ==================================================
    # Experiment-side evaluator
    # ==================================================

    true_planner = AStarPlanner(
        env.grid
    )

    evaluator = DecisionEvaluator(
        planner=true_planner,
        adequacy_tolerance=0,
    )

    scalar_extractor.reset()

    env.reset()

    records = []

    while (
        not env.done
        and env.steps < max_steps
    ):
        # ----------------------------------------------
        # Passive agent observation
        # ----------------------------------------------

        belief.update(
            observer.observe(
                env
            )
        )

        # ----------------------------------------------
        # Candidate action
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

        # ----------------------------------------------
        # Existing scalar representation
        # ----------------------------------------------

        scalar_dict = (
            scalar_extractor.extract(
                belief=belief,
                candidate_action=
                    candidate_action,
                goal=
                    env.goal_position,
            )
            .as_dict()
        )

        scalar_vector = np.asarray(
            [
                scalar_dict[name]
                for name
                in FULL_FEATURES
            ],
            dtype=np.float64,
        )

        # ----------------------------------------------
        # NEW path-centric representation
        # ----------------------------------------------

        path_vector = (
            path_encoder.extract(
                belief=belief,
                goal=
                    env.goal_position,
            )
        )

        # ----------------------------------------------
        # Experiment-only adequacy label
        # ----------------------------------------------

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

        label = int(
            not evaluation
            .decision_adequate
        )

        records.append(
            {
                "split":
                    split,

                "seed":
                    int(seed),

                "step":
                    int(
                        env.steps
                    ),

                "scalar":
                    scalar_vector,

                "path":
                    path_vector,

                "label":
                    label,
            }
        )

        # ----------------------------------------------
        # Execute candidate action
        # ----------------------------------------------

        env.step(
            candidate_action
        )

    return records


# ======================================================
# Collect split
# ======================================================

def collect_split(
    split,
    seeds,
    path_encoder,
):
    all_records = []

    seeds = list(
        seeds
    )

    for index, seed in enumerate(
        seeds,
        start=1,
    ):
        episode_records = (
            collect_episode(
                seed=seed,
                split=split,
                path_encoder=
                    path_encoder,
            )
        )

        all_records.extend(
            episode_records
        )

        if index % 20 == 0:
            print(
                f"{split}: "
                f"{index} maps processed"
            )

    if not all_records:
        raise RuntimeError(
            f"No samples for "
            f"split={split}."
        )

    return {
        "x_scalar":
            np.stack(
                [
                    record["scalar"]
                    for record
                    in all_records
                ]
            ),

        "x_path":
            np.stack(
                [
                    record["path"]
                    for record
                    in all_records
                ]
            ),

        "y":
            np.asarray(
                [
                    record["label"]
                    for record
                    in all_records
                ],
                dtype=np.int64,
            ),

        "seed":
            np.asarray(
                [
                    record["seed"]
                    for record
                    in all_records
                ],
                dtype=np.int64,
            ),

        "step":
            np.asarray(
                [
                    record["step"]
                    for record
                    in all_records
                ],
                dtype=np.int64,
            ),
    }


# ======================================================
# Dataset verification
# ======================================================

def verify_existing_dataset(
    csv_path,
    new_data,
):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Existing dataset not found: "
            f"{csv_path}"
        )

    old_records = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        for row in reader:
            old_records.append(
                (
                    int(
                        row["map_seed"]
                    ),
                    int(
                        row["step"]
                    ),
                    int(
                        row[
                            "label_inadequate"
                        ]
                    ),
                )
            )

    new_records = list(
        zip(
            new_data[
                "seed"
            ].tolist(),

            new_data[
                "step"
            ].tolist(),

            new_data[
                "y"
            ].tolist(),
        )
    )

    if (
        len(old_records)
        != len(new_records)
    ):
        raise RuntimeError(
            "Dataset size mismatch: "
            f"old={len(old_records)}, "
            f"new={len(new_records)}"
        )

    for index, (
        old_record,
        new_record,
    ) in enumerate(
        zip(
            old_records,
            new_records,
        )
    ):
        if (
            old_record
            != new_record
        ):
            raise RuntimeError(
                f"Dataset mismatch at "
                f"row {index}: "
                f"old={old_record}, "
                f"new={new_record}"
            )

    print(
        "Dataset verification PASS: "
        f"{len(old_records)} samples"
    )


# ======================================================
# Save structured dataset
# ======================================================

def save_dataset(
    path,
    data,
):
    np.savez_compressed(
        path,

        x_scalar=
            data["x_scalar"],

        x_path=
            data["x_path"],

        y=
            data["y"],

        seed=
            data["seed"],

        step=
            data["step"],
    )


# ======================================================
# Logistic training
# ======================================================

def train_logistic(
    name,
    feature_names,
    x_train,
    y_train,
):
    print(
        f"Training {name}: "
        f"{x_train.shape[1]} features"
    )

    model = LogisticAdequacyModel(
        feature_names=
            feature_names
    )

    info = model.fit(
        x=x_train,
        y=y_train,
        epochs=1500,
        learning_rate=0.05,
        l2=1e-4,
        positive_weight=None,
        verbose=False,
    )

    print(
        f"  epochs="
        f"{info['epochs']} "
        f"loss="
        f"{info['final_loss']:.6f}"
    )

    return model


# ======================================================
# Comparison output
# ======================================================

def print_comparison(
    model_results,
):
    print()

    print(
        "=== Threshold-independent ==="
    )

    print(
        "model                    "
        "features   AUROC      AP"
    )

    print(
        "-" * 53
    )

    for result in model_results:
        print(
            f"{result['name']:<25}"
            f"{result['features']:>8} "
            f"{result['auc']:>7.3f} "
            f"{result['ap']:>7.3f}"
        )

    print()

    print(
        "=== High-Recall Comparison ==="
    )

    print(
        "target  model                    "
        "P       R       F1      FP    FN"
    )

    print(
        "-" * 76
    )

    for target in [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]:
        for result in model_results:
            point = choose_for_recall(
                result["sweep"],
                target,
            )

            if point is None:
                continue

            print(
                f"{target:>6.2f} "
                f"{result['name']:<24}"
                f"{point['precision']:>7.3f} "
                f"{point['recall']:>7.3f} "
                f"{point['f1']:>7.3f} "
                f"{point['fp']:>5} "
                f"{point['fn']:>5}"
            )

        print()

    print(
        "=== Best F1 ==="
    )

    print(
        "model                    "
        "P       R       F1      FP    FN"
    )

    print(
        "-" * 63
    )

    for result in model_results:
        point = choose_best_f1(
            result["sweep"]
        )

        print(
            f"{result['name']:<25}"
            f"{point['precision']:>7.3f} "
            f"{point['recall']:>7.3f} "
            f"{point['f1']:>7.3f} "
            f"{point['fp']:>5} "
            f"{point['fn']:>5}"
        )


# ======================================================
# Main
# ======================================================

def main():
    print(
        "=== MSA Level 3 "
        "Path-Centric Representation Test ==="
    )

    print()

    print(
        "Research question:"
    )

    print(
        "Does uncertainty structure along the "
        "agent's intended future route improve "
        "adequacy prediction?"
    )

    print()

    path_encoder = (
        PathCentricRepresentation(
            horizon=
                PATH_HORIZON,

            local_radius=
                PATH_LOCAL_RADIUS,
        )
    )

    path_names = (
        path_encoder
        .feature_names()
    )

    print(
        f"Scalar features: "
        f"{len(FULL_FEATURES)}"
    )

    print(
        f"Path features:   "
        f"{len(path_names)}"
    )

    print(
        f"Hybrid features: "
        f"{len(FULL_FEATURES) + len(path_names)}"
    )

    print()

    # ==================================================
    # Recreate exact train / val trajectories
    # ==================================================

    print(
        "Collecting TRAIN path states..."
    )

    train_data = collect_split(
        split="train",
        seeds=TRAIN_SEEDS,
        path_encoder=
            path_encoder,
    )

    print()

    print(
        "Collecting VAL path states..."
    )

    val_data = collect_split(
        split="val",
        seeds=VAL_SEEDS,
        path_encoder=
            path_encoder,
    )

    print()

    # ==================================================
    # Verify exact equivalence
    # ==================================================

    print(
        "=== Dataset Verification ==="
    )

    verify_existing_dataset(
        OLD_TRAIN_FILE,
        train_data,
    )

    verify_existing_dataset(
        OLD_VAL_FILE,
        val_data,
    )

    print()

    print(
        f"Train scalar: "
        f"{train_data['x_scalar'].shape}"
    )

    print(
        f"Train path:   "
        f"{train_data['x_path'].shape}"
    )

    print(
        f"Val scalar:   "
        f"{val_data['x_scalar'].shape}"
    )

    print(
        f"Val path:     "
        f"{val_data['x_path'].shape}"
    )

    print()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_dataset(
        PATH_TRAIN_FILE,
        train_data,
    )

    save_dataset(
        PATH_VAL_FILE,
        val_data,
    )

    # ==================================================
    # Data
    # ==================================================

    x_train_scalar = (
        train_data[
            "x_scalar"
        ]
    )

    x_val_scalar = (
        val_data[
            "x_scalar"
        ]
    )

    x_train_path = (
        train_data[
            "x_path"
        ]
    )

    x_val_path = (
        val_data[
            "x_path"
        ]
    )

    y_train = (
        train_data[
            "y"
        ]
    )

    y_val = (
        val_data[
            "y"
        ]
    )

    x_train_hybrid = (
        np.concatenate(
            [
                x_train_scalar,
                x_train_path,
            ],
            axis=1,
        )
    )

    x_val_hybrid = (
        np.concatenate(
            [
                x_val_scalar,
                x_val_path,
            ],
            axis=1,
        )
    )

    # ==================================================
    # Existing scalar baseline
    # ==================================================

    if not SCALAR_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Scalar Logistic model missing.\n"
            "Run:\n"
            "python -m experiments.train_level3"
        )

    scalar_model = (
        LogisticAdequacyModel.load(
            SCALAR_MODEL_FILE
        )
    )

    scalar_scores = (
        scalar_model.predict_proba(
            x_val_scalar
        )
    )

    # ==================================================
    # Path Logistic
    # ==================================================

    path_model = train_logistic(
        name=
            "path logistic",

        feature_names=
            path_names,

        x_train=
            x_train_path,

        y_train=
            y_train,
    )

    path_scores = (
        path_model.predict_proba(
            x_val_path
        )
    )

    # ==================================================
    # Scalar + Path Logistic
    # ==================================================

    hybrid_names = (
        list(
            FULL_FEATURES
        )
        +
        path_names
    )

    hybrid_model = train_logistic(
        name=
            "scalar + path logistic",

        feature_names=
            hybrid_names,

        x_train=
            x_train_hybrid,

        y_train=
            y_train,
    )

    hybrid_scores = (
        hybrid_model.predict_proba(
            x_val_hybrid
        )
    )

    # ==================================================
    # Evaluation
    # ==================================================

    (
        scalar_rows,
        scalar_sweep,
        scalar_auc,
        scalar_ap,
    ) = evaluate_model(
        model_name=
            "scalar_logistic",

        probabilities=
            scalar_scores,

        y_val=
            y_val,
    )

    (
        path_rows,
        path_sweep,
        path_auc,
        path_ap,
    ) = evaluate_model(
        model_name=
            "path_logistic",

        probabilities=
            path_scores,

        y_val=
            y_val,
    )

    (
        hybrid_rows,
        hybrid_sweep,
        hybrid_auc,
        hybrid_ap,
    ) = evaluate_model(
        model_name=
            "scalar_path_logistic",

        probabilities=
            hybrid_scores,

        y_val=
            y_val,
    )

    # ==================================================
    # Save models using recall >= .90 operating point
    # ==================================================

    path_r90 = choose_for_recall(
        path_sweep,
        0.90,
    )

    hybrid_r90 = choose_for_recall(
        hybrid_sweep,
        0.90,
    )

    if path_r90 is not None:
        path_model.threshold = (
            path_r90[
                "threshold"
            ]
        )

    if hybrid_r90 is not None:
        hybrid_model.threshold = (
            hybrid_r90[
                "threshold"
            ]
        )

    path_model.save(
        PATH_MODEL_FILE
    )

    hybrid_model.save(
        HYBRID_MODEL_FILE
    )

    # ==================================================
    # Save comparison
    # ==================================================

    comparison_rows = (
        scalar_rows
        + path_rows
        + hybrid_rows
    )

    with COMPARISON_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=
                list(
                    comparison_rows[
                        0
                    ].keys()
                ),
        )

        writer.writeheader()

        writer.writerows(
            comparison_rows
        )

    # ==================================================
    # Print
    # ==================================================

    model_results = [
        {
            "name":
                "Scalar Logistic",

            "features":
                len(
                    FULL_FEATURES
                ),

            "auc":
                scalar_auc,

            "ap":
                scalar_ap,

            "sweep":
                scalar_sweep,
        },

        {
            "name":
                "Path Logistic",

            "features":
                len(
                    path_names
                ),

            "auc":
                path_auc,

            "ap":
                path_ap,

            "sweep":
                path_sweep,
        },

        {
            "name":
                "Scalar + Path Logistic",

            "features":
                (
                    len(
                        FULL_FEATURES
                    )
                    +
                    len(
                        path_names
                    )
                ),

            "auc":
                hybrid_auc,

            "ap":
                hybrid_ap,

            "sweep":
                hybrid_sweep,
        },
    ]

    print_comparison(
        model_results
    )

    # ==================================================
    # Diagnostic
    # ==================================================

    print()

    print(
        "=== Path Representation Diagnostic ==="
    )

    auc_gain = (
        hybrid_auc
        - scalar_auc
    )

    ap_gain = (
        hybrid_ap
        - scalar_ap
    )

    print(
        f"Scalar+Path - Scalar AUROC: "
        f"{auc_gain:+.3f}"
    )

    print(
        f"Scalar+Path - Scalar AP:    "
        f"{ap_gain:+.3f}"
    )

    print()

    if (
        auc_gain >= 0.03
        or
        ap_gain >= 0.05
    ):
        print(
            "Meaningful path-centric gain."
        )

        print(
            "Future route structure contains "
            "decision-relevant information that "
            "scalar summaries were discarding."
        )

    elif (
        auc_gain >= 0.01
        or
        ap_gain >= 0.02
    ):
        print(
            "Modest path-centric gain."
        )

        print(
            "Route structure helps, but it is "
            "not yet a complete adequacy "
            "representation."
        )

    else:
        print(
            "Little path-centric gain."
        )

        print(
            "For the CURRENT one-step adequacy label, "
            "this route representation adds little."
        )

        print(
            "Do not conclude that future-route "
            "information is useless for QueryReality; "
            "that requires a horizon/query-value label."
        )

    print()

    print(
        f"Saved path model: "
        f"{PATH_MODEL_FILE}"
    )

    print(
        f"Saved hybrid model: "
        f"{HYBRID_MODEL_FILE}"
    )

    print(
        f"Saved comparison: "
        f"{COMPARISON_FILE}"
    )

    print()

    print(
        "TEST split was NOT used."
    )


if __name__ == "__main__":
    main()