import csv
from pathlib import Path

import numpy as np

from src.env import (
    GridWorld,
)

from src.planner import (
    AStarPlanner,
)

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

from src.level3_spatial import (
    SpatialBeliefPatch,
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
# Paths
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


SPATIAL_TRAIN_FILE = (
    RESULTS_DIR
    / "level3_spatial_train.npz"
)


SPATIAL_VAL_FILE = (
    RESULTS_DIR
    / "level3_spatial_val.npz"
)


SCALAR_LOGISTIC_FILE = (
    MODEL_DIR
    / "level3_logistic_full.npz"
)


PATCH_MODEL_FILE = (
    MODEL_DIR
    / "level3_logistic_patch.npz"
)


HYBRID_MODEL_FILE = (
    MODEL_DIR
    / "level3_logistic_hybrid.npz"
)


COMPARISON_FILE = (
    RESULTS_DIR
    / "level3_scalar_vs_spatial.csv"
)


PATCH_SIZE = 9


# ======================================================
# Agent planner
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
# Collect one episode
# ======================================================

def collect_episode(
    seed,
    split,
    spatial_encoder,
    max_steps=300,
):

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

    scalar_extractor = (
        Level3FeatureExtractor(
            sensitivity_radius=4,
            near_horizon=4,
            local_radius=3,
        )
    )

    # ==================================================
    # Experiment-side ground truth
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

    scalar_extractor.reset()

    env.reset()

    records = []

    while (
        not env.done
        and
        env.steps
        < max_steps
    ):

        # ==============================================
        # Agent observation
        # ==============================================

        belief.update(
            observer.observe(
                env
            )
        )

        # ==============================================
        # Candidate action
        # ==============================================

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
        # Existing scalar representation
        # ==============================================

        scalar_features = (
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
                scalar_features[
                    name
                ]
                for name
                in FULL_FEATURES
            ],
            dtype=np.float64,
        )

        # ==============================================
        # NEW structured representation
        # ==============================================

        patch = (
            spatial_encoder.extract(
                belief=belief,

                goal=
                    env.goal_position,
            )
        )

        patch_vector = (
            spatial_encoder.flatten(
                patch
            )
        )

        # ==============================================
        # Experiment-side label
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

                "patch":
                    patch_vector,

                "label":
                    label,
            }
        )

        # ==============================================
        # Execute reality action
        # ==============================================

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
    spatial_encoder,
):

    all_records = []

    seeds = list(
        seeds
    )

    for index, seed in enumerate(
        seeds,
        start=1,
    ):

        records = (
            collect_episode(
                seed=seed,

                split=split,

                spatial_encoder=
                    spatial_encoder,
            )
        )

        all_records.extend(
            records
        )

        if (
            index % 20
            == 0
        ):

            print(
                f"{split}: "
                f"{index} maps processed"
            )

    if not all_records:

        raise RuntimeError(
            f"No records for "
            f"split={split}."
        )

    x_scalar = np.stack(
        [
            record[
                "scalar"
            ]
            for record
            in all_records
        ]
    )

    x_patch = np.stack(
        [
            record[
                "patch"
            ]
            for record
            in all_records
        ]
    )

    y = np.asarray(
        [
            record[
                "label"
            ]
            for record
            in all_records
        ],
        dtype=np.int64,
    )

    seed_array = np.asarray(
        [
            record[
                "seed"
            ]
            for record
            in all_records
        ],
        dtype=np.int64,
    )

    step_array = np.asarray(
        [
            record[
                "step"
            ]
            for record
            in all_records
        ],
        dtype=np.int64,
    )

    return {
        "x_scalar":
            x_scalar,

        "x_patch":
            x_patch,

        "y":
            y,

        "seed":
            seed_array,

        "step":
            step_array,
    }


# ======================================================
# Verify against existing scalar dataset
# ======================================================

def verify_existing_dataset(
    csv_path,
    new_data,
):
    """
    Strong apples-to-apples check.

    The spatial experiment must contain exactly
    the same:

        seed
        step
        adequacy label

    as the previous scalar dataset.
    """

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
                        row[
                            "map_seed"
                        ]
                    ),

                    int(
                        row[
                            "step"
                        ]
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
                "Dataset mismatch at "
                f"row={index}: "
                f"old={old_record}, "
                f"new={new_record}"
            )

    print(
        f"Dataset verification PASS: "
        f"{len(old_records)} samples"
    )


# ======================================================
# Save spatial arrays
# ======================================================

def save_dataset(
    path,
    data,
):

    np.savez_compressed(
        path,

        x_scalar=
            data[
                "x_scalar"
            ],

        x_patch=
            data[
                "x_patch"
            ],

        y=
            data[
                "y"
            ],

        seed=
            data[
                "seed"
            ],

        step=
            data[
                "step"
            ],
    )


# ======================================================
# Train logistic
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

    model = (
        LogisticAdequacyModel(
            feature_names=
                feature_names
        )
    )

    info = (
        model.fit(
            x=x_train,

            y=y_train,

            epochs=1500,

            learning_rate=0.05,

            l2=1e-4,

            positive_weight=None,

            verbose=False,
        )
    )

    print(
        f"  epochs="
        f"{info['epochs']} "
        f"loss="
        f"{info['final_loss']:.6f}"
    )

    return model


# ======================================================
# Comparison report
# ======================================================

def print_operating_comparison(
    model_results,
):

    print()

    print(
        "=== Threshold-independent ==="
    )

    print(
        "model               "
        "features   AUROC      AP"
    )

    print(
        "-" * 48
    )

    for result in (
        model_results
    ):

        print(
            f"{result['name']:<20}"
            f"{result['features']:>8} "
            f"{result['auc']:>7.3f} "
            f"{result['ap']:>7.3f}"
        )

    print()

    print(
        "=== High-Recall Comparison ==="
    )

    print(
        "target  model               "
        "P       R       F1      FP    FN"
    )

    print(
        "-" * 70
    )

    for target in [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]:

        for result in (
            model_results
        ):

            point = (
                choose_for_recall(
                    result[
                        "sweep"
                    ],

                    target,
                )
            )

            if point is None:
                continue

            print(
                f"{target:>6.2f} "
                f"{result['name']:<19}"
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
        "model               "
        "P       R       F1      FP    FN"
    )

    print(
        "-" * 58
    )

    for result in (
        model_results
    ):

        point = (
            choose_best_f1(
                result[
                    "sweep"
                ]
            )
        )

        print(
            f"{result['name']:<20}"
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
        "Spatial Representation Test ==="
    )

    print()

    print(
        "Research question:"
    )

    print(
        "Does preserving spatial/topological "
        "belief structure improve adequacy prediction?"
    )

    print()

    # ==================================================
    # Encoder
    # ==================================================

    spatial_encoder = (
        SpatialBeliefPatch(
            patch_size=
                PATCH_SIZE
        )
    )

    spatial_names = (
        spatial_encoder
        .feature_names()
    )

    print(
        f"Scalar features: "
        f"{len(FULL_FEATURES)}"
    )

    print(
        f"Spatial features: "
        f"{len(spatial_names)}"
    )

    print(
        f"Hybrid features: "
        f"{len(FULL_FEATURES) + len(spatial_names)}"
    )

    print()

    # ==================================================
    # Collect exact same train / val trajectories
    # ==================================================

    print(
        "Collecting TRAIN spatial states..."
    )

    train_data = (
        collect_split(
            split="train",

            seeds=
                TRAIN_SEEDS,

            spatial_encoder=
                spatial_encoder,
        )
    )

    print()

    print(
        "Collecting VAL spatial states..."
    )

    val_data = (
        collect_split(
            split="val",

            seeds=
                VAL_SEEDS,

            spatial_encoder=
                spatial_encoder,
        )
    )

    print()

    # ==================================================
    # Verify apples-to-apples dataset
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
        f"Train shape scalar: "
        f"{train_data['x_scalar'].shape}"
    )

    print(
        f"Train shape patch:  "
        f"{train_data['x_patch'].shape}"
    )

    print(
        f"Val shape scalar:   "
        f"{val_data['x_scalar'].shape}"
    )

    print(
        f"Val shape patch:    "
        f"{val_data['x_patch'].shape}"
    )

    print()

    # ==================================================
    # Save structured datasets
    # ==================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_dataset(
        SPATIAL_TRAIN_FILE,
        train_data,
    )

    save_dataset(
        SPATIAL_VAL_FILE,
        val_data,
    )

    # ==================================================
    # Inputs
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

    x_train_patch = (
        train_data[
            "x_patch"
        ]
    )

    x_val_patch = (
        val_data[
            "x_patch"
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
                x_train_patch,
            ],
            axis=1,
        )
    )

    x_val_hybrid = (
        np.concatenate(
            [
                x_val_scalar,
                x_val_patch,
            ],
            axis=1,
        )
    )

    # ==================================================
    # Existing scalar Logistic baseline
    # ==================================================

    if not (
        SCALAR_LOGISTIC_FILE.exists()
    ):

        raise FileNotFoundError(
            "Existing scalar logistic model "
            "not found.\n"
            "Run:\n"
            "python -m experiments.train_level3"
        )

    scalar_model = (
        LogisticAdequacyModel
        .load(
            SCALAR_LOGISTIC_FILE
        )
    )

    scalar_scores = (
        scalar_model
        .predict_proba(
            x_val_scalar
        )
    )

    # ==================================================
    # Patch-only Logistic
    # ==================================================

    patch_model = (
        train_logistic(
            name=
                "patch logistic",

            feature_names=
                spatial_names,

            x_train=
                x_train_patch,

            y_train=
                y_train,
        )
    )

    patch_scores = (
        patch_model.predict_proba(
            x_val_patch
        )
    )

    # ==================================================
    # Scalar + spatial Logistic
    # ==================================================

    hybrid_names = (
        list(
            FULL_FEATURES
        )
        +
        spatial_names
    )

    hybrid_model = (
        train_logistic(
            name=
                "hybrid logistic",

            feature_names=
                hybrid_names,

            x_train=
                x_train_hybrid,

            y_train=
                y_train,
        )
    )

    hybrid_scores = (
        hybrid_model.predict_proba(
            x_val_hybrid
        )
    )

    # ==================================================
    # Evaluate
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
        patch_rows,
        patch_sweep,
        patch_auc,
        patch_ap,
    ) = evaluate_model(
        model_name=
            "patch_logistic",

        probabilities=
            patch_scores,

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
            "hybrid_logistic",

        probabilities=
            hybrid_scores,

        y_val=
            y_val,
    )

    # ==================================================
    # Canonical r >= .90 threshold for saved models
    # ==================================================

    patch_r90 = (
        choose_for_recall(
            patch_sweep,
            0.90,
        )
    )

    hybrid_r90 = (
        choose_for_recall(
            hybrid_sweep,
            0.90,
        )
    )

    if patch_r90 is not None:

        patch_model.threshold = (
            patch_r90[
                "threshold"
            ]
        )

    if hybrid_r90 is not None:

        hybrid_model.threshold = (
            hybrid_r90[
                "threshold"
            ]
        )

    patch_model.save(
        PATCH_MODEL_FILE
    )

    hybrid_model.save(
        HYBRID_MODEL_FILE
    )

    # ==================================================
    # Save comparison
    # ==================================================

    comparison_rows = (
        scalar_rows
        + patch_rows
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
                    comparison_rows[0]
                    .keys()
                ),
        )

        writer.writeheader()

        writer.writerows(
            comparison_rows
        )

    # ==================================================
    # Report
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
                "Patch Logistic",

            "features":
                len(
                    spatial_names
                ),

            "auc":
                patch_auc,

            "ap":
                patch_ap,

            "sweep":
                patch_sweep,
        },

        {
            "name":
                "Hybrid Logistic",

            "features":
                len(
                    hybrid_names
                ),

            "auc":
                hybrid_auc,

            "ap":
                hybrid_ap,

            "sweep":
                hybrid_sweep,
        },
    ]

    print_operating_comparison(
        model_results
    )

    # ==================================================
    # Representation diagnostic
    # ==================================================

    print()

    print(
        "=== Representation Diagnostic ==="
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
        f"Hybrid - Scalar AUROC: "
        f"{auc_gain:+.3f}"
    )

    print(
        f"Hybrid - Scalar AP:    "
        f"{ap_gain:+.3f}"
    )

    print()

    if (
        auc_gain >= 0.03
        or
        ap_gain >= 0.05
    ):

        print(
            "Meaningful spatial-representation gain."
        )

        print(
            "Evidence supports the hypothesis that "
            "scalar summaries discard useful "
            "decision-relevant spatial structure."
        )

    elif (
        auc_gain >= 0.01
        or
        ap_gain >= 0.02
    ):

        print(
            "Modest spatial-representation gain."
        )

        print(
            "Topology appears useful, but the current "
            "local patch / linear decoder is not yet "
            "a complete solution."
        )

    else:

        print(
            "Little spatial-representation gain."
        )

        print(
            "A 9x9 local belief patch with a linear "
            "decoder is not enough to explain the "
            "remaining inadequacy-prediction error."
        )

    print()

    print(
        f"Saved patch model: "
        f"{PATCH_MODEL_FILE}"
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