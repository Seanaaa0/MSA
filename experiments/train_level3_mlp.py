import csv
from pathlib import Path

import numpy as np

from src.level3_features import (
    FULL_FEATURES,
)

from src.learned_adequacy import (
    LogisticAdequacyModel,
)

from src.mlp_adequacy import (
    MLPAdequacyModel,
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


TRAIN_FILE = (
    RESULTS_DIR
    / "level3_train.csv"
)


VAL_FILE = (
    RESULTS_DIR
    / "level3_val.csv"
)


LOGISTIC_MODEL_FILE = (
    MODEL_DIR
    / "level3_logistic_full.npz"
)


MLP_MODEL_FILE = (
    MODEL_DIR
    / "level3_mlp_full.npz"
)


COMPARISON_FILE = (
    RESULTS_DIR
    / "level3_logistic_vs_mlp.csv"
)


MLP_SWEEP_FILE = (
    RESULTS_DIR
    / "level3_mlp_threshold_sweep.csv"
)


# ======================================================
# Dataset
# ======================================================

def load_dataset(
    path,
    feature_names,
):
    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        columns = set(
            reader.fieldnames
            or []
        )

        required = set(
            feature_names
        )

        required.add(
            "label_inadequate"
        )

        missing = (
            required
            - columns
        )

        if missing:

            raise ValueError(
                f"{path} missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            rows.append(
                row
            )

    x = np.asarray(
        [
            [
                float(
                    row[
                        feature
                    ]
                )
                for feature
                in feature_names
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    y = np.asarray(
        [
            int(
                row[
                    "label_inadequate"
                ]
            )
            for row in rows
        ],
        dtype=np.int64,
    )

    return (
        x,
        y,
    )


# ======================================================
# Metrics
# ======================================================

def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def classification_metrics(
    y_true,
    predictions,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    tp = int(
        np.sum(
            (y_true == 1)
            &
            (predictions == 1)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            &
            (predictions == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            &
            (predictions == 0)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            &
            (predictions == 0)
        )
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    if (
        precision + recall
        > 0.0
    ):

        f1 = (
            2.0
            * precision
            * recall
            /
            (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0

    return {
        "tp":
            tp,

        "fp":
            fp,

        "tn":
            tn,

        "fn":
            fn,

        "precision":
            precision,

        "recall":
            recall,

        "specificity":
            specificity,

        "f1":
            f1,
    }


def roc_auc(
    y_true,
    scores,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    positives = int(
        np.sum(
            y_true == 1
        )
    )

    negatives = int(
        np.sum(
            y_true == 0
        )
    )

    if (
        positives == 0
        or negatives == 0
    ):

        return float(
            "nan"
        )

    order = np.argsort(
        scores
    )

    sorted_scores = (
        scores[
            order
        ]
    )

    ranks = np.empty(
        len(scores),
        dtype=np.float64,
    )

    index = 0

    while (
        index
        < len(scores)
    ):

        end = (
            index + 1
        )

        while (
            end
            < len(scores)
            and
            sorted_scores[end]
            == sorted_scores[index]
        ):

            end += 1

        average_rank = (
            (
                index + 1
            )
            + end
        ) / 2.0

        ranks[
            order[
                index:end
            ]
        ] = average_rank

        index = end

    positive_rank_sum = (
        ranks[
            y_true == 1
        ].sum()
    )

    auc = (
        positive_rank_sum
        -
        positives
        * (
            positives + 1
        )
        / 2.0
    ) / (
        positives
        * negatives
    )

    return float(
        auc
    )


def average_precision(
    y_true,
    scores,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    positives = int(
        np.sum(
            y_true == 1
        )
    )

    if positives == 0:

        return float(
            "nan"
        )

    order = np.argsort(
        -scores
    )

    sorted_labels = (
        y_true[
            order
        ]
    )

    cumulative_tp = (
        np.cumsum(
            sorted_labels == 1
        )
    )

    cumulative_fp = (
        np.cumsum(
            sorted_labels == 0
        )
    )

    precision = (
        cumulative_tp
        /
        (
            cumulative_tp
            + cumulative_fp
        )
    )

    return float(
        precision[
            sorted_labels == 1
        ].mean()
    )


# ======================================================
# Threshold sweep
# ======================================================

def build_threshold_sweep(
    y_true,
    probabilities,
):
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray(
                    [
                        0.0,
                        1.0,
                    ]
                ),

                probabilities,
            ]
        )
    )

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities
            >= threshold
        ).astype(
            np.int64
        )

        metrics = (
            classification_metrics(
                y_true,
                predictions,
            )
        )

        rows.append(
            {
                "threshold":
                    float(
                        threshold
                    ),

                **metrics,
            }
        )

    return rows


def choose_for_recall(
    sweep_rows,
    target_recall,
):
    valid = [
        row
        for row in sweep_rows
        if (
            row[
                "recall"
            ]
            >= target_recall
        )
    ]

    if not valid:
        return None

    return max(
        valid,
        key=lambda row: (
            row[
                "precision"
            ],

            row[
                "f1"
            ],

            row[
                "threshold"
            ],
        ),
    )


def choose_best_f1(
    sweep_rows,
):
    return max(
        sweep_rows,
        key=lambda row: (
            row[
                "f1"
            ],

            row[
                "precision"
            ],

            row[
                "recall"
            ],
        ),
    )


# ======================================================
# Save helper
# ======================================================

def write_csv(
    path,
    rows,
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,

            fieldnames=
                list(
                    rows[0].keys()
                ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ======================================================
# Evaluation
# ======================================================

def evaluate_model(
    model_name,
    probabilities,
    y_val,
):
    auc = roc_auc(
        y_val,
        probabilities,
    )

    ap = average_precision(
        y_val,
        probabilities,
    )

    sweep = (
        build_threshold_sweep(
            y_true=
                y_val,

            probabilities=
                probabilities,
        )
    )

    rows = []

    for target_recall in [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]:

        operating = (
            choose_for_recall(
                sweep_rows=
                    sweep,

                target_recall=
                    target_recall,
            )
        )

        if operating is None:
            continue

        rows.append(
            {
                "model":
                    model_name,

                "operating_point":
                    (
                        "recall_at_least_"
                        f"{target_recall:.2f}"
                    ),

                "auroc":
                    auc,

                "average_precision":
                    ap,

                **operating,
            }
        )

    best_f1 = (
        choose_best_f1(
            sweep
        )
    )

    rows.append(
        {
            "model":
                model_name,

            "operating_point":
                "best_f1",

            "auroc":
                auc,

            "average_precision":
                ap,

            **best_f1,
        }
    )

    return (
        rows,
        sweep,
        auc,
        ap,
    )


# ======================================================
# Main
# ======================================================

def main():

    print(
        "=== MSA Level 3 "
        "Logistic vs MLP ==="
    )

    print()

    (
        x_train,
        y_train,
    ) = load_dataset(
        TRAIN_FILE,
        FULL_FEATURES,
    )

    (
        x_val,
        y_val,
    ) = load_dataset(
        VAL_FILE,
        FULL_FEATURES,
    )

    print(
        f"Train samples: "
        f"{len(y_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(y_val)}"
    )

    print(
        f"Train inadequate rate: "
        f"{y_train.mean():.3f}"
    )

    print(
        f"Validation inadequate rate: "
        f"{y_val.mean():.3f}"
    )

    print()

    # ==================================================
    # Existing Logistic baseline
    # ==================================================

    if not LOGISTIC_MODEL_FILE.exists():

        raise FileNotFoundError(
            "Logistic model not found.\n"
            "Run:\n"
            "python -m experiments.train_level3"
        )

    logistic = (
        LogisticAdequacyModel
        .load(
            LOGISTIC_MODEL_FILE
        )
    )

    logistic_scores = (
        logistic.predict_proba(
            x_val
        )
    )

    # ==================================================
    # Train nonlinear MLP
    # ==================================================

    print(
        "Training MLP:"
    )

    print(
        "  architecture = "
        "16 -> 32 -> 16 -> 1"
    )

    print(
        "  optimizer = Adam"
    )

    print(
        "  weighted BCE = yes"
    )

    print(
        "  validation used for training = NO"
    )

    print()

    mlp = MLPAdequacyModel(
        feature_names=
            FULL_FEATURES,

        hidden_1=32,

        hidden_2=16,

        threshold=0.5,

        seed=42,
    )

    train_info = (
        mlp.fit(
            x=x_train,
            y=y_train,

            epochs=800,

            learning_rate=
                0.003,

            l2=1e-4,

            positive_weight=None,

            patience=100,

            verbose=True,
        )
    )

    print()

    print(
        "=== MLP Training Complete ==="
    )

    print(
        f"epochs: "
        f"{train_info['epochs']}"
    )

    print(
        f"best training loss: "
        f"{train_info['best_loss']:.6f}"
    )

    print(
        f"positive weight: "
        f"{train_info['positive_weight']:.3f}"
    )

    print()

    mlp_scores = (
        mlp.predict_proba(
            x_val
        )
    )

    # ==================================================
    # Evaluate both
    # ==================================================

    (
        logistic_rows,
        logistic_sweep,
        logistic_auc,
        logistic_ap,
    ) = evaluate_model(
        model_name=
            "logistic",

        probabilities=
            logistic_scores,

        y_val=
            y_val,
    )

    (
        mlp_rows,
        mlp_sweep,
        mlp_auc,
        mlp_ap,
    ) = evaluate_model(
        model_name=
            "mlp",

        probabilities=
            mlp_scores,

        y_val=
            y_val,
    )

    # ==================================================
    # Save a canonical MLP threshold
    #
    # Keep same current research preference:
    # recall >= 0.90
    # ==================================================

    mlp_r90 = (
        choose_for_recall(
            sweep_rows=
                mlp_sweep,

            target_recall=
                0.90,
        )
    )

    if mlp_r90 is not None:

        mlp.threshold = (
            mlp_r90[
                "threshold"
            ]
        )

    mlp.save(
        MLP_MODEL_FILE
    )

    # ==================================================
    # Save files
    # ==================================================

    all_rows = (
        logistic_rows
        + mlp_rows
    )

    write_csv(
        COMPARISON_FILE,
        all_rows,
    )

    mlp_sweep_rows = [
        {
            "model":
                "mlp",

            **row,
        }
        for row in mlp_sweep
    ]

    write_csv(
        MLP_SWEEP_FILE,
        mlp_sweep_rows,
    )

    # ==================================================
    # Report
    # ==================================================

    print(
        "=== Threshold-independent ==="
    )

    print(
        "model       AUROC      AP"
    )

    print(
        "-" * 30
    )

    print(
        f"{'Logistic':<10}"
        f"{logistic_auc:>7.3f} "
        f"{logistic_ap:>7.3f}"
    )

    print(
        f"{'MLP':<10}"
        f"{mlp_auc:>7.3f} "
        f"{mlp_ap:>7.3f}"
    )

    print()

    print(
        "Delta:"
    )

    print(
        f"  AUROC: "
        f"{mlp_auc - logistic_auc:+.3f}"
    )

    print(
        f"  AP:    "
        f"{mlp_ap - logistic_ap:+.3f}"
    )

    print()

    # ==================================================
    # Operating-point comparison
    # ==================================================

    print(
        "=== High-Recall Comparison ==="
    )

    print(
        "target  model      "
        "threshold   P       R       "
        "F1      FP    FN"
    )

    print(
        "-" * 74
    )

    for target in [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]:

        logistic_point = (
            choose_for_recall(
                logistic_sweep,
                target,
            )
        )

        mlp_point = (
            choose_for_recall(
                mlp_sweep,
                target,
            )
        )

        for (
            name,
            point,
        ) in [
            (
                "Logistic",
                logistic_point,
            ),
            (
                "MLP",
                mlp_point,
            ),
        ]:

            if point is None:
                continue

            print(
                f"{target:>6.2f} "
                f"{name:<10}"
                f"{point['threshold']:>9.4f} "
                f"{point['precision']:>7.3f} "
                f"{point['recall']:>7.3f} "
                f"{point['f1']:>7.3f} "
                f"{point['fp']:>5} "
                f"{point['fn']:>5}"
            )

        print()

    # ==================================================
    # Best F1
    # ==================================================

    logistic_best = (
        choose_best_f1(
            logistic_sweep
        )
    )

    mlp_best = (
        choose_best_f1(
            mlp_sweep
        )
    )

    print(
        "=== Best F1 ==="
    )

    print(
        "model       threshold   "
        "P       R       F1      "
        "FP    FN"
    )

    print(
        "-" * 62
    )

    for (
        name,
        point,
    ) in [
        (
            "Logistic",
            logistic_best,
        ),
        (
            "MLP",
            mlp_best,
        ),
    ]:

        print(
            f"{name:<10}"
            f"{point['threshold']:>9.4f} "
            f"{point['precision']:>7.3f} "
            f"{point['recall']:>7.3f} "
            f"{point['f1']:>7.3f} "
            f"{point['fp']:>5} "
            f"{point['fn']:>5}"
        )

    print()

    # ==================================================
    # Research interpretation helper
    # ==================================================

    print(
        "=== Capacity Diagnostic ==="
    )

    auc_gain = (
        mlp_auc
        - logistic_auc
    )

    ap_gain = (
        mlp_ap
        - logistic_ap
    )

    if (
        auc_gain >= 0.03
        or
        ap_gain >= 0.05
    ):

        print(
            "MLP shows a meaningful nonlinear gain."
        )

        print(
            "This suggests classifier capacity / "
            "feature interactions matter."
        )

    elif (
        auc_gain >= 0.01
        or
        ap_gain >= 0.02
    ):

        print(
            "MLP shows a modest gain."
        )

        print(
            "Nonlinearity helps, but representation "
            "may still be the main limitation."
        )

    else:

        print(
            "MLP shows little improvement."
        )

        print(
            "This suggests the main bottleneck is "
            "probably representation rather than "
            "linear classifier capacity."
        )

    print()

    print(
        f"Saved MLP model: "
        f"{MLP_MODEL_FILE}"
    )

    print(
        f"Saved comparison: "
        f"{COMPARISON_FILE}"
    )

    print(
        f"Saved MLP sweep: "
        f"{MLP_SWEEP_FILE}"
    )

    print()

    print(
        "TEST split was NOT used."
    )


if __name__ == "__main__":
    main()