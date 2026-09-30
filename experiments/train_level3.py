import argparse
import csv
from pathlib import Path

import numpy as np

from src.learned_adequacy import (
    LogisticAdequacyModel,
)

from src.level3_features import (
    UNCERTAINTY_FEATURES,
    HISTORY_FEATURES,
    CONTEXT_FEATURES,
    FULL_FEATURES,
)


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


OUTPUT_FILE = (
    RESULTS_DIR
    / "level3_logistic_ablation.csv"
)


ABLATIONS = {
    "uncertainty_only":
        list(
            UNCERTAINTY_FEATURES
        ),

    "uncertainty_history":
        list(
            UNCERTAINTY_FEATURES
            + HISTORY_FEATURES
        ),

    "uncertainty_context":
        list(
            UNCERTAINTY_FEATURES
            + CONTEXT_FEATURES
        ),

    "full":
        list(
            FULL_FEATURES
        ),
}


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

        missing = [
            name
            for name
            in feature_names
            if name not in columns
        ]

        if missing:
            raise ValueError(
                f"{path} is missing "
                f"features: {missing}"
            )

        if (
            "label_inadequate"
            not in columns
        ):
            raise ValueError(
                f"{path} has no "
                "label_inadequate column."
            )

        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No rows in {path}"
        )

    x = np.asarray(
        [
            [
                float(
                    row[name]
                )
                for name
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

def confusion_counts(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    tp = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 1)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 0)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 0)
        )
    )

    return (
        tp,
        fp,
        tn,
        fn,
    )


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
    y_pred,
):
    (
        tp,
        fp,
        tn,
        fn,
    ) = confusion_counts(
        y_true,
        y_pred,
    )

    accuracy = safe_divide(
        tp + tn,
        tp + fp + tn + fn,
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

    f1 = safe_divide(
        2.0
        * precision
        * recall,

        precision
        + recall,
    )

    balanced_accuracy = (
        recall
        + specificity
    ) / 2.0

    return {
        "tp":
            tp,

        "fp":
            fp,

        "tn":
            tn,

        "fn":
            fn,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "specificity":
            specificity,

        "balanced_accuracy":
            balanced_accuracy,

        "f1":
            f1,
    }


def roc_auc(
    y_true,
    probabilities,
):
    """
    Rank-based AUROC.

    Continuous logistic probabilities should make
    exact ties rare. Average tie handling is included.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    positive_count = int(
        np.sum(
            y_true == 1
        )
    )

    negative_count = int(
        np.sum(
            y_true == 0
        )
    )

    if (
        positive_count == 0
        or negative_count == 0
    ):
        return float("nan")

    order = np.argsort(
        probabilities
    )

    sorted_probs = (
        probabilities[order]
    )

    ranks = np.empty(
        len(probabilities),
        dtype=np.float64,
    )

    index = 0

    while index < len(
        sorted_probs
    ):

        end = index + 1

        while (
            end
            < len(sorted_probs)
            and
            sorted_probs[end]
            == sorted_probs[index]
        ):
            end += 1

        # Rank convention starts at 1.
        average_rank = (
            (
                index + 1
            )
            +
            end
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
        positive_count
        * (
            positive_count
            + 1
        )
        / 2.0
    ) / (
        positive_count
        * negative_count
    )

    return float(
        auc
    )


# ======================================================
# Threshold selection
# ======================================================

def choose_threshold(
    y_true,
    probabilities,
    target_recall=0.90,
):
    """
    Select an operating threshold on VALIDATION only.

    Primary requirement:
        recall >= target_recall

    Within that constraint:
        maximize precision,
        then F1.

    This reflects the MSA preference that
    false negatives are expensive, while still
    avoiding the trivial "query everything" policy.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    candidate_thresholds = (
        np.unique(
            np.concatenate(
                [
                    probabilities,
                    np.asarray(
                        [
                            0.0,
                            1.0,
                        ]
                    ),
                ]
            )
        )
    )

    best = None

    for threshold in (
        candidate_thresholds
    ):

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

        candidate = {
            "threshold":
                float(threshold),

            **metrics,
        }

        reaches_target = (
            metrics["recall"]
            >= target_recall
        )

        if not reaches_target:
            continue

        if best is None:

            best = candidate
            continue

        candidate_score = (
            candidate[
                "precision"
            ],

            candidate[
                "f1"
            ],

            candidate[
                "threshold"
            ],
        )

        best_score = (
            best[
                "precision"
            ],

            best[
                "f1"
            ],

            best[
                "threshold"
            ],
        )

        if (
            candidate_score
            > best_score
        ):
            best = candidate

    # --------------------------------------------------
    # Fallback:
    # If target recall cannot be reached,
    # maximize recall first.
    # --------------------------------------------------

    if best is None:

        for threshold in (
            candidate_thresholds
        ):

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

            candidate = {
                "threshold":
                    float(
                        threshold
                    ),

                **metrics,
            }

            if best is None:

                best = candidate
                continue

            candidate_score = (
                candidate[
                    "recall"
                ],

                candidate[
                    "precision"
                ],

                candidate[
                    "f1"
                ],
            )

            best_score = (
                best[
                    "recall"
                ],

                best[
                    "precision"
                ],

                best[
                    "f1"
                ],
            )

            if (
                candidate_score
                > best_score
            ):
                best = candidate

    return best


# ======================================================
# Training
# ======================================================

def train_ablation(
    name,
    feature_names,
    target_recall,
    epochs,
):

    (
        x_train,
        y_train,
    ) = load_dataset(
        TRAIN_FILE,
        feature_names,
    )

    (
        x_val,
        y_val,
    ) = load_dataset(
        VAL_FILE,
        feature_names,
    )

    model = (
        LogisticAdequacyModel(
            feature_names=
                feature_names
        )
    )

    train_info = (
        model.fit(
            x=x_train,
            y=y_train,

            epochs=epochs,

            learning_rate=0.05,

            l2=1e-4,

            positive_weight=None,

            verbose=False,
        )
    )

    train_probabilities = (
        model.predict_proba(
            x_train
        )
    )

    val_probabilities = (
        model.predict_proba(
            x_val
        )
    )

    threshold_result = (
        choose_threshold(
            y_true=y_val,

            probabilities=
                val_probabilities,

            target_recall=
                target_recall,
        )
    )

    threshold = (
        threshold_result[
            "threshold"
        ]
    )

    model.threshold = (
        threshold
    )

    # ----------------------------------------------
    # Final metrics at selected threshold
    # ----------------------------------------------

    train_predictions = (
        train_probabilities
        >= threshold
    ).astype(
        np.int64
    )

    val_predictions = (
        val_probabilities
        >= threshold
    ).astype(
        np.int64
    )

    train_metrics = (
        classification_metrics(
            y_train,
            train_predictions,
        )
    )

    val_metrics = (
        classification_metrics(
            y_val,
            val_predictions,
        )
    )

    train_auc = roc_auc(
        y_train,
        train_probabilities,
    )

    val_auc = roc_auc(
        y_val,
        val_probabilities,
    )

    # ----------------------------------------------
    # Save learned model
    # ----------------------------------------------

    model_path = (
        MODEL_DIR
        /
        f"level3_logistic_{name}.npz"
    )

    model.save(
        model_path
    )

    return {
        "name":
            name,

        "features":
            len(
                feature_names
            ),

        "threshold":
            threshold,

        "epochs":
            train_info[
                "epochs"
            ],

        "positive_weight":
            train_info[
                "positive_weight"
            ],

        "train_auc":
            train_auc,

        "train_precision":
            train_metrics[
                "precision"
            ],

        "train_recall":
            train_metrics[
                "recall"
            ],

        "train_f1":
            train_metrics[
                "f1"
            ],

        "val_auc":
            val_auc,

        "val_accuracy":
            val_metrics[
                "accuracy"
            ],

        "val_precision":
            val_metrics[
                "precision"
            ],

        "val_recall":
            val_metrics[
                "recall"
            ],

        "val_f1":
            val_metrics[
                "f1"
            ],

        "val_tp":
            val_metrics[
                "tp"
            ],

        "val_fp":
            val_metrics[
                "fp"
            ],

        "val_tn":
            val_metrics[
                "tn"
            ],

        "val_fn":
            val_metrics[
                "fn"
            ],

        "model_path":
            str(
                model_path
            ),
    }


# ======================================================
# Main
# ======================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.90,

        help=(
            "Minimum validation recall "
            "used when selecting threshold."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1500,
    )

    args = parser.parse_args()

    if not (
        0.0
        < args.target_recall
        <= 1.0
    ):
        raise ValueError(
            "target-recall must be "
            "in (0, 1]."
        )

    print(
        "=== MSA Level 3 "
        "Learned Adequacy Baseline ==="
    )

    print(
        f"Target validation recall: "
        f"{args.target_recall:.2f}"
    )

    print()

    # --------------------------------------------------
    # Dataset summary
    # --------------------------------------------------

    (
        _,
        train_labels,
    ) = load_dataset(
        TRAIN_FILE,
        FULL_FEATURES,
    )

    (
        _,
        val_labels,
    ) = load_dataset(
        VAL_FILE,
        FULL_FEATURES,
    )

    print(
        f"Train samples: "
        f"{len(train_labels)} "
        f"(inadequate="
        f"{train_labels.mean():.3f})"
    )

    print(
        f"Val samples:   "
        f"{len(val_labels)} "
        f"(inadequate="
        f"{val_labels.mean():.3f})"
    )

    naive_accuracy = (
        1.0
        - val_labels.mean()
    )

    print(
        f"Always-adequate "
        f"validation accuracy: "
        f"{naive_accuracy:.3f}"
    )

    print()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for (
        name,
        feature_names,
    ) in ABLATIONS.items():

        print(
            f"Training: {name} "
            f"({len(feature_names)} features)"
        )

        result = (
            train_ablation(
                name=name,

                feature_names=
                    feature_names,

                target_recall=
                    args.target_recall,

                epochs=
                    args.epochs,
            )
        )

        results.append(
            result
        )

        print(
            f"  threshold="
            f"{result['threshold']:.3f}"
        )

        print(
            f"  val AUC="
            f"{result['val_auc']:.3f}"
        )

        print(
            f"  P="
            f"{result['val_precision']:.3f} "
            f"R="
            f"{result['val_recall']:.3f} "
            f"F1="
            f"{result['val_f1']:.3f}"
        )

        print(
            f"  TP="
            f"{result['val_tp']} "
            f"FP="
            f"{result['val_fp']} "
            f"FN="
            f"{result['val_fn']} "
            f"TN="
            f"{result['val_tn']}"
        )

        print()

    # ==================================================
    # Save ablation table
    # ==================================================

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                list(
                    results[0].keys()
                ),
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ==================================================
    # Compact comparison
    # ==================================================

    print(
        "=== Validation Ablation ==="
    )

    print(
        "name                    "
        "feat   AUC     P       R       F1      FN"
    )

    print(
        "-" * 70
    )

    for result in results:

        print(
            f"{result['name']:<24}"
            f"{result['features']:>4} "
            f"{result['val_auc']:>7.3f} "
            f"{result['val_precision']:>7.3f} "
            f"{result['val_recall']:>7.3f} "
            f"{result['val_f1']:>7.3f} "
            f"{result['val_fn']:>7}"
        )

    print()

    print(
        f"Saved ablation results to: "
        f"{OUTPUT_FILE}"
    )

    print()

    print(
        "TEST split has NOT been used."
    )


if __name__ == "__main__":
    main()