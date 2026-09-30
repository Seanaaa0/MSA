
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.learned_adequacy import (
    LogisticAdequacyModel,
)


# ======================================================
# Paths
# ======================================================

RESULTS_DIR = Path(
    "results"
)


DEFAULT_MODEL = Path(
    "models"
) / "level3_logistic_full.npz"


DEFAULT_VAL_FILE = (
    RESULTS_DIR
    / "level3_val.csv"
)


THRESHOLD_SWEEP_FILE = (
    RESULTS_DIR
    / "level3_threshold_sweep.csv"
)


OPERATING_POINTS_FILE = (
    RESULTS_DIR
    / "level3_operating_points.csv"
)


SCORE_BINS_FILE = (
    RESULTS_DIR
    / "level3_score_bins.csv"
)


SEED_BREAKDOWN_FILE = (
    RESULTS_DIR
    / "level3_seed_breakdown.csv"
)


PR_PLOT_FILE = (
    RESULTS_DIR
    / "level3_precision_recall.png"
)


FP_FN_PLOT_FILE = (
    RESULTS_DIR
    / "level3_fp_fn_tradeoff.png"
)


# ======================================================
# Basic helpers
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


def format_float(
    value,
    digits=3,
):
    if value is None:
        return "N/A"

    if np.isnan(value):
        return "N/A"

    return (
        f"{value:.{digits}f}"
    )


# ======================================================
# Dataset loading
# ======================================================

def load_validation_dataset(
    path,
    feature_names,
):
    """
    Load validation CSV.

    Only the model's saved feature_names are used
    as model inputs.

    map_seed is retained only for experiment-side
    breakdown analysis.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Validation dataset not found: "
            f"{path}"
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
                "Validation dataset missing "
                f"columns: {sorted(missing)}"
            )

        for row in reader:
            rows.append(
                row
            )

    if not rows:

        raise RuntimeError(
            "Validation dataset is empty."
        )

    # --------------------------------------------------
    # Model input
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Ground-truth label
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Experiment-side grouping metadata
    # --------------------------------------------------

    if (
        "map_seed"
        in rows[0]
    ):

        seeds = np.asarray(
            [
                int(
                    row[
                        "map_seed"
                    ]
                )
                for row in rows
            ],
            dtype=np.int64,
        )

    else:

        seeds = np.zeros(
            len(rows),
            dtype=np.int64,
        )

    return (
        x,
        y,
        seeds,
    )


# ======================================================
# Confusion / metrics
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


def classification_metrics(
    y_true,
    predictions,
):
    (
        tp,
        fp,
        tn,
        fn,
    ) = confusion_counts(
        y_true,
        predictions,
    )

    total = (
        tp
        + fp
        + tn
        + fn
    )

    accuracy = safe_divide(
        tp + tn,
        total,
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

    false_positive_rate = (
        safe_divide(
            fp,
            fp + tn,
        )
    )

    false_negative_rate = (
        safe_divide(
            fn,
            fn + tp,
        )
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

    balanced_accuracy = (
        recall
        + specificity
    ) / 2.0

    predicted_positive_rate = (
        safe_divide(
            tp + fp,
            total,
        )
    )

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

        "fpr":
            false_positive_rate,

        "fnr":
            false_negative_rate,

        "f1":
            f1,

        "balanced_accuracy":
            balanced_accuracy,

        "predicted_positive_rate":
            predicted_positive_rate,
    }


def metrics_at_threshold(
    y_true,
    probabilities,
    threshold,
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

    return {
        "threshold":
            float(
                threshold
            ),

        **metrics,
    }


# ======================================================
# Threshold-independent metrics
# ======================================================

def roc_auc(
    y_true,
    scores,
):
    """
    Rank-based AUROC.

    Does not depend on a classification threshold.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
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

        # ranks are 1-based
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


def average_precision(
    y_true,
    scores,
):
    """
    Average Precision (AP).

    A threshold-independent summary of the
    precision-recall ranking.

    Particularly useful for imbalanced data.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    positive_count = int(
        np.sum(
            y_true == 1
        )
    )

    if positive_count == 0:
        return float("nan")

    order = np.argsort(
        -scores
    )

    sorted_y = (
        y_true[
            order
        ]
    )

    cumulative_tp = (
        np.cumsum(
            sorted_y
            == 1
        )
    )

    cumulative_fp = (
        np.cumsum(
            sorted_y
            == 0
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

    # AP:
    # mean precision at every positive-ranked sample
    ap = (
        precision[
            sorted_y == 1
        ].mean()
    )

    return float(
        ap
    )


# ======================================================
# Threshold sweep
# ======================================================

def build_threshold_sweep(
    y_true,
    probabilities,
):
    """
    Evaluate every threshold where the classifier's
    predictions could actually change.

    More precise than only checking:
        0.1, 0.2, 0.3, ...
    """

    unique_probabilities = (
        np.unique(
            probabilities
        )
    )

    thresholds = (
        np.unique(
            np.concatenate(
                [
                    np.asarray(
                        [
                            0.0,
                            1.0,
                        ],
                        dtype=np.float64,
                    ),

                    unique_probabilities,
                ]
            )
        )
    )

    thresholds = np.sort(
        thresholds
    )

    rows = []

    for threshold in thresholds:

        rows.append(
            metrics_at_threshold(
                y_true=
                    y_true,

                probabilities=
                    probabilities,

                threshold=
                    threshold,
            )
        )

    return rows


# ======================================================
# Operating point selection
# ======================================================

def choose_for_target_recall(
    sweep_rows,
    target_recall,
):
    """
    Among thresholds satisfying recall >= target,
    choose:

        1. highest precision
        2. highest F1
        3. highest threshold

    This prevents the trivial threshold=0 solution
    from being treated as useful.
    """

    valid = [
        row
        for row in sweep_rows
        if (
            row["recall"]
            >= target_recall
        )
    ]

    if not valid:
        return None

    return max(
        valid,
        key=lambda row: (
            row["precision"],
            row["f1"],
            row["threshold"],
        ),
    )


def choose_best_metric(
    sweep_rows,
    metric_name,
):
    return max(
        sweep_rows,
        key=lambda row: (
            row[
                metric_name
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
# Score distribution
# ======================================================

def score_distribution_summary(
    y_true,
    probabilities,
):
    adequate_scores = (
        probabilities[
            y_true == 0
        ]
    )

    inadequate_scores = (
        probabilities[
            y_true == 1
        ]
    )

    quantiles = [
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
    ]

    result = {}

    for (
        name,
        values,
    ) in [
        (
            "adequate",
            adequate_scores,
        ),
        (
            "inadequate",
            inadequate_scores,
        ),
    ]:

        result[name] = {
            "mean":
                float(
                    np.mean(
                        values
                    )
                ),

            "std":
                float(
                    np.std(
                        values
                    )
                ),
        }

        for quantile in quantiles:

            key = (
                f"q"
                f"{int(quantile * 100):02d}"
            )

            result[name][key] = (
                float(
                    np.quantile(
                        values,
                        quantile,
                    )
                )
            )

    return result


# ======================================================
# Score bins
# ======================================================

def build_score_bins(
    y_true,
    probabilities,
    bin_count=10,
):
    """
    Equal-count score bins.

    This is NOT calibration training.

    It only checks whether higher model scores
    correspond to higher empirical inadequacy rates.

    Because the model was trained with class weighting,
    its score should currently be treated as a risk score,
    not automatically as calibrated probability.
    """

    order = np.argsort(
        probabilities
    )

    groups = np.array_split(
        order,
        bin_count,
    )

    rows = []

    for index, group in enumerate(
        groups,
        start=1,
    ):

        if len(group) == 0:
            continue

        scores = (
            probabilities[
                group
            ]
        )

        labels = (
            y_true[
                group
            ]
        )

        rows.append(
            {
                "bin":
                    index,

                "count":
                    len(group),

                "score_min":
                    float(
                        np.min(
                            scores
                        )
                    ),

                "score_max":
                    float(
                        np.max(
                            scores
                        )
                    ),

                "score_mean":
                    float(
                        np.mean(
                            scores
                        )
                    ),

                "observed_inadequate_rate":
                    float(
                        np.mean(
                            labels
                        )
                    ),
            }
        )

    return rows


# ======================================================
# Per-seed analysis
# ======================================================

def seed_breakdown(
    y_true,
    probabilities,
    seeds,
    operating_points,
):
    """
    Inspect whether aggregate performance is hiding
    bad behavior on particular generated maps.
    """

    unique_seeds = (
        np.unique(
            seeds
        )
    )

    rows = []

    for (
        operating_name,
        threshold,
    ) in operating_points:

        for seed in unique_seeds:

            mask = (
                seeds
                == seed
            )

            seed_y = (
                y_true[
                    mask
                ]
            )

            seed_scores = (
                probabilities[
                    mask
                ]
            )

            predictions = (
                seed_scores
                >= threshold
            ).astype(
                np.int64
            )

            metrics = (
                classification_metrics(
                    seed_y,
                    predictions,
                )
            )

            positive_count = int(
                np.sum(
                    seed_y == 1
                )
            )

            rows.append(
                {
                    "operating_point":
                        operating_name,

                    "threshold":
                        threshold,

                    "map_seed":
                        int(seed),

                    "samples":
                        len(
                            seed_y
                        ),

                    "inadequate_count":
                        positive_count,

                    **metrics,
                }
            )

    return rows


# ======================================================
# CSV output
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
# Plotting
# ======================================================

def create_precision_recall_plot(
    sweep_rows,
    prevalence,
):
    recall = np.asarray(
        [
            row["recall"]
            for row in sweep_rows
        ]
    )

    precision = np.asarray(
        [
            row["precision"]
            for row in sweep_rows
        ]
    )

    order = np.argsort(
        recall
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        recall[order],
        precision[order],
    )

    plt.axhline(
        prevalence,
        linestyle="--",
        label=(
            "Random / prevalence "
            f"({prevalence:.3f})"
        ),
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "MSA Level 3 Validation "
        "Precision–Recall Trade-off"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PR_PLOT_FILE,
        dpi=160,
    )

    plt.close()


def create_fp_fn_plot(
    sweep_rows,
):
    thresholds = np.asarray(
        [
            row["threshold"]
            for row in sweep_rows
        ]
    )

    fp = np.asarray(
        [
            row["fp"]
            for row in sweep_rows
        ]
    )

    fn = np.asarray(
        [
            row["fn"]
            for row in sweep_rows
        ]
    )

    order = np.argsort(
        thresholds
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        thresholds[order],
        fp[order],
        label="False Positives",
    )

    plt.plot(
        thresholds[order],
        fn[order],
        label="False Negatives",
    )

    plt.xlabel(
        "Threshold"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "MSA Level 3 Validation "
        "FP–FN Trade-off"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FP_FN_PLOT_FILE,
        dpi=160,
    )

    plt.close()


# ======================================================
# Console reporting
# ======================================================

def print_metrics(
    title,
    row,
):

    print(
        title
    )

    print(
        f"  threshold:             "
        f"{row['threshold']:.4f}"
    )

    print(
        f"  precision:             "
        f"{row['precision']:.3f}"
    )

    print(
        f"  recall:                "
        f"{row['recall']:.3f}"
    )

    print(
        f"  F1:                    "
        f"{row['f1']:.3f}"
    )

    print(
        f"  specificity:           "
        f"{row['specificity']:.3f}"
    )

    print(
        f"  balanced accuracy:     "
        f"{row['balanced_accuracy']:.3f}"
    )

    print(
        f"  predicted positive:    "
        f"{row['predicted_positive_rate']:.3f}"
    )

    print(
        f"  TP={row['tp']} "
        f"FP={row['fp']} "
        f"FN={row['fn']} "
        f"TN={row['tn']}"
    )

    print()


def print_seed_worst_cases(
    seed_rows,
    operating_name,
    top_k=10,
):
    selected = [
        row
        for row in seed_rows
        if (
            row[
                "operating_point"
            ]
            == operating_name
        )
    ]

    if not selected:
        return

    # Maps with positives first,
    # ranked by false negatives.
    selected = [
        row
        for row in selected
        if (
            row[
                "inadequate_count"
            ]
            > 0
        )
    ]

    selected.sort(
        key=lambda row: (
            row["fn"],
            row["fp"],
        ),
        reverse=True,
    )

    print(
        f"=== Worst seeds: "
        f"{operating_name} ==="
    )

    print(
        "seed    samples  bad   "
        "TP   FP   FN   TN"
    )

    print(
        "-" * 48
    )

    for row in (
        selected[
            :top_k
        ]
    ):

        print(
            f"{row['map_seed']:>6} "
            f"{row['samples']:>8} "
            f"{row['inadequate_count']:>4} "
            f"{row['tp']:>4} "
            f"{row['fp']:>4} "
            f"{row['fn']:>4} "
            f"{row['tn']:>4}"
        )

    print()


# ======================================================
# Main
# ======================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=
            DEFAULT_MODEL,
    )

    parser.add_argument(
        "--val-file",
        type=Path,
        default=
            DEFAULT_VAL_FILE,
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
    )

    args = parser.parse_args()

    # ==================================================
    # Load model
    # ==================================================

    if not args.model.exists():

        raise FileNotFoundError(
            f"Model not found: "
            f"{args.model}\n"
            "Run experiments.train_level3 first."
        )

    model = (
        LogisticAdequacyModel
        .load(
            args.model
        )
    )

    # ==================================================
    # Load validation data
    # ==================================================

    (
        x_val,
        y_val,
        seeds,
    ) = (
        load_validation_dataset(
            path=
                args.val_file,

            feature_names=
                model.feature_names,
        )
    )

    probabilities = (
        model.predict_proba(
            x_val
        )
    )

    prevalence = float(
        np.mean(
            y_val
        )
    )

    adequate_count = int(
        np.sum(
            y_val == 0
        )
    )

    inadequate_count = int(
        np.sum(
            y_val == 1
        )
    )

    # ==================================================
    # Threshold-independent evaluation
    # ==================================================

    auc = roc_auc(
        y_val,
        probabilities,
    )

    ap = average_precision(
        y_val,
        probabilities,
    )

    # ==================================================
    # Sweep all thresholds
    # ==================================================

    sweep_rows = (
        build_threshold_sweep(
            y_true=
                y_val,

            probabilities=
                probabilities,
        )
    )

    # ==================================================
    # Existing saved threshold
    # ==================================================

    saved_threshold_row = (
        metrics_at_threshold(
            y_true=
                y_val,

            probabilities=
                probabilities,

            threshold=
                model.threshold,
        )
    )

    # ==================================================
    # Target-recall operating points
    # ==================================================

    target_recalls = [
        0.99,
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]

    operating_rows = []

    for target in (
        target_recalls
    ):

        result = (
            choose_for_target_recall(
                sweep_rows=
                    sweep_rows,

                target_recall=
                    target,
            )
        )

        if result is None:
            continue

        operating_rows.append(
            {
                "name":
                    (
                        "recall_at_least_"
                        f"{target:.2f}"
                    ),

                "target_recall":
                    target,

                **result,
            }
        )

    # ==================================================
    # Other useful operating points
    # ==================================================

    best_f1 = (
        choose_best_metric(
            sweep_rows,
            "f1",
        )
    )

    best_balanced = (
        choose_best_metric(
            sweep_rows,
            "balanced_accuracy",
        )
    )

    operating_rows.append(
        {
            "name":
                "best_f1",

            "target_recall":
                "",

            **best_f1,
        }
    )

    operating_rows.append(
        {
            "name":
                "best_balanced_accuracy",

            "target_recall":
                "",

            **best_balanced,
        }
    )

    operating_rows.append(
        {
            "name":
                "saved_model_threshold",

            "target_recall":
                "",

            **saved_threshold_row,
        }
    )

    # ==================================================
    # Score distribution
    # ==================================================

    distributions = (
        score_distribution_summary(
            y_true=
                y_val,

            probabilities=
                probabilities,
        )
    )

    # ==================================================
    # Score bins
    # ==================================================

    score_bins = (
        build_score_bins(
            y_true=
                y_val,

            probabilities=
                probabilities,

            bin_count=10,
        )
    )

    # ==================================================
    # Per-map analysis
    # ==================================================

    selected_operating_points = [
        (
            "saved_model_threshold",
            model.threshold,
        ),
        (
            "best_f1",
            best_f1[
                "threshold"
            ],
        ),
    ]

    for target in [
        0.95,
        0.90,
        0.85,
        0.80,
    ]:

        result = (
            choose_for_target_recall(
                sweep_rows=
                    sweep_rows,

                target_recall=
                    target,
            )
        )

        if result is not None:

            selected_operating_points.append(
                (
                    (
                        "recall_at_least_"
                        f"{target:.2f}"
                    ),

                    result[
                        "threshold"
                    ],
                )
            )

    seed_rows = (
        seed_breakdown(
            y_true=
                y_val,

            probabilities=
                probabilities,

            seeds=
                seeds,

            operating_points=
                selected_operating_points,
        )
    )

    # ==================================================
    # Save outputs
    # ==================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        THRESHOLD_SWEEP_FILE,
        sweep_rows,
    )

    write_csv(
        OPERATING_POINTS_FILE,
        operating_rows,
    )

    write_csv(
        SCORE_BINS_FILE,
        score_bins,
    )

    write_csv(
        SEED_BREAKDOWN_FILE,
        seed_rows,
    )

    if not args.no_plots:

        create_precision_recall_plot(
            sweep_rows=
                sweep_rows,

            prevalence=
                prevalence,
        )

        create_fp_fn_plot(
            sweep_rows=
                sweep_rows,
        )

    # ==================================================
    # Console summary
    # ==================================================

    print(
        "=== MSA Level 3 "
        "Threshold Analysis ==="
    )

    print()

    print(
        f"Model:             "
        f"{args.model}"
    )

    print(
        f"Validation file:   "
        f"{args.val_file}"
    )

    print()

    print(
        f"Samples:           "
        f"{len(y_val)}"
    )

    print(
        f"Adequate:          "
        f"{adequate_count}"
    )

    print(
        f"Inadequate:        "
        f"{inadequate_count}"
    )

    print(
        f"Inadequate rate:   "
        f"{prevalence:.3f}"
    )

    print()

    print(
        "=== Threshold-independent ==="
    )

    print(
        f"AUROC:             "
        f"{auc:.3f}"
    )

    print(
        f"Average Precision: "
        f"{ap:.3f}"
    )

    print(
        f"Random AP baseline:"
        f" {prevalence:.3f}"
    )

    print()

    # ==================================================
    # Score distributions
    # ==================================================

    print(
        "=== Score Distribution ==="
    )

    print(
        "NOTE: score is currently a risk score, "
        "not guaranteed calibrated probability."
    )

    print()

    for class_name in [
        "adequate",
        "inadequate",
    ]:

        data = (
            distributions[
                class_name
            ]
        )

        print(
            f"{class_name:<12} "
            f"mean={data['mean']:.3f} "
            f"q10={data['q10']:.3f} "
            f"q25={data['q25']:.3f} "
            f"q50={data['q50']:.3f} "
            f"q75={data['q75']:.3f} "
            f"q90={data['q90']:.3f}"
        )

    print()

    # ==================================================
    # Existing operating point
    # ==================================================

    print_metrics(
        title=
            "=== Saved model threshold ===",

        row=
            saved_threshold_row,
    )

    # ==================================================
    # Recall trade-off
    # ==================================================

    print(
        "=== Recall-constrained "
        "Operating Points ==="
    )

    print(
        "target   threshold   "
        "precision  recall    F1     "
        "FP    FN   pred+"
    )

    print(
        "-" * 76
    )

    for row in operating_rows:

        if not str(
            row["name"]
        ).startswith(
            "recall_at_least_"
        ):
            continue

        print(
            f"{row['target_recall']:>6.2f} "
            f"{row['threshold']:>10.4f} "
            f"{row['precision']:>10.3f} "
            f"{row['recall']:>7.3f} "
            f"{row['f1']:>7.3f} "
            f"{row['fp']:>5} "
            f"{row['fn']:>5} "
            f"{row['predicted_positive_rate']:>7.3f}"
        )

    print()

    # ==================================================
    # Best points
    # ==================================================

    print_metrics(
        title=
            "=== Best F1 point ===",

        row=
            best_f1,
    )

    print_metrics(
        title=
            "=== Best balanced-accuracy point ===",

        row=
            best_balanced,
    )

    # ==================================================
    # Score-bin monotonicity
    # ==================================================

    print(
        "=== Score Bins ==="
    )

    print(
        "bin   count   score mean   "
        "observed inadequate"
    )

    print(
        "-" * 52
    )

    for row in score_bins:

        print(
            f"{row['bin']:>3} "
            f"{row['count']:>7} "
            f"{row['score_mean']:>12.3f} "
            f"{row['observed_inadequate_rate']:>20.3f}"
        )

    print()

    # ==================================================
    # Seed robustness
    # ==================================================

    print_seed_worst_cases(
        seed_rows=
            seed_rows,

        operating_name=
            "saved_model_threshold",

        top_k=10,
    )

    print_seed_worst_cases(
        seed_rows=
            seed_rows,

        operating_name=
            "best_f1",

        top_k=10,
    )

    # ==================================================
    # Files
    # ==================================================

    print(
        "=== Saved Analysis ==="
    )

    print(
        f"Threshold sweep: "
        f"{THRESHOLD_SWEEP_FILE}"
    )

    print(
        f"Operating points:"
        f" {OPERATING_POINTS_FILE}"
    )

    print(
        f"Score bins:       "
        f"{SCORE_BINS_FILE}"
    )

    print(
        f"Seed breakdown:   "
        f"{SEED_BREAKDOWN_FILE}"
    )

    if not args.no_plots:

        print(
            f"PR curve:         "
            f"{PR_PLOT_FILE}"
        )

        print(
            f"FP/FN plot:       "
            f"{FP_FN_PLOT_FILE}"
        )

    print()

    print(
        "TEST split was NOT used."
    )


if __name__ == "__main__":
    main()