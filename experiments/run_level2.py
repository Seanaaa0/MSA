import csv
from pathlib import Path

from src.env import GridWorld
from src.maps import available_maps

from src.observation import (
    LocalObserver,
)

from src.adequacy import (
    DecisionStabilityMonitor,
)

from src.query import (
    QueryReality,
)

from src.level2_episode import (
    Level2EpisodeRunner,
)


# ======================================================
# Configuration
# ======================================================

POLICIES = [
    "always_trust",
    "always_query",
    "monitor",
]


MAP_NAMES = list(
    available_maps()
)


RESULTS_DIR = Path(
    "results"
)


OUTPUT_FILE = (
    RESULTS_DIR
    / "level2_partial_observation.csv"
)


# ======================================================
# Helpers
# ======================================================

def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return 1.0

    return (
        numerator
        / denominator
    )


# ======================================================
# Experiment
# ======================================================

def run():

    rows = []

    # --------------------------------------------------
    # Run every map / policy pair
    # --------------------------------------------------

    for map_name in MAP_NAMES:

        for policy in POLICIES:

            env = GridWorld(
                map_name=map_name
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

            runner = (
                Level2EpisodeRunner(
                    env=env,

                    observer=observer,

                    monitor=monitor,

                    query_reality=query,

                    policy=policy,

                    max_steps=300,

                    counterfactual_max_steps=300,
                )
            )

            result = (
                runner.run(
                    verbose=False
                )
            )

            row = {
                "map_name":
                    map_name,

                **result,
            }

            rows.append(
                row
            )

            print(
                f"{map_name:<12} "
                f"{policy:<14} "
                f"success="
                f"{result['success']} "
                f"steps="
                f"{result['steps']} "
                f"queries="
                f"{result['queries']} "
                f"beneficial="
                f"{result['beneficial_query_opportunities']} "
                f"TP="
                f"{result['true_positive']} "
                f"FP="
                f"{result['false_positive']} "
                f"FN="
                f"{result['false_negative']}"
            )

    # ==================================================
    # Save CSV
    # ==================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ==================================================
    # Aggregate policy comparison
    # ==================================================

    print()

    print(
        "=== Aggregate Policy Comparison ==="
    )

    print()

    print(
        "policy         "
        "success  steps  queries  "
        "benefit-opps  TP  FP  FN  TN"
    )

    print(
        "-" * 78
    )

    for policy in POLICIES:

        policy_rows = [
            row
            for row in rows
            if (
                row["policy"]
                == policy
            )
        ]

        successes = sum(
            row["success"]
            for row in policy_rows
        )

        steps = sum(
            row["steps"]
            for row in policy_rows
        )

        queries = sum(
            row["queries"]
            for row in policy_rows
        )

        beneficial = sum(
            row[
                "beneficial_query_opportunities"
            ]
            for row in policy_rows
        )

        tp = sum(
            row["true_positive"]
            for row in policy_rows
        )

        fp = sum(
            row["false_positive"]
            for row in policy_rows
        )

        fn = sum(
            row["false_negative"]
            for row in policy_rows
        )

        tn = sum(
            row["true_negative"]
            for row in policy_rows
        )

        print(
            f"{policy:<15}"
            f"{successes:>7} "
            f"{steps:>6} "
            f"{queries:>8} "
            f"{beneficial:>13} "
            f"{tp:>3} "
            f"{fp:>3} "
            f"{fn:>3} "
            f"{tn:>3}"
        )

    # ==================================================
    # Monitor-specific analysis
    # ==================================================

    monitor_rows = [
        row
        for row in rows
        if (
            row["policy"]
            == "monitor"
        )
    ]

    tp = sum(
        row["true_positive"]
        for row in monitor_rows
    )

    fp = sum(
        row["false_positive"]
        for row in monitor_rows
    )

    fn = sum(
        row["false_negative"]
        for row in monitor_rows
    )

    tn = sum(
        row["true_negative"]
        for row in monitor_rows
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    total_queries = sum(
        row["queries"]
        for row in monitor_rows
    )

    total_steps = sum(
        row["steps"]
        for row in monitor_rows
    )

    total_positive_gain = sum(
        row[
            "positive_step_gain_total"
        ]
        for row in monitor_rows
    )

    print()

    print(
        "=== MSA Monitor Query-Value Analysis ==="
    )

    print(
        f"True positives:          "
        f"{tp}"
    )

    print(
        f"False positives:         "
        f"{fp}"
    )

    print(
        f"False negatives:         "
        f"{fn}"
    )

    print(
        f"True negatives:          "
        f"{tn}"
    )

    print()

    print(
        f"Query precision:         "
        f"{precision:.3f}"
    )

    print(
        f"Query recall:            "
        f"{recall:.3f}"
    )

    print()

    print(
        f"Monitor total steps:     "
        f"{total_steps}"
    )

    print(
        f"Monitor total queries:   "
        f"{total_queries}"
    )

    print(
        f"Potential positive "
        f"step gain encountered:   "
        f"{total_positive_gain}"
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  TP = queried when querying would "
        "improve future consequence"
    )

    print(
        "  FP = queried although querying would "
        "not improve future consequence"
    )

    print(
        "  FN = trusted although querying would "
        "have improved future consequence"
    )

    print(
        "  TN = trusted when querying had no "
        "future benefit"
    )

    print()

    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    run()