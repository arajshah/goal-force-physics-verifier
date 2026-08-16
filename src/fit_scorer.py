from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


METRICS = [
    "goal_completion",
    "direction_alignment",
    "contact_causality",
    "motion_stability",
    "flow_consistency",
]

EXPECTED_SCENARIOS = {
    "ball_dominos",
    "pendulum",
    "pool",
    "bulb",
    "cantaloupes",
    "tennis",
    "golf",
    "toycar",
    "paw_tool1",
    "paw_tool2",
}

EXPECTED_SEEDS = {5, 6, 7, 8}


def validate_training_data(
    metrics_frame: pd.DataFrame,
    rankings_frame: pd.DataFrame,
) -> None:

    required_metric_columns = {
        "scenario",
        "seed",
        *METRICS,
    }
    missing_metrics = required_metric_columns - set(metrics_frame.columns)
    if missing_metrics:
        raise ValueError(
            f"metrics.csv is missing: {sorted(missing_metrics)}"
        )

    required_rank_columns = {"scenario", "seed", "rank"}
    missing_ranks = required_rank_columns - set(rankings_frame.columns)
    if missing_ranks:
        raise ValueError(
            f"rankings.csv is missing: {sorted(missing_ranks)}"
        )

    metric_scenarios = set(metrics_frame["scenario"].astype(str))
    ranking_scenarios = set(rankings_frame["scenario"].astype(str))

    if metric_scenarios != EXPECTED_SCENARIOS:
        raise ValueError(
            "metrics.csv scenario set does not match the frozen 10-scenario "
            f"training split.\nExpected: {sorted(EXPECTED_SCENARIOS)}\n"
            f"Found:    {sorted(metric_scenarios)}"
        )

    if ranking_scenarios != EXPECTED_SCENARIOS:
        raise ValueError(
            "rankings.csv scenario set does not match the frozen 10-scenario "
            f"training split.\nExpected: {sorted(EXPECTED_SCENARIOS)}\n"
            f"Found:    {sorted(ranking_scenarios)}"
        )

    if len(metrics_frame) != 40:
        raise ValueError(
            f"Expected 40 metric rows (10 scenarios x 4 seeds), "
            f"found {len(metrics_frame)}."
        )

    if len(rankings_frame) != 40:
        raise ValueError(
            f"Expected 40 ranking rows (10 scenarios x 4 seeds), "
            f"found {len(rankings_frame)}."
        )

    for name, frame in [
        ("metrics.csv", metrics_frame),
        ("rankings.csv", rankings_frame),
    ]:
        duplicates = frame.duplicated(["scenario", "seed"], keep=False)
        if duplicates.any():
            bad = frame.loc[duplicates, ["scenario", "seed"]]
            raise ValueError(
                f"{name} contains duplicate scenario/seed rows:\n"
                f"{bad.to_string(index=False)}"
            )

        for scenario, group in frame.groupby("scenario"):
            seeds = set(group["seed"].astype(int))
            if seeds != EXPECTED_SEEDS:
                raise ValueError(
                    f"{name}: {scenario} should have seeds "
                    f"{sorted(EXPECTED_SEEDS)}, found {sorted(seeds)}."
                )

    for scenario, group in rankings_frame.groupby("scenario"):
        ranks = sorted(group["rank"].astype(int).tolist())
        if ranks != [1, 2, 3, 4]:
            raise ValueError(
                f"rankings.csv: {scenario} should contain ranks "
                f"[1, 2, 3, 4], found {ranks}."
            )


def minmax_within_scenario(
    frame: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, bool]]]:

    result = frame.copy()
    availability: dict[str, dict[str, bool]] = {}

    for scenario, indices in result.groupby("scenario").groups.items():
        availability[scenario] = {}

        for metric in metrics:
            values = pd.to_numeric(
                result.loc[indices, metric],
                errors="coerce",
            ).astype(float)

            finite = np.isfinite(values.to_numpy())
            finite_count = int(finite.sum())

            if finite_count == 0:
                # Metric does not apply to this scenario (e.g. toycar contact).
                result.loc[indices, f"{metric}_norm"] = 0.5
                availability[scenario][metric] = False
                continue

            if finite_count != len(values):
                raise ValueError(
                    f"{scenario}: metric '{metric}' is available for only "
                    f"{finite_count}/{len(values)} candidates. Fix the metric "
                    "extractor rather than partially imputing it."
                )

            minimum = float(values.min())
            maximum = float(values.max())

            if maximum - minimum < 1e-12:
                normalized = np.full(len(values), 0.5, dtype=float)
            else:
                normalized = (values - minimum) / (maximum - minimum)

            result.loc[indices, f"{metric}_norm"] = normalized
            availability[scenario][metric] = True

    return result, availability


def weight_grid(
    dimension: int,
    step: float,
) -> list[np.ndarray]:
    units = round(1.0 / step)

    if not np.isclose(units * step, 1.0):
        raise ValueError("--step must divide 1.0 exactly.")

    weights: list[np.ndarray] = []

    for cuts in itertools.combinations_with_replacement(
        range(dimension),
        units,
    ):
        counts = np.bincount(cuts, minlength=dimension)
        weights.append(counts.astype(float) / units)

    return weights


def score_rows(
    frame: pd.DataFrame,
    metrics: list[str],
    weights: np.ndarray,
) -> np.ndarray:
    columns = [f"{metric}_norm" for metric in metrics]
    matrix = frame[columns].to_numpy(dtype=float)

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Non-finite normalized metric encountered during scoring."
        )

    return matrix @ weights


def pairwise_accuracy(
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> float:
    correct = 0.0
    total = 0

    work = frame.copy()
    work["_score"] = scores

    for _, group in work.groupby("scenario"):
        rows = group.to_dict("records")

        for left, right in itertools.combinations(rows, 2):
            human_difference = right["rank"] - left["rank"]
            score_difference = left["_score"] - right["_score"]

            if abs(score_difference) < 1e-12:
                correct += 0.5
            elif human_difference * score_difference > 0:
                correct += 1.0

            total += 1

    return correct / total if total else 0.0


def mean_rank_correlation(
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> tuple[float, float]:
    work = frame.copy()
    work["_score"] = scores

    spearman_values: list[float] = []
    kendall_values: list[float] = []

    for _, group in work.groupby("scenario"):
        human_quality = -group["rank"].to_numpy(dtype=float)
        model_score = group["_score"].to_numpy(dtype=float)

        spearman = spearmanr(
            human_quality,
            model_score,
        ).statistic
        kendall = kendalltau(
            human_quality,
            model_score,
        ).statistic

        spearman_values.append(
            0.0 if np.isnan(spearman) else float(spearman)
        )
        kendall_values.append(
            0.0 if np.isnan(kendall) else float(kendall)
        )

    return (
        float(np.mean(spearman_values)),
        float(np.mean(kendall_values)),
    )


def fit_weights(
    frame: pd.DataFrame,
    metrics: list[str],
    candidates: list[np.ndarray],
) -> np.ndarray:
    best_weights: np.ndarray | None = None
    best_key: tuple[float, float, float, float] | None = None
    uniform = np.full(len(metrics), 1.0 / len(metrics))

    for weights in candidates:
        scores = score_rows(frame, metrics, weights)
        pairwise = pairwise_accuracy(frame, scores)
        spearman, kendall = mean_rank_correlation(frame, scores)

        # Primary objective: human pairwise agreement.
        # Tie-breaks: Spearman, Kendall, then less-extreme weights.
        distance_from_uniform = float(
            np.sum((weights - uniform) ** 2)
        )
        key = (
            pairwise,
            spearman,
            kendall,
            -distance_from_uniform,
        )

        if best_key is None or key > best_key:
            best_key = key
            best_weights = weights.copy()

    if best_weights is None:
        raise RuntimeError("No candidate weights were evaluated.")

    return best_weights


def evaluate_scenario(
    group: pd.DataFrame,
    scores: np.ndarray,
) -> dict[str, object]:
    scored = group.copy()
    scored["verifier_score"] = scores

    pairwise = pairwise_accuracy(scored, scores)
    human_quality = -scored["rank"].to_numpy(dtype=float)

    spearman = spearmanr(
        human_quality,
        scores,
    ).statistic
    kendall = kendalltau(
        human_quality,
        scores,
    ).statistic

    verifier_row = scored.sort_values(
        ["verifier_score", "seed"],
        ascending=[False, True],
    ).iloc[0]
    human_best_row = scored.sort_values("rank").iloc[0]

    baseline_rows = scored[scored["seed"] == 5]
    baseline_rank = (
        int(baseline_rows.iloc[0]["rank"])
        if len(baseline_rows)
        else None
    )

    return {
        "pairwise_accuracy": float(pairwise),
        "spearman": (
            0.0 if np.isnan(spearman) else float(spearman)
        ),
        "kendall": (
            0.0 if np.isnan(kendall) else float(kendall)
        ),
        "verifier_seed": int(verifier_row["seed"]),
        "verifier_human_rank": int(verifier_row["rank"]),
        "human_best_seed": int(human_best_row["seed"]),
        "top1_correct": int(
            verifier_row["seed"] == human_best_row["seed"]
        ),
        "baseline_seed": 5,
        "baseline_human_rank": baseline_rank,
        "rank_improvement_over_baseline": (
            baseline_rank - int(verifier_row["rank"])
            if baseline_rank is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/metrics.csv"),
    )
    parser.add_argument(
        "--rankings",
        type=Path,
        default=Path("data/rankings.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
    )
    args = parser.parse_args()

    metrics_frame = pd.read_csv(args.metrics)
    rankings_frame = pd.read_csv(args.rankings)

    validate_training_data(metrics_frame, rankings_frame)

    merged = metrics_frame.merge(
        rankings_frame,
        on=["scenario", "seed"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 40:
        raise ValueError(
            f"Expected 40 merged candidates, found {len(merged)}."
        )

    normalized, availability = minmax_within_scenario(
        merged,
        METRICS,
    )

    candidates = weight_grid(len(METRICS), args.step)
    scenarios = sorted(normalized["scenario"].unique())

    weight_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    scored_rows: list[pd.DataFrame] = []

    for held_out in scenarios:
        train = normalized[
            normalized["scenario"] != held_out
        ].copy()
        test = normalized[
            normalized["scenario"] == held_out
        ].copy()

        weights = fit_weights(
            train,
            METRICS,
            candidates,
        )

        test_scores = score_rows(
            test,
            METRICS,
            weights,
        )

        test["fold"] = f"held_out_{held_out}"
        test["verifier_score"] = test_scores
        test["verifier_rank"] = test["verifier_score"].rank(
            ascending=False,
            method="min",
        ).astype(int)
        scored_rows.append(test)

        result = evaluate_scenario(test, test_scores)
        result["fold"] = f"held_out_{held_out}"
        result["held_out_scenario"] = held_out

        unavailable = [
            metric
            for metric in METRICS
            if not availability[held_out][metric]
        ]
        result["unavailable_metrics"] = ",".join(unavailable)
        evaluation_rows.append(result)

        row: dict[str, object] = {
            "fold": f"held_out_{held_out}",
            "training_scenarios": ",".join(
                scenario
                for scenario in scenarios
                if scenario != held_out
            ),
        }
        row.update({
            metric: float(weight)
            for metric, weight in zip(METRICS, weights)
        })
        weight_rows.append(row)

    # Development-only fit using all 10 training scenarios.
    # These weights are the ones we freeze before touching the final test set.
    global_weights = fit_weights(
        normalized,
        METRICS,
        candidates,
    )

    global_scores = score_rows(
        normalized,
        METRICS,
        global_weights,
    )

    global_frame = normalized.copy()
    global_frame["fold"] = "development_all"
    global_frame["verifier_score"] = global_scores
    global_frame["verifier_rank"] = (
        global_frame.groupby("scenario")["verifier_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    scored_rows.append(global_frame)

    global_row: dict[str, object] = {
        "fold": "development_all",
        "training_scenarios": ",".join(scenarios),
    }
    global_row.update({
        metric: float(weight)
        for metric, weight in zip(METRICS, global_weights)
    })
    weight_rows.append(global_row)

    args.output.mkdir(parents=True, exist_ok=True)

    weights_path = args.output / "weights.csv"
    evaluation_path = args.output / "evaluation.csv"
    scored_path = args.output / "scored_candidates.csv"
    config_path = args.output / "scorer_config.json"

    pd.DataFrame(weight_rows).to_csv(
        weights_path,
        index=False,
    )

    evaluation = pd.DataFrame(evaluation_rows)
    evaluation.to_csv(
        evaluation_path,
        index=False,
    )

    pd.concat(
        scored_rows,
        ignore_index=True,
    ).to_csv(
        scored_path,
        index=False,
    )

    availability_json = {
        scenario: {
            metric: bool(is_available)
            for metric, is_available in metric_map.items()
        }
        for scenario, metric_map in availability.items()
    }

    config = {
        "metrics": METRICS,
        "normalization": (
            "within-scenario min-max; metrics unavailable for an entire "
            "scenario are assigned neutral constant 0.5"
        ),
        "missing_metric_policy": (
            "all-missing within a scenario => normalized to 0.5 for all "
            "four candidates; partially missing within a scenario => error"
        ),
        "metric_availability": availability_json,
        "weight_constraint": "nonnegative, sum to 1",
        "grid_step": args.step,
        "development_all_weights": {
            metric: float(weight)
            for metric, weight in zip(
                METRICS,
                global_weights,
            )
        },
        "training_scenarios": scenarios,
        "training_candidates": int(len(normalized)),
        "warning": (
            "development_all uses all 10 training scenarios and is not an "
            "unbiased generalization estimate. Use leave-one-scenario-out "
            "results for development generalization, then freeze these "
            "weights before evaluating the 2 untouched final-test scenarios."
        ),
    }

    config_path.write_text(
        json.dumps(config, indent=2) + "\n"
    )

    print("\nMetric availability")
    for scenario in scenarios:
        unavailable = [
            metric
            for metric in METRICS
            if not availability[scenario][metric]
        ]
        if unavailable:
            print(
                f"  {scenario}: unavailable -> "
                f"{', '.join(unavailable)}"
            )
        else:
            print(f"  {scenario}: all metrics available")

    print("\nLeave-one-scenario-out evaluation")
    print(
        evaluation[
            [
                "held_out_scenario",
                "pairwise_accuracy",
                "spearman",
                "kendall",
                "verifier_seed",
                "verifier_human_rank",
                "human_best_seed",
                "top1_correct",
                "baseline_human_rank",
                "rank_improvement_over_baseline",
                "unavailable_metrics",
            ]
        ].to_string(index=False)
    )

    print("\nMean held-out results")
    print(
        evaluation[
            [
                "pairwise_accuracy",
                "spearman",
                "kendall",
                "top1_correct",
                "rank_improvement_over_baseline",
            ]
        ].mean(numeric_only=True).to_string()
    )

    print("\nFrozen development-all weights")
    for metric, weight in zip(METRICS, global_weights):
        print(f"  {metric}: {weight:.3f}")

    print(f"\nSaved {weights_path}")
    print(f"Saved {evaluation_path}")
    print(f"Saved {scored_path}")
    print(f"Saved {config_path}")


if __name__ == "__main__":
    main()