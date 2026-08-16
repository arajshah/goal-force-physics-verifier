import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


TEST_SCENARIOS = {"soccer", "paw_tool3"}


def normalize_within_scenario(df, metrics):
    out = df.copy()

    for scenario, idx in out.groupby("scenario").groups.items():
        for metric in metrics:
            values = pd.to_numeric(out.loc[idx, metric], errors="coerce").astype(float)
            finite = np.isfinite(values)

            if finite.sum() == 0:
                out.loc[idx, f"{metric}_norm"] = 0.5
            elif finite.sum() != len(values):
                raise ValueError(
                    f"{scenario}: partially missing metric {metric}"
                )
            else:
                lo, hi = values.min(), values.max()

                if hi - lo < 1e-12:
                    out.loc[idx, f"{metric}_norm"] = 0.5
                else:
                    out.loc[idx, f"{metric}_norm"] = (
                        values - lo
                    ) / (hi - lo)

    return out


def pairwise_accuracy(group):
    correct = 0.0
    total = 0

    rows = group.to_dict("records")

    for a, b in itertools.combinations(rows, 2):
        human_diff = b["rank"] - a["rank"]
        score_diff = a["verifier_score"] - b["verifier_score"]

        if abs(score_diff) < 1e-12:
            correct += 0.5
        elif human_diff * score_diff > 0:
            correct += 1.0

        total += 1

    return correct / total


def main():
    metrics_df = pd.read_csv("results/test_metrics.csv")
    rankings_df = pd.read_csv("data/rankings.csv")

    with open("results/scorer_config.json") as f:
        config = json.load(f)

    metrics = config["metrics"]
    weights = config["development_all_weights"]

    metrics_df = metrics_df[
        metrics_df["scenario"].isin(TEST_SCENARIOS)
    ].copy()

    rankings_df = rankings_df[
        rankings_df["scenario"].isin(TEST_SCENARIOS)
    ].copy()

    if len(metrics_df) != 8:
        raise ValueError(f"Expected 8 test metric rows, found {len(metrics_df)}")

    if len(rankings_df) != 8:
        raise ValueError(f"Expected 8 test ranking rows, found {len(rankings_df)}")

    df = metrics_df.merge(
        rankings_df,
        on=["scenario", "seed"],
        validate="one_to_one",
    )

    df = normalize_within_scenario(df, metrics)

    df["verifier_score"] = sum(
        df[f"{metric}_norm"] * weights[metric]
        for metric in metrics
    )

    df["verifier_rank"] = (
        df.groupby("scenario")["verifier_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    results = []

    for scenario, group in df.groupby("scenario"):
        group = group.copy()

        best = group.sort_values(
            ["verifier_score", "seed"],
            ascending=[False, True],
        ).iloc[0]

        human_best = group.sort_values("rank").iloc[0]
        baseline = group[group["seed"] == 5].iloc[0]

        human_quality = -group["rank"].to_numpy(float)
        scores = group["verifier_score"].to_numpy(float)

        spearman = spearmanr(human_quality, scores).statistic
        kendall = kendalltau(human_quality, scores).statistic

        results.append({
            "scenario": scenario,
            "pairwise_accuracy": pairwise_accuracy(group),
            "spearman": 0.0 if np.isnan(spearman) else spearman,
            "kendall": 0.0 if np.isnan(kendall) else kendall,
            "verifier_seed": int(best["seed"]),
            "verifier_human_rank": int(best["rank"]),
            "human_best_seed": int(human_best["seed"]),
            "top1_correct": int(best["seed"] == human_best["seed"]),
            "baseline_seed": 5,
            "baseline_human_rank": int(baseline["rank"]),
            "rank_improvement_over_baseline":
                int(baseline["rank"]) - int(best["rank"]),
        })

    result_df = pd.DataFrame(results)

    df.to_csv(
        "results/final_test_scored_candidates.csv",
        index=False,
    )

    result_df.to_csv(
        "results/final_test_evaluation.csv",
        index=False,
    )

    print("\nFROZEN WEIGHTS")
    for metric in metrics:
        print(f"{metric}: {weights[metric]:.3f}")

    print("\nFINAL TEST RESULTS")
    print(result_df.to_string(index=False))

    print("\nMEAN TEST RESULTS")
    print(
        result_df[
            [
                "pairwise_accuracy",
                "spearman",
                "kendall",
                "top1_correct",
                "rank_improvement_over_baseline",
            ]
        ].mean().to_string()
    )


if __name__ == "__main__":
    main()