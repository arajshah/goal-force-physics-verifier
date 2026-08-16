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


def validate_data(metrics_df: pd.DataFrame, rankings_df: pd.DataFrame) -> None:
    required_metrics = {"scenario", "seed", "interaction_mode", *METRICS}
    missing = required_metrics - set(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics.csv missing columns: {sorted(missing)}")

    required_ranks = {"scenario", "seed", "rank"}
    missing = required_ranks - set(rankings_df.columns)
    if missing:
        raise ValueError(f"rankings.csv missing columns: {sorted(missing)}")

    if set(metrics_df["scenario"]) != EXPECTED_SCENARIOS:
        raise ValueError("metrics.csv does not contain exactly the frozen 10 training scenarios.")

    train_rankings = rankings_df[rankings_df["scenario"].isin(EXPECTED_SCENARIOS)].copy()
    if set(train_rankings["scenario"]) != EXPECTED_SCENARIOS:
        raise ValueError("rankings.csv is missing one or more frozen training scenarios.")

    if len(metrics_df) != 40:
        raise ValueError(f"Expected 40 metric rows, found {len(metrics_df)}.")

    if len(train_rankings) != 40:
        raise ValueError(f"Expected 40 training ranking rows, found {len(train_rankings)}.")

    for name, frame in [("metrics", metrics_df), ("rankings", train_rankings)]:
        if frame.duplicated(["scenario", "seed"]).any():
            raise ValueError(f"{name} contains duplicate scenario/seed rows.")

        for scenario, group in frame.groupby("scenario"):
            seeds = set(group["seed"].astype(int))
            if seeds != EXPECTED_SEEDS:
                raise ValueError(f"{scenario}: expected seeds 5,6,7,8; found {sorted(seeds)}.")

    for scenario, group in train_rankings.groupby("scenario"):
        if sorted(group["rank"].astype(int)) != [1, 2, 3, 4]:
            raise ValueError(f"{scenario}: expected ranks 1,2,3,4.")


def normalize_within_scenario(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, bool]]]:
    out = df.copy()
    availability: dict[str, dict[str, bool]] = {}

    for scenario, idx in out.groupby("scenario").groups.items():
        availability[scenario] = {}

        for metric in METRICS:
            values = pd.to_numeric(out.loc[idx, metric], errors="coerce").astype(float)
            finite = np.isfinite(values.to_numpy())
            count = int(finite.sum())

            if count == 0:
                out.loc[idx, f"{metric}_norm"] = 0.5
                availability[scenario][metric] = False
                continue

            if count != len(values):
                raise ValueError(
                    f"{scenario}: metric {metric} is only partially available."
                )

            lo = float(values.min())
            hi = float(values.max())

            if hi - lo < 1e-12:
                out.loc[idx, f"{metric}_norm"] = 0.5
            else:
                out.loc[idx, f"{metric}_norm"] = (values - lo) / (hi - lo)

            availability[scenario][metric] = True

    return out, availability


def make_weight_grid(step: float) -> list[np.ndarray]:
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError("--step must divide 1.0 exactly.")

    weights = []
    for cuts in itertools.combinations_with_replacement(range(len(METRICS)), units):
        counts = np.bincount(cuts, minlength=len(METRICS))
        weights.append(counts.astype(float) / units)

    return weights


def score(df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    matrix = df[[f"{m}_norm" for m in METRICS]].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite normalized metric encountered.")
    return matrix @ weights


def pairwise_accuracy(df: pd.DataFrame, scores: np.ndarray) -> float:
    work = df.copy()
    work["_score"] = scores

    correct = 0.0
    total = 0

    for _, group in work.groupby("scenario"):
        rows = group.to_dict("records")
        for a, b in itertools.combinations(rows, 2):
            human_diff = b["rank"] - a["rank"]
            score_diff = a["_score"] - b["_score"]

            if abs(score_diff) < 1e-12:
                correct += 0.5
            elif human_diff * score_diff > 0:
                correct += 1.0

            total += 1

    return correct / total if total else 0.0


def mean_correlations(df: pd.DataFrame, scores: np.ndarray) -> tuple[float, float]:
    work = df.copy()
    work["_score"] = scores

    spearmans = []
    kendalls = []

    for _, group in work.groupby("scenario"):
        human_quality = -group["rank"].to_numpy(float)
        model_score = group["_score"].to_numpy(float)

        s = spearmanr(human_quality, model_score).statistic
        k = kendalltau(human_quality, model_score).statistic

        spearmans.append(0.0 if np.isnan(s) else float(s))
        kendalls.append(0.0 if np.isnan(k) else float(k))

    return float(np.mean(spearmans)), float(np.mean(kendalls))


def fit_weights(df: pd.DataFrame, grid: list[np.ndarray]) -> np.ndarray:
    uniform = np.full(len(METRICS), 1.0 / len(METRICS))

    best = None
    best_key = None

    for weights in grid:
        scores = score(df, weights)
        pairwise = pairwise_accuracy(df, scores)
        spearman, kendall = mean_correlations(df, scores)

        distance = float(np.sum((weights - uniform) ** 2))
        key = (pairwise, spearman, kendall, -distance)

        if best_key is None or key > best_key:
            best_key = key
            best = weights.copy()

    if best is None:
        raise RuntimeError("No weights evaluated.")

    return best


def evaluate_one(group: pd.DataFrame, scores: np.ndarray) -> dict[str, object]:
    work = group.copy()
    work["verifier_score"] = scores

    human_quality = -work["rank"].to_numpy(float)
    s = spearmanr(human_quality, scores).statistic
    k = kendalltau(human_quality, scores).statistic

    best = work.sort_values(
        ["verifier_score", "seed"],
        ascending=[False, True],
    ).iloc[0]

    human_best = work.sort_values("rank").iloc[0]
    baseline = work[work["seed"] == 5].iloc[0]

    return {
        "pairwise_accuracy": pairwise_accuracy(work, scores),
        "spearman": 0.0 if np.isnan(s) else float(s),
        "kendall": 0.0 if np.isnan(k) else float(k),
        "verifier_seed": int(best["seed"]),
        "verifier_human_rank": int(best["rank"]),
        "human_best_seed": int(human_best["seed"]),
        "top1_correct": int(best["seed"] == human_best["seed"]),
        "baseline_human_rank": int(baseline["rank"]),
        "rank_improvement_over_baseline": int(baseline["rank"]) - int(best["rank"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--rankings", type=Path, default=Path("data/rankings.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()

    metrics_df = pd.read_csv(args.metrics)
    rankings_df = pd.read_csv(args.rankings)

    validate_data(metrics_df, rankings_df)

    rankings_df = rankings_df[
        rankings_df["scenario"].isin(EXPECTED_SCENARIOS)
    ].copy()

    merged = metrics_df.merge(
        rankings_df,
        on=["scenario", "seed"],
        how="inner",
        validate="one_to_one",
    )

    normalized, availability = normalize_within_scenario(merged)
    grid = make_weight_grid(args.step)

    scenario_modes = (
        normalized[["scenario", "interaction_mode"]]
        .drop_duplicates()
        .set_index("scenario")["interaction_mode"]
        .to_dict()
    )

    scenarios = sorted(EXPECTED_SCENARIOS)

    eval_rows = []
    weight_rows = []
    scored_rows = []

    # ------------------------------------------------------------
    # Adaptive LOSO
    # ------------------------------------------------------------
    for held_out in scenarios:
        mode = scenario_modes[held_out]

        same_mode_train = normalized[
            (normalized["scenario"] != held_out)
            & (normalized["interaction_mode"] == mode)
        ].copy()

        if len(same_mode_train["scenario"].unique()) >= 1:
            fit_frame = same_mode_train
            fit_scope = "same_interaction_mode"
        else:
            # toycar/single_target currently has no second training scenario.
            fit_frame = normalized[
                normalized["scenario"] != held_out
            ].copy()
            fit_scope = "global_fallback_no_same_mode_training_scenario"

        weights = fit_weights(fit_frame, grid)

        test = normalized[
            normalized["scenario"] == held_out
        ].copy()

        scores = score(test, weights)

        result = evaluate_one(test, scores)
        result.update({
            "held_out_scenario": held_out,
            "interaction_mode": mode,
            "fit_scope": fit_scope,
            "num_fit_scenarios": int(fit_frame["scenario"].nunique()),
        })
        eval_rows.append(result)

        row = {
            "fold": f"held_out_{held_out}",
            "held_out_scenario": held_out,
            "interaction_mode": mode,
            "fit_scope": fit_scope,
            "training_scenarios": ",".join(sorted(fit_frame["scenario"].unique())),
        }
        row.update({
            metric: float(weight)
            for metric, weight in zip(METRICS, weights)
        })
        weight_rows.append(row)

        test["fold"] = f"held_out_{held_out}"
        test["verifier_score"] = scores
        test["verifier_rank"] = (
            test["verifier_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        scored_rows.append(test)

    # ------------------------------------------------------------
    # Fit one development-all vector per interaction mode.
    # Exploratory only; some groups are tiny.
    # ------------------------------------------------------------
    adaptive_development_weights = {}

    for mode, group in normalized.groupby("interaction_mode"):
        weights = fit_weights(group, grid)

        adaptive_development_weights[mode] = {
            metric: float(weight)
            for metric, weight in zip(METRICS, weights)
        }

        row = {
            "fold": f"development_all_{mode}",
            "held_out_scenario": "",
            "interaction_mode": mode,
            "fit_scope": "all_training_scenarios_in_mode",
            "training_scenarios": ",".join(sorted(group["scenario"].unique())),
        }
        row.update(adaptive_development_weights[mode])
        weight_rows.append(row)

    evaluation = pd.DataFrame(eval_rows)

    args.output.mkdir(parents=True, exist_ok=True)

    evaluation_path = args.output / "adaptive_evaluation.csv"
    weights_path = args.output / "adaptive_weights.csv"
    scored_path = args.output / "adaptive_scored_candidates.csv"
    config_path = args.output / "adaptive_scorer_config.json"

    evaluation.to_csv(evaluation_path, index=False)
    pd.DataFrame(weight_rows).to_csv(weights_path, index=False)
    pd.concat(scored_rows, ignore_index=True).to_csv(scored_path, index=False)

    config = {
        "metrics": METRICS,
        "strategy": "interaction-conditioned weights",
        "grid_step": args.step,
        "adaptive_development_weights": adaptive_development_weights,
        "scenario_modes": scenario_modes,
        "single_target_loso_policy": (
            "If no other scenario with the same interaction_mode exists, "
            "fall back to global training scenarios excluding the held-out scenario."
        ),
        "warning": (
            "Exploratory only. Interaction groups are small: collision=4, "
            "strike=2, direct_actuation=3, single_target=1. Do not treat "
            "mode-specific fitted weights as statistically reliable."
        ),
    }

    config_path.write_text(json.dumps(config, indent=2) + "\n")

    print("\nADAPTIVE LOSO RESULTS")
    print(
        evaluation[
            [
                "held_out_scenario",
                "interaction_mode",
                "fit_scope",
                "num_fit_scenarios",
                "pairwise_accuracy",
                "spearman",
                "kendall",
                "verifier_seed",
                "verifier_human_rank",
                "human_best_seed",
                "top1_correct",
                "baseline_human_rank",
                "rank_improvement_over_baseline",
            ]
        ].to_string(index=False)
    )

    print("\nMEAN ADAPTIVE LOSO RESULTS")
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

    print("\nDEVELOPMENT-ALL WEIGHTS BY INTERACTION MODE")
    for mode in sorted(adaptive_development_weights):
        print(f"\n{mode}")
        for metric, weight in adaptive_development_weights[mode].items():
            print(f"  {metric}: {weight:.3f}")

    print(f"\nSaved {evaluation_path}")
    print(f"Saved {weights_path}")
    print(f"Saved {scored_path}")
    print(f"Saved {config_path}")


if __name__ == "__main__":
    main()