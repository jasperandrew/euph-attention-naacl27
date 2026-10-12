"""Summarize per-head Cohen's d across all layers and attention metrics.

Reads cached finetuned attention profiles for the four full-model runs in each
split. For every run, layer, metric, and head, d compares euphemistic with
non-euphemistic examples (positive d means a larger value for euphemistic
examples). The script writes run-level and aggregated descriptive summaries;
it does not perform head-level significance tests.

Run with: conda run -n attn python head_level_summary.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / "output" / "plots"
OUTPUT_DIR = ROOT / "output" / "stats"
FIGURE_DIR = ROOT / "output" / "paper"
FIGURE_PATH = FIGURE_DIR / "layer10_11_per_head_d.png"
SPLITS = {
    "Stratified": [f"split-42_train-{seed}_freeze-0" for seed in (41, 42, 43, 44)],
    "PET-disjoint": [f"split-lrec_train-{seed}_freeze-0" for seed in (41, 42, 43, 44)],
}
METRICS = (
    "Inward (Context → PET)",
    "Outward (PET → Context)",
    "Internal (PET → PET)",
    "Entropy",
    "Sink (PET → Sink)",
    "Sink (Sink → PET)",
    "Distance",
)


def cohens_d(euph, non_euph):
    """Return pooled-standard-deviation Cohen's d for two sample arrays."""
    n_e, n_n = len(euph), len(non_euph)
    if n_e < 2 or n_n < 2:
        raise ValueError("Each class needs at least two examples")
    pooled_sd = np.sqrt(
        ((n_e - 1) * euph.var(ddof=1) + (n_n - 1) * non_euph.var(ddof=1))
        / (n_e + n_n - 2)
    )
    return float((euph.mean() - non_euph.mean()) / pooled_sd) if pooled_sd else 0.0


def collect_effects():
    records = []
    expected_layers = None
    expected_heads = None

    for split, run_names in SPLITS.items():
        for run_name in run_names:
            path = PROFILE_DIR / run_name / "data" / "ft_df.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing cached profile: {path}")
            frame = pd.read_parquet(path)
            required = {"Layer", "Head", "Condition", *METRICS}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")

            layers = tuple(sorted(int(value) for value in frame["Layer"].unique()))
            heads = tuple(sorted(int(value) for value in frame["Head"].unique()))
            if expected_layers is None:
                expected_layers, expected_heads = layers, heads
            elif layers != expected_layers or heads != expected_heads:
                raise ValueError(f"Layer/head inventory differs in {path}")

            for layer in layers:
                layer_frame = frame[frame["Layer"] == layer]
                for metric in METRICS:
                    for head in heads:
                        head_frame = layer_frame[layer_frame["Head"] == head]
                        euph = head_frame.loc[
                            head_frame["Condition"] == "Euphemistic", metric
                        ].to_numpy()
                        non_euph = head_frame.loc[
                            head_frame["Condition"] == "Non-Euphemistic", metric
                        ].to_numpy()
                        records.append(
                            {
                                "split": split,
                                "run": run_name,
                                "layer": layer,
                                "metric": metric,
                                "head": head,
                                "n_euphemistic": len(euph),
                                "n_non_euphemistic": len(non_euph),
                                "cohens_d": cohens_d(euph, non_euph),
                            }
                        )

    return pd.DataFrame.from_records(records), expected_layers, expected_heads


def summarize_by_head(run_effects):
    """Aggregate each head's effect across runs, separately by split and overall."""
    group_cols = ["layer", "metric", "head"]
    summaries = []
    for split, frame in [("All runs", run_effects), *run_effects.groupby("split")]:
        summary = (
            frame.groupby(group_cols)["cohens_d"]
            .agg(mean_d="mean", sd_d="std", min_d="min", max_d="max", n_runs="count")
            .reset_index()
        )
        positive = (
            frame.assign(positive=frame["cohens_d"] > 0)
            .groupby(group_cols)["positive"]
            .sum()
            .reset_index(name="positive_runs")
        )
        summary.insert(0, "split", split)
        summaries.append(summary.merge(positive, on=group_cols))
    return pd.concat(summaries, ignore_index=True)


def summarize_by_layer(run_effects, heads):
    """Aggregate effect sizes over heads and runs for each layer and metric."""
    group_cols = ["layer", "metric"]
    summaries = []
    for split, frame in [("All runs", run_effects), *run_effects.groupby("split")]:
        summary = (
            frame.groupby(group_cols)["cohens_d"]
            .agg(mean_d="mean", mean_abs_d=lambda values: values.abs().mean(), n_head_runs="count")
            .reset_index()
        )
        positive = (
            frame.assign(positive=frame["cohens_d"] > 0)
            .groupby(group_cols)["positive"]
            .sum()
            .reset_index(name="positive_head_runs")
        )
        head_means = frame.groupby(group_cols + ["head"])["cohens_d"].mean().reset_index()
        positive_heads = (
            head_means.assign(positive=head_means["cohens_d"] > 0)
            .groupby(group_cols)["positive"]
            .sum()
            .reset_index(name="positive_mean_heads")
        )
        top_heads = (
            head_means.sort_values(group_cols + ["cohens_d"], ascending=[True, True, False])
            .groupby(group_cols, as_index=False)
            .first()[group_cols + ["head", "cohens_d"]]
            .rename(columns={"head": "largest_mean_d_head", "cohens_d": "largest_mean_head_d"})
        )
        summary.insert(0, "split", split)
        summaries.append(
            summary.merge(positive, on=group_cols)
            .merge(positive_heads, on=group_cols)
            .merge(top_heads, on=group_cols)
        )

    result = pd.concat(summaries, ignore_index=True)
    result["n_heads"] = len(heads)
    return result


def main():
    run_effects, layers, heads = collect_effects()
    if not run_effects["split"].value_counts().eq(4 * len(layers) * len(METRICS) * len(heads)).all():
        raise ValueError("Expected four full-model runs for each split")

    head_summary = summarize_by_head(run_effects)
    layer_summary = summarize_by_layer(run_effects, heads)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "run-level per-head effects": OUTPUT_DIR / "head_level_effects_by_run.csv",
        "per-head run summaries": OUTPUT_DIR / "head_level_effects_by_head.csv",
        "layer-metric summaries": OUTPUT_DIR / "head_level_effects_by_layer.csv",
    }
    run_effects.to_csv(outputs["run-level per-head effects"], index=False)
    head_summary.to_csv(outputs["per-head run summaries"], index=False)
    layer_summary.to_csv(outputs["layer-metric summaries"], index=False)

    from src.profiler.plots import plot_layer10_11_per_head_d
    plot_layer10_11_per_head_d(run_effects, FIGURE_PATH)

    print(
        f"Analyzed {len(run_effects):,} per-head effects across "
        f"{len(layers)} layers, {len(METRICS)} metrics, {len(heads)} heads, "
        f"and {len(SPLITS) * 4} full-model runs."
    )
    print("\nAll-run layer and metric summaries (mean d; positive head-runs / total):")
    all_runs = layer_summary[layer_summary["split"] == "All runs"]
    for row in all_runs.sort_values(["layer", "metric"]).itertuples():
        print(
            f"L{row.layer:02d} | {row.metric:<25} | {row.mean_d:>+6.2f} | "
            f"{row.positive_head_runs}/{row.n_head_runs} | "
            f"{row.positive_mean_heads}/{row.n_heads} heads positive on average"
        )
    print("\nWrote:")
    for label, path in outputs.items():
        print(f"  {label}: {path.relative_to(ROOT)}")
    print(f"  six-panel per-head figure: {FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
