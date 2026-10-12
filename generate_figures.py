"""Generate run-aggregated Figure 1 from cached attention profiles.

Run with ``conda run -n attn python generate_figures.py``. The script reads
the cached base and fully unfrozen profiles for the four training seeds under
each split; it does not run model inference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / "output" / "plots"
FIG_DIR = ROOT / "output" / "paper"

SPLITS = {
    "Stratified": [f"split-42_train-{seed}_freeze-0" for seed in (41, 42, 43, 44)],
    "PET-disjoint": [f"split-lrec_train-{seed}_freeze-0" for seed in (41, 42, 43, 44)],
}
COMBINED_OUTPUT = "aggregate_profile_ft_minus_base_by_label.png"
MODEL_COMBINED_OUTPUT = "aggregate_profile_base_vs_ft_by_label.png"

METRICS = [
    ("Internal (PET → PET)", "Internal (PET → PET)"),
    ("Inward (Context → PET)", "Inward (Context → PET)"),
    ("Sink (PET → Sink)", "From Sink (PET → Sink)"),
    ("Distance", "Distance"),
    ("Outward (PET → Context)", "Outward (PET → Context)"),
    ("Sink (Sink → PET)", "To Sink (Sink → PET)"),
]
CONDITIONS = ("Euphemistic", "Non-Euphemistic")
MODELS = ("Finetuned", "Base")
MODEL_LINESTYLES = {"Finetuned": "-", "Base": "--"}
COLORS = {"Euphemistic": "#66c2a5", "Non-Euphemistic": "#fc8d62"}
LAYERS = np.arange(1, 13)
BOOTSTRAPS = 3000
BOOTSTRAP_SEED = 26012


def _load_run_attention_profiles(run_name):
    """Load paired caches and return per-example changes and model values."""
    run_dir = PROFILE_DIR / run_name / "data"
    ft = pd.read_parquet(run_dir / "ft_df.parquet").reset_index(drop=True)
    base = pd.read_parquet(run_dir / "base_df.parquet").reset_index(drop=True)

    identity_cols = ["Layer", "Head", "Condition", "Sample"]
    if len(ft) != len(base) or not ft[identity_cols].equals(base[identity_cols]):
        raise ValueError(f"Base and finetuned profile rows are not aligned for {run_name}")

    n_layers = ft["Layer"].nunique()
    n_heads = ft["Head"].nunique()
    rows_per_example = n_layers * n_heads
    if len(ft) % rows_per_example:
        raise ValueError(f"Unexpected profile row count in {run_name}")

    # profile_dataframe appends one layer×head block per input example. Recover
    # that instance index because Sample stores the PET string, which can repeat.
    instance = np.arange(len(ft)) // rows_per_example
    for start in range(0, len(ft), rows_per_example):
        block = ft.iloc[start : start + rows_per_example]
        if (block["Condition"].nunique() != 1 or block["Sample"].nunique() != 1
                or len(block[["Layer", "Head"]].drop_duplicates()) != rows_per_example):
            raise ValueError(f"Profile rows are not complete example blocks in {run_name}")

    delta = ft[[metric for metric, _ in METRICS]].subtract(
        base[[metric for metric, _ in METRICS]]
    )
    delta["Condition"] = ft["Condition"].to_numpy()
    delta["Layer"] = ft["Layer"].to_numpy()
    delta["Instance"] = instance

    def reduce_examples(frame):
        result = {}
        for condition in CONDITIONS:
            per_example = (
                frame.loc[frame["Condition"] == condition]
                .groupby(["Instance", "Layer"], sort=True)[[metric for metric, _ in METRICS]]
                .mean()
                .reset_index()
            )
            per_example = per_example[per_example["Layer"].isin(LAYERS)]
            instances = sorted(per_example["Instance"].unique())
            result[condition] = per_example[[metric for metric, _ in METRICS]].to_numpy().reshape(
                len(instances), len(LAYERS), len(METRICS)
            )
        return result

    data = reduce_examples(delta)
    model_data = {}
    for model, frame in (("Base", base), ("Finetuned", ft)):
        values = frame[[metric for metric, _ in METRICS]].copy()
        values["Condition"] = ft["Condition"].to_numpy()
        values["Layer"] = ft["Layer"].to_numpy()
        values["Instance"] = instance
        model_data[model] = reduce_examples(values)
    data["models"] = model_data
    return data


def _load_all_run_attention_profiles():
    runs = {}
    for split, run_names in SPLITS.items():
        runs[split] = {
            run_name: _load_run_attention_profiles(run_name) for run_name in run_names
        }

        # All training seeds within a split must be evaluated on the same
        # ordered examples, so bootstrap resampling can be paired across runs.
        for condition in CONDITIONS:
            counts = {arr.shape[0] for arr in
                      (runs[split][name][condition] for name in run_names)}
            if len(counts) != 1:
                raise ValueError(f"Example counts differ across {split} training runs")
    return runs


def _aggregate_mean_change_curves(split_runs):
    return {
        condition: np.mean(
            [run_data[condition].mean(axis=0) for run_data in split_runs.values()], axis=0
        )
        for condition in CONDITIONS
    }


def _bootstrap_change_curve_intervals(all_runs, split_names, rng):
    """Bootstrap finetuned-minus-base curve intervals over runs and examples."""
    draws = {condition: [] for condition in CONDITIONS}
    n_runs_by_split = {split: len(all_runs[split]) for split in split_names}
    for _ in range(BOOTSTRAPS):
        split_means = {condition: [] for condition in CONDITIONS}
        for split in split_names:
            run_names = list(all_runs[split])
            selected_runs = rng.choice(run_names, size=n_runs_by_split[split], replace=True)
            for condition in CONDITIONS:
                n_examples = next(iter(all_runs[split].values()))[condition].shape[0]
                selected_examples = rng.integers(0, n_examples, size=n_examples)
                run_means = [
                    all_runs[split][run_name][condition][selected_examples].mean(axis=0)
                    for run_name in selected_runs
                ]
                split_means[condition].append(np.mean(run_means, axis=0))
        for condition in CONDITIONS:
            draws[condition].append(np.mean(split_means[condition], axis=0))

    return {
        condition: np.percentile(np.asarray(draws[condition]), [2.5, 97.5], axis=0)
        for condition in CONDITIONS
    }


def _aggregate_mean_base_finetuned_curves(split_runs):
    return {
        model: {
            condition: np.mean(
                [run_data["models"][model][condition].mean(axis=0)
                 for run_data in split_runs.values()], axis=0
            )
            for condition in CONDITIONS
        }
        for model in MODELS
    }


def _bootstrap_base_finetuned_curve_intervals(all_runs, split_names, rng):
    """Hierarchical bootstrap of base and finetuned values by split and class."""
    draws = {model: {condition: [] for condition in CONDITIONS} for model in MODELS}
    n_runs_by_split = {split: len(all_runs[split]) for split in split_names}
    for _ in range(BOOTSTRAPS):
        split_means = {
            model: {condition: [] for condition in CONDITIONS} for model in MODELS
        }
        for split in split_names:
            run_names = list(all_runs[split])
            selected_runs = rng.choice(run_names, size=n_runs_by_split[split], replace=True)
            for condition in CONDITIONS:
                n_examples = next(iter(all_runs[split].values()))["models"][MODELS[0]][condition].shape[0]
                selected_examples = rng.integers(0, n_examples, size=n_examples)
                for model in MODELS:
                    run_means = [
                        all_runs[split][run_name]["models"][model][condition][selected_examples].mean(axis=0)
                        for run_name in selected_runs
                    ]
                    split_means[model][condition].append(np.mean(run_means, axis=0))
        for model in MODELS:
            for condition in CONDITIONS:
                draws[model][condition].append(
                    np.mean(split_means[model][condition], axis=0)
                )

    return {
        model: {
            condition: np.percentile(np.asarray(draws[model][condition]), [2.5, 97.5], axis=0)
            for condition in CONDITIONS
        }
        for model in MODELS
    }


def _plot_change_curves(means, bands, y_limits, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 5.8), sharex=True)
    for index, (metric, label) in enumerate(METRICS):
        ax = axes.flat[index]
        metric_index = next(i for i, (name, _) in enumerate(METRICS) if name == metric)
        for condition in CONDITIONS:
            mean = means[condition][:, metric_index]
            low = bands[condition][0, :, metric_index]
            high = bands[condition][1, :, metric_index]
            ax.plot(LAYERS, mean, marker="o", markersize=4.5, linewidth=1.8,
                    color=COLORS[condition], label=condition)
            ax.fill_between(LAYERS, low, high, color=COLORS[condition], alpha=0.20, linewidth=0)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.55)
        ax.set_title(label, fontsize=12)
        ax.set_xticks(LAYERS)
        ax.set_xlim(0.7, 12.3)
        ax.set_ylim(y_limits[metric])
        ax.set_xlabel("Layer")
        ax.tick_params(axis="x", labelsize=8)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path.relative_to(ROOT)}")


def _plot_base_finetuned_curves(means, bands, y_limits, output_path):
    """Plot base and finetuned profiles with the Figure 1 layout and metrics."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(14, 5.8), sharex=True)
    for index, (metric, label) in enumerate(METRICS):
        ax = axes.flat[index]
        metric_index = index
        for condition in CONDITIONS:
            for model in MODELS:
                mean = means[model][condition][:, metric_index]
                low = bands[model][condition][0, :, metric_index]
                high = bands[model][condition][1, :, metric_index]
                ax.plot(
                    LAYERS, mean, marker="o", markersize=4.5, linewidth=1.8,
                    color=COLORS[condition], linestyle=MODEL_LINESTYLES[model],
                )
                ax.fill_between(
                    LAYERS, low, high, color=COLORS[condition], alpha=0.17, linewidth=0,
                )
        ax.set_title(label, fontsize=12)
        ax.set_xticks(LAYERS)
        ax.set_xlim(0.7, 12.3)
        ax.set_ylim(y_limits[metric])
        ax.set_xlabel("Layer")
        ax.tick_params(axis="x", labelsize=8)

    condition_handles = [
        Line2D([], [], color=COLORS[condition], linewidth=2, label=condition)
        for condition in CONDITIONS
    ]
    model_handles = [
        Line2D([], [], color="#333333", linewidth=2, linestyle=MODEL_LINESTYLES[model], label=model)
        for model in MODELS
    ]
    fig.legend(condition_handles, CONDITIONS, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.37, 1.04), title="Condition")
    fig.legend(model_handles, MODELS, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.78, 1.04), title="Model")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path.relative_to(ROOT)}")


def generate_aggregate_attention_figures():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_runs = _load_all_run_attention_profiles()
    variants = {"combined": list(SPLITS)}
    summaries = {}
    for variant, split_names in variants.items():
        flattened_runs = {
            f"{split}:{run_name}": data
            for split in split_names
            for run_name, data in all_runs[split].items()
        }
        means = _aggregate_mean_change_curves(flattened_runs)
        rng = np.random.default_rng(BOOTSTRAP_SEED + sum(map(ord, variant)))
        bands = _bootstrap_change_curve_intervals(all_runs, split_names, rng)
        summaries[variant] = (means, bands)

    model_summaries = {}
    for variant, split_names in variants.items():
        flattened_runs = {
            f"{split}:{run_name}": data
            for split in split_names
            for run_name, data in all_runs[split].items()
        }
        means = _aggregate_mean_base_finetuned_curves(flattened_runs)
        rng = np.random.default_rng(BOOTSTRAP_SEED + sum(map(ord, variant)) + 1)
        bands = _bootstrap_base_finetuned_curve_intervals(all_runs, split_names, rng)
        model_summaries[variant] = (means, bands)

    # Keep each metric's y-scale identical across the combined and
    # split-specific versions for direct visual comparison.
    y_limits = {}
    for metric_index, (metric, _) in enumerate(METRICS):
        low, high = 0.0, 0.0
        for means, bands in summaries.values():
            for condition in CONDITIONS:
                low = min(low, means[condition][:, metric_index].min(),
                          bands[condition][0, :, metric_index].min())
                high = max(high, means[condition][:, metric_index].max(),
                           bands[condition][1, :, metric_index].max())
        pad = max((high - low) * 0.06, 1e-5)
        y_limits[metric] = (low - pad, high + pad)

    model_y_limits = {}
    for metric_index, (metric, _) in enumerate(METRICS):
        low, high = 0.0, 0.0
        for means, bands in model_summaries.values():
            for model in MODELS:
                for condition in CONDITIONS:
                    low = min(low, means[model][condition][:, metric_index].min(),
                              bands[model][condition][0, :, metric_index].min())
                    high = max(high, means[model][condition][:, metric_index].max(),
                               bands[model][condition][1, :, metric_index].max())
        pad = max((high - low) * 0.06, 1e-5)
        model_y_limits[metric] = (max(0.0, low - pad), high + pad)

    means, bands = summaries["combined"]
    _plot_change_curves(means, bands, y_limits, FIG_DIR / COMBINED_OUTPUT)

    means, bands = model_summaries["combined"]
    _plot_base_finetuned_curves(
        means, bands, model_y_limits, FIG_DIR / MODEL_COMBINED_OUTPUT
    )


if __name__ == "__main__":
    generate_aggregate_attention_figures()
