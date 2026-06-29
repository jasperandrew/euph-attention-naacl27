"""
Plotting functions for the single-sentence attention profiler.

Takes the aggregated DataFrame produced by profiler.analysis.profile_dataframe
and renders summary plots across layers and conditions.
"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

from .. import config
from ..plot_utils import _save_or_show

# Fixed hue→color mapping: with a named palette, seaborn assigns colors by
# order of first appearance in the data, which differs between eval splits
# (EN presents Euphemistic first, SP Non-Euphemistic first). Pinned to match
# the original EN figures: Euphemistic green, Non-Euphemistic orange.
_SET2 = sns.color_palette('Set2')
_LINE_PALETTE = {
    'Euphemistic': _SET2[0],
    'Non-Euphemistic': _SET2[1],
    'Ground Truth': _SET2[0],
    'Predicted': _SET2[1],
}


def _hue_order(df, hue_col):
    """Fixed hue order (alphabetical) so legend order matches across datasets."""
    return sorted(df[hue_col].unique())

_PROFILE_METRICS = [
    'Outward (PET → Context)',
    'Internal (PET → PET)',
    'Sink (PET → Sink)',
    'Inward (Context → PET)',
    'Entropy',
    'Distance',
]

_PROFILE_METRIC_LABELS = ['Outward', 'Internal', 'Sink', 'Inward', 'Entropy', 'Distance']

# Heads that are polyfunctional-and-leading at layer 11 (H3, H7, H11)
_POLY_HEADS = {3, 7, 11}


def _cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / s if s > 0 else 0.0


def _lighten(color, strength):
    """Blend color toward white by (1 − strength); equivalent to alpha=strength on white."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * (1 - strength),
            g + (1 - g) * (1 - strength),
            b + (1 - b) * (1 - strength))


def _sig_marker(p_bonf):
    if p_bonf < 1e-3:
        return '***'
    if p_bonf < 1e-2:
        return '**'
    if p_bonf < 5e-2:
        return '*'
    return ''


def _layer_range_label(agg_df):
    layers = sorted(agg_df['Layer'].unique())
    return f"Layers {layers[0]}–{layers[-1]}"


def _plot_profile_by(agg_df, hue_col, title_suffix, file_suffix, output_dir, title_prefix='Aggregate Attention Profile', zero_line=False):
    layers = sorted(agg_df['Layer'].unique())
    hue_order = _hue_order(agg_df, hue_col)

    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
    axes_flat = axes.flatten()

    for i, metric in enumerate(_PROFILE_METRICS):
        ax = axes_flat[i]
        sns.lineplot(data=agg_df, x='Layer', y=metric, hue=hue_col, hue_order=hue_order,
                     ax=ax, marker='o', palette=_LINE_PALETTE, legend=(i == 0))
        ax.set_title(metric, fontsize=11)
        ax.set_xticks(layers)
        ax.set_xlabel('Layer' if i >= 3 else '')
        ax.set_ylabel('')
        if zero_line:
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[0].get_legend().remove()
    fig.legend(handles=handles, labels=labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.97), ncol=len(handles),
               frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save_or_show(fig, output_dir, f'aggregate_profile_{file_suffix}.png')

    # Head variance is a layer-level metric duplicated across heads; drop duplicates
    # to avoid artificially narrowing the confidence interval in the lineplot.
    head_var_df = agg_df[['Layer', hue_col, 'Sample', 'Head Variance']].drop_duplicates()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=head_var_df, x='Layer', y='Head Variance', hue=hue_col, hue_order=hue_order,
                 marker='s', ax=ax2, palette=_LINE_PALETTE)
    ax2.set_title(f"Aggregate Head Specialization ({title_suffix})")
    ax2.set_xticks(layers)
    plt.tight_layout()
    _save_or_show(fig2, output_dir, f'aggregate_head_variance_{file_suffix}.png')


def _plot_profile_by_head(agg_df, hue_col, title_suffix, file_suffix, output_dir):
    """6-panel line plot of each metric by head, split by hue_col."""
    hue_order = _hue_order(agg_df, hue_col)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    fig.suptitle(f"Attention Profile by Head — {_layer_range_label(agg_df)} ({title_suffix})", fontsize=14)
    axes = axes.flatten()

    for i, metric in enumerate(_PROFILE_METRICS):
        sns.lineplot(data=agg_df, x='Head', y=metric, hue=hue_col, hue_order=hue_order,
                     ax=axes[i], marker='o', palette=_LINE_PALETTE)
        axes[i].set_title(metric, fontsize=11)
        axes[i].set_xticks(range(1, config.NUM_HEADS + 1))

    plt.tight_layout()
    _save_or_show(fig, output_dir, f'head_profile_{file_suffix}.png')


def _diff_pivot(df_pos, df_neg, metric):
    """Mean(metric | df_pos) − mean(metric | df_neg) as a Head×Layer pivot table."""
    pos = df_pos.groupby(['Layer', 'Head'])[metric].mean().unstack('Layer')
    neg = df_neg.groupby(['Layer', 'Head'])[metric].mean().unstack('Layer')
    return pos - neg


def _plot_metric_diff_grid(diffs, title, filename, output_dir):
    """6-panel Head×Layer heatmap grid from a list of (metric_name, pivot_df) pairs."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)
    axes = axes.flatten()

    for ax, (metric, diff) in zip(axes, diffs):
        max_abs = diff.abs().max().max()
        sns.heatmap(diff, cmap='RdBu_r', ax=ax, vmin=-max_abs, vmax=max_abs,
                    annot=True, fmt='.2f', annot_kws={'fontsize': 7},
                    cbar_kws={'label': 'Diff'}, square=False)
        ax.set_title(metric)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Head')

    plt.tight_layout()
    _save_or_show(fig, output_dir, filename)


def plot_metric_diff_heatmaps(agg_df, output_dir=None):
    """6-panel Layer×Head heatmap of Euph − Non-Euph mean difference for each profiler metric."""
    layer_label = _layer_range_label(agg_df)

    euph_label = agg_df[agg_df['Condition'] == 'Euphemistic']
    non_label = agg_df[agg_df['Condition'] == 'Non-Euphemistic']
    _plot_metric_diff_grid(
        [(m, _diff_pivot(euph_label, non_label, m)) for m in _PROFILE_METRICS],
        title=f"Attention Profile Difference (Euph − Non-Euph) by Layer × Head — Ground Truth ({layer_label})",
        filename="metric_diff_heatmaps_by_label.png",
        output_dir=output_dir,
    )

    if 'Predicted' in agg_df.columns:
        euph_pred = agg_df[agg_df['Predicted'] == 'Euphemistic']
        non_pred = agg_df[agg_df['Predicted'] == 'Non-Euphemistic']
        _plot_metric_diff_grid(
            [(m, _diff_pivot(euph_pred, non_pred, m)) for m in _PROFILE_METRICS],
            title=f"Attention Profile Difference (Euph − Non-Euph) by Layer × Head — Model Prediction ({layer_label})",
            filename="metric_diff_heatmaps_by_prediction.png",
            output_dir=output_dir,
        )


def plot_label_vs_pred_diff_heatmaps(agg_df, output_dir=None):
    """For each class, 6-panel heatmap of (Predicted=X) − (GroundTruth=X) per metric."""
    if 'Predicted' not in agg_df.columns:
        return

    layer_label = _layer_range_label(agg_df)
    for cls in ['Euphemistic', 'Non-Euphemistic']:
        df_label = agg_df[agg_df['Condition'] == cls]
        df_pred = agg_df[agg_df['Predicted'] == cls]
        diffs = [(m, _diff_pivot(df_pred, df_label, m)) for m in _PROFILE_METRICS]
        slug = cls.lower().replace('-', '').replace(' ', '_')
        _plot_metric_diff_grid(
            diffs,
            title=f"Attention Profile Difference (Predicted={cls} − GroundTruth={cls}) by Layer × Head ({layer_label})",
            filename=f"metric_diff_heatmaps_label_vs_pred_{slug}.png",
            output_dir=output_dir,
        )


def _plot_label_vs_prediction(agg_df, target_class, output_dir):
    """For one class, overlay the profile of ground-truth-positive vs predicted-positive samples."""
    df_label = agg_df[agg_df['Condition'] == target_class].copy()
    df_label['Source'] = 'Ground Truth'
    df_pred = agg_df[agg_df['Predicted'] == target_class].copy()
    df_pred['Source'] = 'Predicted'
    combined = pd.concat([df_label, df_pred], ignore_index=True)
    slug = target_class.lower().replace('-', '').replace(' ', '_')
    _plot_profile_by(combined, 'Source', f'{target_class}: Ground Truth vs Predicted', f'label_vs_pred_{slug}', output_dir)


def plot_base_vs_finetuned(ft_df, base_df, output_dir=None, unfrozen_only=False):
    """6-panel line chart overlaying base and finetuned raw profiles by condition.

    Color encodes ground-truth class; line style encodes model (solid=finetuned,
    dashed=base). Shaded bands are 95% bootstrapped CIs from seaborn.
    """
    ft = ft_df.copy()
    ft['Model'] = 'Finetuned'
    base = base_df.copy()
    base['Model'] = 'Base'
    combined = pd.concat([ft, base], ignore_index=True)
    if unfrozen_only:
        combined = combined[combined['Layer'] > config.FROZEN_LAYERS]

    layers = sorted(combined['Layer'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
    axes_flat = axes.flatten()

    for i, metric in enumerate(_PROFILE_METRICS):
        ax = axes_flat[i]
        sns.lineplot(
            data=combined, x='Layer', y=metric,
            hue='Condition', style='Model',
            hue_order=['Euphemistic', 'Non-Euphemistic'],
            style_order=['Finetuned', 'Base'],
            palette=_LINE_PALETTE,
            ax=ax, legend=(i == 0),
        )
        ax.set_title(_PROFILE_METRIC_LABELS[i], fontsize=11)
        ax.set_xticks(layers)
        ax.set_xlabel('Layer' if i >= 3 else '')
        ax.set_ylabel('')

    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[0].get_legend().remove()
    fig.legend(handles=handles, labels=labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.97), ncol=len(handles),
               frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save_or_show(fig, output_dir, 'aggregate_profile_base_vs_ft_by_label.png')


def plot_finetuning_diff_profile(diff_df, output_dir=None, unfrozen_only=False):
    """Plot per-condition profiles of (finetuned − base) metric values.

    Expects diff_df to have the same columns as profile_dataframe output, where each
    metric value is already the row-wise difference ft − base.
    """
    if unfrozen_only:
        diff_df = diff_df[diff_df['Layer'] > config.FROZEN_LAYERS]
    _plot_profile_by(
        diff_df, 'Condition',
        title_suffix='Finetuned − Base, by Ground Truth',
        file_suffix='ft_minus_base_by_label',
        output_dir=output_dir,
        title_prefix='Finetuning-Induced Attention Change',
        zero_line=True,
    )


def plot_layer11_lollipop(df, output_dir=None):
    """6-panel per-head Cohen's d lollipop at layer 11 (Euph vs Non-Euph).

    Color encodes direction (red=positive, blue=negative); intensity encodes
    significance after Bonferroni correction. Polyfunctional-and-leading heads
    are outlined and bold-labeled.
    """
    df11 = df[df['Layer'] == 11]
    n_tests = len(_PROFILE_METRICS) * config.NUM_HEADS

    d_grid = np.zeros((len(_PROFILE_METRICS), config.NUM_HEADS))
    p_grid = np.zeros_like(d_grid)
    for i, m in enumerate(_PROFILE_METRICS):
        for j, head in enumerate(range(1, config.NUM_HEADS + 1)):
            sub = df11[df11['Head'] == head]
            e = sub[sub['Condition'] == 'Euphemistic'][m].values
            n = sub[sub['Condition'] == 'Non-Euphemistic'][m].values
            _, p = scipy.stats.mannwhitneyu(e, n, alternative='two-sided')
            d_grid[i, j] = _cohens_d(e, n)
            p_grid[i, j] = min(p * n_tests, 1.0)

    pos_color, neg_color = '#c0392b', '#2471a3'
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
    axes = axes.flatten()

    for ax, (i, m_label) in zip(axes, enumerate(_PROFILE_METRIC_LABELS)):
        ds, p_bs = d_grid[i], p_grid[i]
        ys = np.arange(config.NUM_HEADS)
        for y, d, p_b in zip(ys, ds, p_bs):
            base = pos_color if d > 0 else neg_color
            strength = 1.0 if p_b < 1e-3 else 0.6 if p_b < 5e-2 else 0.25
            color = _lighten(base, strength)
            ax.plot([0, d], [y, y], color=color, lw=2.2, solid_capstyle='butt')
            ax.plot(d, y, 'o', color=color, ms=7,
                    markeredgecolor='black' if (y + 1) in _POLY_HEADS else 'none',
                    markeredgewidth=1.2)
        ax.axvline(0, color='black', lw=0.6)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"H{h}" for h in range(1, config.NUM_HEADS + 1)])
        for tick in ax.get_yticklabels():
            if int(tick.get_text()[1:]) in _POLY_HEADS:
                tick.set_fontweight('bold')
        ax.set_title(m_label, fontsize=11)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(-1.15, 1.15)
        ax.set_axisbelow(True)

    for ax in axes[3:]:
        ax.set_xlabel("Cohen's $d$")
    plt.tight_layout()
    _save_or_show(fig, output_dir, 'layer11_per_head_d_lollipop.png')


def plot_correct_vs_misclassified(df, output_dir=None):
    """Grouped bar chart of Cohen's d at layer 11, split by classification correctness.

    Requires df to have a 'Predicted' column (produced when the profiler is run
    with predictions). Shows how the layer-11 effect sizes collapse and sign-flip
    for misclassified samples relative to correctly classified ones.
    """
    if 'Predicted' not in df.columns:
        raise ValueError("DataFrame must contain a 'Predicted' column.")

    l11 = df[df['Layer'] == 11].copy()
    l11['correct'] = l11['Condition'] == l11['Predicted']
    correct = l11[l11['correct']]
    wrong = l11[~l11['correct']]

    d_correct, d_wrong = [], []
    for m in _PROFILE_METRICS:
        e_c = correct[correct['Condition'] == 'Euphemistic'][m].values
        n_c = correct[correct['Condition'] == 'Non-Euphemistic'][m].values
        e_w = wrong[wrong['Condition'] == 'Euphemistic'][m].values
        n_w = wrong[wrong['Condition'] == 'Non-Euphemistic'][m].values
        d_correct.append(_cohens_d(e_c, n_c))
        d_wrong.append(_cohens_d(e_w, n_w) if len(e_w) > 1 and len(n_w) > 1 else 0.0)

    x = np.arange(len(_PROFILE_METRIC_LABELS))
    width = 0.32
    cor_color, mis_color = '#2471a3', '#c0392b'

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.bar(x - width / 2, d_correct, width, label='Correct', color=cor_color, alpha=0.85)
    ax.bar(x + width / 2, d_wrong, width, label='Misclassified', color=mis_color, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(_PROFILE_METRIC_LABELS, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel("Cohen's $d$", fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_or_show(fig, output_dir, 'layer11_correct_vs_misclassified.png', dpi=300)


def plot_real_vs_masked(ft_df, mask_df, output_dir=None):
    """Grouped bar chart of Cohen's d at layer 11: real PET vs masked PET.

    Shows which metrics survive PET-token masking (position-driven) vs collapse
    (content-driven). Both DataFrames must have a 'Sample' column for per-sample collapse.
    """
    def _per_sample_d(df):
        ps = df[df['Layer'] == 11].groupby(['Sample', 'Condition'])[_PROFILE_METRICS].mean().reset_index()
        ds = []
        for m in _PROFILE_METRICS:
            e = ps[ps['Condition'] == 'Euphemistic'][m].values
            n = ps[ps['Condition'] == 'Non-Euphemistic'][m].values
            ds.append(_cohens_d(e, n))
        return ds

    d_real = _per_sample_d(ft_df)
    d_mask = _per_sample_d(mask_df)

    x = np.arange(len(_PROFILE_METRIC_LABELS))
    width = 0.32
    real_color, mask_color = '#2471a3', '#e67e22'

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.bar(x - width / 2, d_real, width, label='Real PET', color=real_color, alpha=0.85)
    ax.bar(x + width / 2, d_mask, width, label='Masked PET', color=mask_color, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(_PROFILE_METRIC_LABELS, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel("Cohen's $d$", fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_or_show(fig, output_dir, 'layer11_real_vs_masked.png', dpi=300)


def plot_aggregate_profiles(agg_df, output_dir=None, unfrozen_only=False):
    """Plots attention metrics by layer and by head, split by ground truth and prediction.

    Args:
        unfrozen_only: if True, restrict all plots to layers > config.FROZEN_LAYERS.
    """
    if unfrozen_only:
        agg_df = agg_df[agg_df['Layer'] > config.FROZEN_LAYERS]

    _plot_profile_by(agg_df, 'Condition', 'by Ground Truth', 'by_label', output_dir)
    if 'Predicted' in agg_df.columns:
        _plot_profile_by(agg_df, 'Predicted', 'by Model Prediction', 'by_prediction', output_dir)
        for cls in agg_df['Condition'].unique():
            _plot_label_vs_prediction(agg_df, cls, output_dir)

    plot_metric_diff_heatmaps(agg_df, output_dir=output_dir)
    plot_label_vs_pred_diff_heatmaps(agg_df, output_dir=output_dir)

    _plot_profile_by_head(agg_df, 'Condition', 'by Ground Truth', 'by_label', output_dir)
    if 'Predicted' in agg_df.columns:
        _plot_profile_by_head(agg_df, 'Predicted', 'by Model Prediction', 'by_prediction', output_dir)
        for cls in agg_df['Condition'].unique():
            df_label = agg_df[agg_df['Condition'] == cls].copy()
            df_label['Source'] = 'Ground Truth'
            df_pred = agg_df[agg_df['Predicted'] == cls].copy()
            df_pred['Source'] = 'Predicted'
            slug = cls.lower().replace('-', '').replace(' ', '_')
            _plot_profile_by_head(pd.concat([df_label, df_pred], ignore_index=True),
                                  'Source', f'{cls}: Ground Truth vs Predicted',
                                  f'label_vs_pred_{slug}', output_dir)
