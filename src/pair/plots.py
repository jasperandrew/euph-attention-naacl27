"""
Plotting functions that operate on euph/non-euph sentence pairs.

Every function here takes both a non-euphemistic and a euphemistic sentence
(pre-tokenized) and visualises some difference between them.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .. import config
from ..attention_utils import (
    get_attn_weights, get_attn_head_weights,
    _find_unbounded_pet, get_pet_attention, get_non_pet_attention,
    get_bound_attention, get_attention_entropy,
)
from ..plot_utils import _save_or_show, _plot_attn_on_axis
from .analysis import compute_sink_diff_for_sample


def plot_euph_attention_analysis(model, layer_i, head_i, pet_tokens,
                                  tokens_n, tokenized_n, tokens_y, tokenized_y,
                                  attn_n=None, attn_y=None,
                                  output_dir=None):
    """Three-panel: non-euph | euph | euph-minus-non-euph diff (full sequence)."""
    if attn_n is None:
        attn_n = get_attn_weights(model, tokenized_n)
    if attn_y is None:
        attn_y = get_attn_weights(model, tokenized_y)
    head_n = get_attn_head_weights(attn_n, layer_i, head_i, len(tokens_n))
    head_y = get_attn_head_weights(attn_y, layer_i, head_i, len(tokens_y))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Finetuned Euph vs. Non-Euph Attention Analysis (Layer {layer_i+1}, Head {head_i+1})', fontsize=16)

    _plot_attn_on_axis(axes[0], head_n, tokens_n, 'Non-Euph Attn')
    _plot_attn_on_axis(axes[1], head_y, tokens_y, 'Euph Attn')

    if len(tokens_n) == len(tokens_y):
        n_start, n_end = _find_unbounded_pet(tokens_n, pet_tokens)
        y_start, y_end = _find_unbounded_pet(tokens_y, pet_tokens)

        if n_start is not None and y_start is not None and n_start == y_start and n_end == y_end:
            display_tokens = []
            for i in range(len(tokens_n)):
                tn = tokens_n[i].replace(config.TOK_DELIM, '')
                ty = tokens_y[i].replace(config.TOK_DELIM, '')
                display_tokens.append(tn if tn == ty else f"{tn}|{ty}")

            diff = head_y - head_n
            max_abs = np.max(np.abs(diff))
            sns.heatmap(
                diff, cmap='RdBu_r',
                xticklabels=display_tokens, yticklabels=display_tokens,
                ax=axes[2], square=True, cbar_kws={'shrink': 0.8},
                vmin=-max_abs, vmax=max_abs,
            )
            axes[2].set_title('1-to-1 Attn Diff (Euph - Non-Euph)', fontsize=10)
        else:
            axes[2].set_title('1-to-1 Attn Diff (PET Mismatch)', fontsize=10)
            axes[2].axis('off')
    else:
        axes[2].set_title('1-to-1 Attn Diff (Length Mismatch)', fontsize=10)
        axes[2].axis('off')

    plt.tight_layout()
    filename = f"euph_attn_analysis_L{layer_i+1}_H{head_i+1}.png" if output_dir else None
    _save_or_show(fig, output_dir, filename)


def plot_finetune_attn_diff(og_model, ft_model, layer_i, head_i,
                             tokens_n, tokenized_n, tokens_y, tokenized_y,
                             og_attn_n=None, og_attn_y=None, ft_attn_n=None, ft_attn_y=None,
                             output_dir=None):
    """Two-panel: base-minus-finetuned attention diff for non-euph and euph sentences."""
    if og_attn_n is None:
        og_attn_n = get_attn_weights(og_model, tokenized_n)
    if og_attn_y is None:
        og_attn_y = get_attn_weights(og_model, tokenized_y)
    if ft_attn_n is None:
        ft_attn_n = get_attn_weights(ft_model, tokenized_n)
    if ft_attn_y is None:
        ft_attn_y = get_attn_weights(ft_model, tokenized_y)
    og_head_n = get_attn_head_weights(og_attn_n, layer_i, head_i, len(tokens_n))
    ft_head_n = get_attn_head_weights(ft_attn_n, layer_i, head_i, len(tokens_n))
    og_head_y = get_attn_head_weights(og_attn_y, layer_i, head_i, len(tokens_y))
    ft_head_y = get_attn_head_weights(ft_attn_y, layer_i, head_i, len(tokens_y))

    diff_n = og_head_n - ft_head_n
    diff_y = og_head_y - ft_head_y
    max_abs = np.max(np.abs([diff_n, diff_y]))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'Finetuned Attention Difference (Layer {layer_i+1}, Head {head_i+1})')
    _plot_attn_on_axis(axes[0], diff_n, tokens_n, 'Non-Euphemistic', cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    _plot_attn_on_axis(axes[1], diff_y, tokens_y, 'Euphemistic', cmap='RdBu_r', xlabel=None, ylabel=None, vmin=-max_abs, vmax=max_abs)

    plt.tight_layout()
    filename = f"finetune_attn_diff_layer{layer_i+1}_head{head_i+1}.png" if output_dir else None
    _save_or_show(fig, output_dir, filename)


def plot_avg_diff(og_model, ft_model, pet_tokens,
                  tokens_n, tokenized_n, tokens_y, tokenized_y,
                  example_type=None, token_type=None, hue_category='Token Type',
                  output_dir=None):
    """Bar chart of mean base-vs-finetuned attention diff per non-frozen layer, by token type."""
    og_attn_n = get_attn_weights(og_model, tokenized_n)
    og_attn_y = get_attn_weights(og_model, tokenized_y)
    ft_attn_n = get_attn_weights(ft_model, tokenized_n)
    ft_attn_y = get_attn_weights(ft_model, tokenized_y)

    pet_indices_n = _find_unbounded_pet(tokens_n, pet_tokens)
    pet_indices_y = _find_unbounded_pet(tokens_y, pet_tokens)

    layer_avg_diffs = []

    for layer_i in range(config.FROZEN_LAYERS, config.NUM_LAYERS):
        pet_diffs_n, non_pet_diffs_n, bound_diffs_n = [], [], []
        pet_diffs_y, non_pet_diffs_y, bound_diffs_y = [], [], []

        for head_i in range(config.NUM_HEADS):
            og_head_n = get_attn_head_weights(og_attn_n, layer_i, head_i, len(tokens_n))
            ft_head_n = get_attn_head_weights(ft_attn_n, layer_i, head_i, len(tokens_n))
            og_head_y = get_attn_head_weights(og_attn_y, layer_i, head_i, len(tokens_y))
            ft_head_y = get_attn_head_weights(ft_attn_y, layer_i, head_i, len(tokens_y))

            og_pet_n, _ = get_pet_attention(og_head_n, tokens_n, pet_indices_n)
            ft_pet_n, _ = get_pet_attention(ft_head_n, tokens_n, pet_indices_n)
            if og_pet_n is not None and ft_pet_n is not None:
                pet_diffs_n.append(np.mean(og_pet_n - ft_pet_n))

            og_pet_y, _ = get_pet_attention(og_head_y, tokens_y, pet_indices_y)
            ft_pet_y, _ = get_pet_attention(ft_head_y, tokens_y, pet_indices_y)
            if og_pet_y is not None and ft_pet_y is not None:
                pet_diffs_y.append(np.mean(og_pet_y - ft_pet_y))

            og_non_n, _ = get_non_pet_attention(og_head_n, tokens_n, pet_indices_n)
            ft_non_n, _ = get_non_pet_attention(ft_head_n, tokens_n, pet_indices_n)
            if og_non_n and ft_non_n and len(og_non_n) == len(ft_non_n):
                seg_diffs = [np.mean(o - f) for o, f in zip(og_non_n, ft_non_n)]
                if seg_diffs:
                    non_pet_diffs_n.append(np.mean(seg_diffs))

            og_non_y, _ = get_non_pet_attention(og_head_y, tokens_y, pet_indices_y)
            ft_non_y, _ = get_non_pet_attention(ft_head_y, tokens_y, pet_indices_y)
            if og_non_y and ft_non_y and len(og_non_y) == len(ft_non_y):
                seg_diffs = [np.mean(o - f) for o, f in zip(og_non_y, ft_non_y)]
                if seg_diffs:
                    non_pet_diffs_y.append(np.mean(seg_diffs))

            if config.BOUNDED:
                og_b_n, _ = get_bound_attention(og_head_n, tokens_n)
                ft_b_n, _ = get_bound_attention(ft_head_n, tokens_n)
                if og_b_n is not None and ft_b_n is not None:
                    bound_diffs_n.append(np.mean(og_b_n - ft_b_n))

                og_b_y, _ = get_bound_attention(og_head_y, tokens_y)
                ft_b_y, _ = get_bound_attention(ft_head_y, tokens_y)
                if og_b_y is not None and ft_b_y is not None:
                    bound_diffs_y.append(np.mean(og_b_y - ft_b_y))

        def _append(diffs, example_label, token_label):
            if diffs:
                layer_avg_diffs.append({
                    'Layer': layer_i + 1,
                    'Example Type': example_label,
                    'Token Type': token_label,
                    'Average_Difference': np.mean(diffs),
                })

        _append(pet_diffs_n, 'Non-Euphemistic', 'PET')
        _append(non_pet_diffs_n, 'Non-Euphemistic', 'Non-PET')
        _append(bound_diffs_n, 'Non-Euphemistic', 'Boundary')
        _append(pet_diffs_y, 'Euphemistic', 'PET')
        _append(non_pet_diffs_y, 'Euphemistic', 'Non-PET')
        _append(bound_diffs_y, 'Euphemistic', 'Boundary')

    df_diffs = pd.DataFrame(layer_avg_diffs)
    filtered = df_diffs.copy()
    if example_type:
        filtered = filtered[filtered['Example Type'] == example_type]
    if token_type:
        filtered = filtered[filtered['Token Type'] == token_type]

    title_parts = ['Average Attention Difference per Non-Frozen Layer']
    if example_type:
        title_parts.append(f'({example_type} Example)')
    if token_type:
        title_parts.append(f'({token_type} Tokens)')
    if not example_type and not token_type:
        title_parts.append('(All Examples and Token Types)')

    fig = plt.figure(figsize=(8, 3))
    sns.barplot(x='Layer', y='Average_Difference', hue=hue_category, data=filtered, palette='viridis')
    plt.title(' '.join(title_parts))
    plt.xlabel('Layer Number')
    plt.ylabel('Average Difference')
    plt.legend(title=hue_category)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        parts = ['avg_diff']
        if example_type:
            parts.append(example_type.replace(' ', '_'))
        if token_type:
            parts.append(token_type)
        filename = '_'.join(parts).lower() + '.png'
        fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_entropy_by_layer(model, pet_tokens,
                           tokens_n, tokenized_n, tokens_y, tokenized_y,
                           label="Finetuned", compare_model=None, compare_label="Base",
                           attn_n=None, attn_y=None,
                           output_dir=None):
    """Line plot of mean PET entropy per layer, split by euph/non-euph condition."""
    if attn_y is None:
        attn_y = get_attn_weights(model, tokenized_y)
    if attn_n is None:
        attn_n = get_attn_weights(model, tokenized_n)
    df_y = get_attention_entropy(attn_y, tokens_y, pet_tokens, model_label=f"{label} / Euphemistic")
    df_n = get_attention_entropy(attn_n, tokens_n, pet_tokens, model_label=f"{label} / Non-Euphemistic")
    dfs = [df_y, df_n]

    if compare_model is not None:
        attn_y_cmp = get_attn_weights(compare_model, tokenized_y)
        attn_n_cmp = get_attn_weights(compare_model, tokenized_n)
        dfs.append(get_attention_entropy(attn_y_cmp, tokens_y, pet_tokens, model_label=f"{compare_label} / Euphemistic"))
        dfs.append(get_attention_entropy(attn_n_cmp, tokens_n, pet_tokens, model_label=f"{compare_label} / Non-Euphemistic"))

    df_all = pd.concat(dfs)
    df_mean = df_all.groupby(['Layer', 'Condition'])['Entropy'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df_mean, x='Layer', y='Entropy', hue='Condition', marker='o', ax=ax)
    ax.set_title('Mean PET Token Attention Entropy by Layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Entropy (nats)')
    ax.legend(title='Condition', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()

    _save_or_show(fig, output_dir, 'pet_entropy_by_layer.png')
    return df_all


def plot_entropy_diff_heatmap(model, pet_tokens,
                               tokens_n, tokenized_n, tokens_y, tokenized_y,
                               label="Finetuned", attn_n=None, attn_y=None,
                               output_dir=None):
    """Layer×head heatmap of PET entropy difference (euph minus non-euph)."""
    if attn_y is None:
        attn_y = get_attn_weights(model, tokenized_y)
    if attn_n is None:
        attn_n = get_attn_weights(model, tokenized_n)
    df_y = get_attention_entropy(attn_y, tokens_y, pet_tokens, model_label=label)
    df_n = get_attention_entropy(attn_n, tokens_n, pet_tokens, model_label=label)

    if df_y is None or df_n is None:
        print("PET tokens not found — skipping entropy diff heatmap.")
        return

    pivot_y = df_y.groupby(['Layer', 'Head'])['Entropy'].mean().unstack('Head')
    pivot_n = df_n.groupby(['Layer', 'Head'])['Entropy'].mean().unstack('Head')
    diff_pivot = pivot_y - pivot_n

    fig, ax = plt.subplots(figsize=(6, 5))
    max_abs = np.max(np.abs(diff_pivot.values))
    sns.heatmap(diff_pivot, cmap='RdBu_r', ax=ax, vmin=-max_abs, vmax=max_abs,
                cbar_kws={'label': 'Entropy Diff (Euph - Non-Euph) (nats)'})
    ax.set_title(f'PET Attention Entropy Difference Heatmap (Euph - Non-Euph) — {label}')
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    plt.tight_layout()

    _save_or_show(fig, output_dir, f'entropy_diff_heatmap_{label.lower()}.png')


def plot_all_sink_diffs(model, samples, output_dir=None):
    """One heatmap panel per sample showing PET sink attention diff (euph minus non-euph)."""
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    summary = []
    diffs_by_phrase = {}
    for ax, sample in zip(axes, samples):
        pet = sample['pet_phrase']
        label = pet if isinstance(pet, str) else ' / '.join(pet).replace(config.TOK_DELIM, '')
        print(f"Processing: {label}")
        result = compute_sink_diff_for_sample(model, sample)
        diffs_by_phrase[pet] = result
        if result is None:
            ax.set_title(f'{label}\n(not found)')
            ax.axis('off')
            continue

        diff, scalar_diff = result
        summary.append({'PET': label, 'Mean Sink Diff': scalar_diff})
        print(f"  Mean sink diff (Euph - Non-Euph): {scalar_diff:+.4f}")

        pivot = diff.pivot(index='Layer', columns='Head', values='Sink (PET → Sink)')
        max_abs = pivot.abs().max().max()
        sns.heatmap(pivot, cmap='RdBu_r', center=0, vmin=-max_abs, vmax=max_abs,
                    ax=ax, cbar_kws={'label': 'Sink Diff'}, square=False)
        ax.set_title(f'"{label}"\n{scalar_diff:+.4f} mean diff')
        ax.set_xlabel('Head')
        ax.set_ylabel('Layer' if ax == axes[0] else '')

    fig.suptitle('PET Sink Attention Difference (Euph - Non-Euph) by Sample', fontsize=13, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, output_dir, 'sink_attn_all_samples.png')

    summary_df = pd.DataFrame(summary)
    print("\n--- Summary ---")
    print(summary_df.to_string(index=False))
    return summary_df, diffs_by_phrase
