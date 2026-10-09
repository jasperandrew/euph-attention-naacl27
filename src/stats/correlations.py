"""Length-attention correlation summaries for sample-level profiles."""

import numpy as np
import scipy.stats


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p-values, preserving input order."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p_values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p_values[valid])]
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def context_length_correlations(per_sample, lengths):
    """Correlate context length with per-sample, per-layer inward/outward attention.

    ``per_sample`` must contain one row per sample/layer, with metric values
    already averaged over heads. ``lengths`` must align with sample IDs and
    provide a ``context_len`` column.
    """
    lengths = lengths.reset_index(drop=True).copy()
    lengths['sample_id'] = lengths.index
    samples = per_sample.merge(
        lengths[['sample_id', 'context_len']], on='sample_id', validate='many_to_one',
    )
    metrics = ('Inward (Context → PET)', 'Outward (PET → Context)')
    layers = sorted(int(layer) for layer in per_sample['Layer'].unique())
    results = []
    for condition in ('Euphemistic', 'Non-Euphemistic'):
        condition_df = samples[samples['Condition'] == condition]
        for layer in layers:
            layer_df = condition_df[condition_df['Layer'] == layer]
            for metric in metrics:
                x = layer_df['context_len'].to_numpy(dtype=float)
                y = layer_df[metric].to_numpy(dtype=float)
                rho, p_value = np.nan, np.nan
                if len(x) >= 3 and np.unique(x).size > 1 and np.unique(y).size > 1:
                    rho, p_value = scipy.stats.spearmanr(x, y)
                results.append({
                    'condition': condition,
                    'layer': layer,
                    'metric': metric,
                    'n': int(len(x)),
                    'spearman_rho': float(rho) if np.isfinite(rho) else None,
                    'p_raw': float(p_value) if np.isfinite(p_value) else None,
                })

    adjusted = benjamini_hochberg([
        result['p_raw'] if result['p_raw'] is not None else np.nan
        for result in results
    ])
    for result, p_adjusted in zip(results, adjusted):
        result['p_bh'] = float(p_adjusted) if np.isfinite(p_adjusted) else None

    return {
        'method': 'Spearman rank correlation',
        'context_length_definition': 'non-PET, non-sink tokens',
        'attention_aggregation': 'mean across heads for each sample and layer',
        'multiple_testing_correction': 'Benjamini-Hochberg across all condition × layer × metric tests',
        'results': results,
    }
