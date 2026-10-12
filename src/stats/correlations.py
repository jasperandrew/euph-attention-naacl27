"""Length-attention correlation summaries for sample-level profiles."""

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression


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
    metrics = tuple(metric for metric in (
        'Inward (Context → PET)', 'Outward (PET → Context)',
        'Sink (Sink → PET)', 'All (All → PET)',
    ) if metric in samples.columns)
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


def context_length_adjusted_attention(per_sample, lengths):
    """Return per-sample attention values adjusted to a common context length.

    For every layer and metric, fit a pooled linear model with a shared context-
    length slope and a condition intercept. Subtracting the fitted length term
    centers each sample at the run's mean context length while retaining the
    condition contrast. The raw measures are retained alongside adjusted values.
    """
    lengths = lengths.reset_index(drop=True).copy()
    lengths['sample_id'] = lengths.index
    samples = per_sample.merge(
        lengths[['sample_id', 'context_len']], on='sample_id', validate='many_to_one',
    )
    metrics = tuple(metric for metric in (
        'Inward (Context → PET)', 'Outward (PET → Context)',
        'Sink (Sink → PET)', 'All (All → PET)',
    ) if metric in samples.columns)
    results = []
    datapoints = []
    for layer in sorted(int(value) for value in samples['Layer'].unique()):
        layer_df = samples[samples['Layer'] == layer].copy()
        x = layer_df['context_len'].to_numpy(dtype=float)
        reference_length = float(np.mean(x)) if len(x) else None
        centered_x = x - reference_length if len(x) else x
        condition = (layer_df['Condition'] == 'Euphemistic').to_numpy(dtype=float)
        predictors = np.column_stack([condition, centered_x])
        for metric in metrics:
            y = layer_df[metric].to_numpy(dtype=float)
            finite = np.isfinite(y) & np.isfinite(centered_x)
            beta = None
            adjusted = np.full(len(y), np.nan)
            if finite.sum() >= 3 and np.unique(centered_x[finite]).size > 1:
                regression = LinearRegression().fit(predictors[finite], y[finite])
                beta = float(regression.coef_[1])
                adjusted[finite] = y[finite] - beta * centered_x[finite]
            for (_, row), raw_value, adjusted_value in zip(
                    layer_df.iterrows(), y, adjusted):
                datapoints.append({
                    'sample_id': int(row['sample_id']),
                    'condition': row['Condition'],
                    'layer': layer,
                    'context_len': int(row['context_len']),
                    'metric': metric,
                    'raw_value': float(raw_value) if np.isfinite(raw_value) else None,
                    'length_adjusted_value': (
                        float(adjusted_value) if np.isfinite(adjusted_value) else None
                    ),
                })
            for cond in ('Euphemistic', 'Non-Euphemistic'):
                mask = (layer_df['Condition'].to_numpy() == cond) & np.isfinite(adjusted)
                values = adjusted[mask]
                results.append({
                    'condition': cond,
                    'layer': layer,
                    'metric': metric,
                    'n': int(mask.sum()),
                    'mean_raw': float(np.mean(y[mask])) if mask.any() else None,
                    'mean_length_adjusted': float(np.mean(values)) if len(values) else None,
                    'reference_context_len': reference_length,
                    'context_length_slope': beta,
                })

    return {
        'method': 'Pooled linear context-length adjustment with condition intercepts',
        'adjustment': 'value - shared_context_length_slope * (context_len - run_mean_context_len)',
        'reference': 'run mean context length',
        'metrics': list(metrics),
        'results': results,
        'data_points': datapoints,
    }
