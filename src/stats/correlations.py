"""Length-attention correlation summaries for sample-level profiles."""

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression


LENGTH_ADJUSTED_METRICS = (
    'Inward (Context → PET)', 'Outward (PET → Context)',
    'Internal (PET → PET)', 'Sink (Sink → PET)', 'All (All → PET)',
)
FULL_ATTENTION_TEST_FAMILY_SIZE = 8 * 12


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
    metrics = tuple(metric for metric in LENGTH_ADJUSTED_METRICS
                    if metric in samples.columns)
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
    metrics = tuple(metric for metric in LENGTH_ADJUSTED_METRICS
                    if metric in samples.columns)
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

    adjustment = {
        'method': 'Pooled linear context-length adjustment with condition intercepts',
        'adjustment': 'value - shared_context_length_slope * (context_len - run_mean_context_len)',
        'reference': 'run mean context length',
        'metrics': list(metrics),
        'multiple_testing_correction': (
            'Bonferroni across the existing 8-metric × 12-layer family '
            f'({FULL_ATTENTION_TEST_FAMILY_SIZE} tests per run)'
        ),
        'results': results,
        'data_points': datapoints,
    }
    adjustment['class_effect_tests'] = context_length_adjusted_tests(
        datapoints, bonferroni_tests=FULL_ATTENTION_TEST_FAMILY_SIZE,
    )
    return adjustment


def _length_adjusted_class_test(values, conditions, context_lengths,
                                bonferroni_tests):
    """Test the class coefficient in attention ~ class + context length.

    Uses an HC3 heteroskedasticity-robust OLS standard error. This tests the
    adjusted class contrast while retaining the common context-length slope
    used by ``context_length_adjusted_attention``.
    """
    y = np.asarray(values, dtype=float)
    condition = np.asarray(conditions, dtype=float)
    context = np.asarray(context_lengths, dtype=float)
    finite = np.isfinite(y) & np.isfinite(condition) & np.isfinite(context)
    y, condition, context = y[finite], condition[finite], context[finite]
    n = len(y)
    if (n <= 3 or np.unique(condition).size != 2 or
            np.unique(context).size < 2):
        return None

    centered_context = context - context.mean()
    design = np.column_stack((np.ones(n), condition, centered_context))
    try:
        bread = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return None
    coefficients = bread @ design.T @ y
    residuals = y - design @ coefficients
    leverage = np.einsum('ij,jk,ik->i', design, bread, design)
    if np.any(1.0 - leverage <= np.finfo(float).eps):
        return None

    hc3_residuals = residuals / (1.0 - leverage)
    scores = design * hc3_residuals[:, None]
    covariance = bread @ (scores.T @ scores) @ bread
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    coefficient = float(coefficients[1])
    if not np.isfinite(standard_error) or standard_error <= 0:
        p_raw = 1.0 if coefficient == 0 else 0.0
        statistic = None
    else:
        statistic = coefficient / standard_error
        p_raw = float(2.0 * scipy.stats.t.sf(abs(statistic), df=n - 3))
    p_bonferroni = min(p_raw * bonferroni_tests, 1.0)
    return {
        'n_euphemistic': int(np.sum(condition == 1)),
        'n_non_euphemistic': int(np.sum(condition == 0)),
        'adjusted_mean_difference': coefficient,
        'hc3_standard_error': standard_error,
        't_statistic': float(statistic) if statistic is not None else None,
        'degrees_of_freedom': n - 3,
        'p_raw': p_raw,
        'p_bonferroni': p_bonferroni,
        'significant_bonferroni_0_05': p_bonferroni < 0.05,
    }


def context_length_adjusted_tests(data_points, bonferroni_tests=FULL_ATTENTION_TEST_FAMILY_SIZE):
    """Test per-layer adjusted class differences from saved adjustment points."""
    points = pd.DataFrame(data_points)
    if points.empty:
        return []
    results = []
    for (layer, metric), group in points.groupby(['layer', 'metric'], sort=True):
        conditions = (group['condition'] == 'Euphemistic').to_numpy(dtype=float)
        test = _length_adjusted_class_test(
            group['raw_value'].to_numpy(dtype=float), conditions,
            group['context_len'].to_numpy(dtype=float), bonferroni_tests,
        )
        if test is not None:
            results.append({
                'layer': int(layer), 'metric': metric,
                'test': 'HC3 robust OLS class coefficient; attention ~ class + context length',
                'n_bonferroni_tests': int(bonferroni_tests),
                **test,
            })
    return results
