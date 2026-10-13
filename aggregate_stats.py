"""Aggregate generated per-run evaluation and statistical results."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from src import config
from src.stats.profiling import METRIC_COLUMNS
from src.stats.correlations import (
    FULL_ATTENTION_TEST_FAMILY_SIZE,
    LENGTH_ADJUSTED_METRICS,
    context_length_adjusted_attention,
    context_length_adjusted_tests,
    context_length_correlations,
    pet_context_length_adjusted_attention,
)

logger = logging.getLogger(__name__)


METRICS = METRIC_COLUMNS
PROFILE_KEYS = ('real_pet', 'masked_pet', 'random_span', 'base_real_pet')


def read_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled_var = ((len(a) - 1) * a.var(ddof=1) +
                  (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var)) if pooled_var > 0 else 0.0


def collapse_profile(rows):
    """Match sanity_checks.to_per_sample(): average heads per sample/layer."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    n_layers, n_heads = int(df['Layer'].max()), int(df['Head'].max())
    block_size = n_layers * n_heads
    if len(df) % block_size:
        raise ValueError(f"Profile row count {len(df)} is not divisible by {block_size}")
    df['sample_id'] = np.arange(len(df)) // block_size
    group_cols = ['sample_id', 'Layer', 'Condition']
    if 'Predicted' in df.columns:
        group_cols.append('Predicted')
    available_metrics = [metric for metric in METRICS if metric in df.columns]
    return df.groupby(group_cols, sort=False)[available_metrics].mean().reset_index()


def effect(a, b, bonferroni_tests=None):
    result = {'n_euphemistic': len(a), 'n_non_euphemistic': len(b),
              'cohens_d': cohens_d(a, b)}
    if a and b:
        p = float(mannwhitneyu(a, b, alternative='two-sided').pvalue)
        result['p_raw'] = p
        result['p_bonferroni'] = min(p * bonferroni_tests, 1.0) if bonferroni_tests else p
    else:
        result.update({'p_raw': None, 'p_bonferroni': None})
    return result


def control_effects(rows):
    ps = collapse_profile(rows)
    if ps.empty:
        return {}
    n_layers = int(ps['Layer'].max())
    available_metrics = [metric for metric in METRICS if metric in ps.columns]
    out = {}
    for layer in (10, 11):
        layer_ps = ps[ps['Layer'] == layer]
        if layer_ps.empty:
            continue
        out[layer] = {}
        for metric in available_metrics:
            e = layer_ps.loc[layer_ps['Condition'] == 'Euphemistic', metric].tolist()
            n = layer_ps.loc[layer_ps['Condition'] == 'Non-Euphemistic', metric].tolist()
            out[layer][metric] = effect(
                e, n, bonferroni_tests=len(available_metrics) * n_layers,
            )
    return out


def paired_pet_random_contrasts(real_rows, random_rows):
    """Compare class effects on per-example PET-minus-random-span profiles.

    Profiles are emitted in test-example order, with one complete layer/head
    block per example. Verify that the real and random files have matching
    examples before pairing their per-sample, per-layer averages.
    """
    if not real_rows or not random_rows:
        return []

    def sample_metadata(rows):
        frame = pd.DataFrame(rows)
        n_layers, n_heads = int(frame['Layer'].max()), int(frame['Head'].max())
        block_size = n_layers * n_heads
        if len(frame) % block_size:
            raise ValueError(
                f"Profile row count {len(frame)} is not divisible by {block_size}"
            )
        frame['sample_id'] = np.arange(len(frame)) // block_size
        metadata = frame.groupby('sample_id', sort=True).agg(
            Condition=('Condition', 'first'),
            Sample=('Sample', 'first'),
            n_conditions=('Condition', 'nunique'),
            n_samples=('Sample', 'nunique'),
        )
        if (metadata['n_conditions'] != 1).any() or (metadata['n_samples'] != 1).any():
            raise ValueError('Profile blocks contain inconsistent sample metadata')
        return metadata[['Condition', 'Sample']]

    real_meta = sample_metadata(real_rows)
    random_meta = sample_metadata(random_rows)
    if not real_meta.equals(random_meta):
        raise ValueError(
            'Real-PET and random-span profiles do not contain the same examples '
            'in the same order; paired span contrasts cannot be computed safely'
        )

    real = collapse_profile(real_rows)
    random_span = collapse_profile(random_rows)
    metrics = [metric for metric in METRICS
               if metric in real.columns and metric in random_span.columns]
    paired = real[['sample_id', 'Layer', 'Condition', *metrics]].merge(
        random_span[['sample_id', 'Layer', 'Condition', *metrics]],
        on=['sample_id', 'Layer', 'Condition'],
        suffixes=('_real_pet', '_random_span'),
        validate='one_to_one',
    )
    for metric in metrics:
        paired[metric] = paired[f'{metric}_real_pet'] - paired[f'{metric}_random_span']

    n_layers = int(paired['Layer'].max())
    family_size = len(metrics) * n_layers
    results = []
    for layer in range(1, n_layers + 1):
        layer_data = paired[paired['Layer'] == layer]
        for metric in metrics:
            euphemistic = layer_data.loc[
                layer_data['Condition'] == 'Euphemistic', metric
            ].tolist()
            non_euphemistic = layer_data.loc[
                layer_data['Condition'] == 'Non-Euphemistic', metric
            ].tolist()
            test = effect(euphemistic, non_euphemistic, family_size)
            results.append({
                'layer': layer,
                'metric': metric,
                'paired_difference': 'real_pet_minus_random_span',
                'mean_difference_euphemistic': (
                    float(np.mean(euphemistic)) if euphemistic else None
                ),
                'mean_difference_non_euphemistic': (
                    float(np.mean(non_euphemistic)) if non_euphemistic else None
                ),
                **test,
            })
    return results


def misclassification_effects(rows):
    ps = collapse_profile(rows)
    if ps.empty or 'Predicted' not in ps.columns:
        return {}
    ps['outcome'] = np.where(ps['Condition'] == ps['Predicted'], 'correct', 'misclassified')
    result = {}
    available_metrics = [metric for metric in METRICS if metric in ps.columns]
    for layer in (10, 11):
        layer_ps = ps[ps['Layer'] == layer]
        result[layer] = {}
        for metric in available_metrics:
            result[layer][metric] = {}
            for outcome in ('correct', 'misclassified'):
                group = layer_ps[layer_ps['outcome'] == outcome]
                e = group.loc[group['Condition'] == 'Euphemistic', metric].tolist()
                n = group.loc[group['Condition'] == 'Non-Euphemistic', metric].tolist()
                result[layer][metric][outcome] = {
                    'n_euphemistic': len(e), 'n_non_euphemistic': len(n),
                    'cohens_d': cohens_d(e, n),
                }
    return result


def label_shuffle_summary(rows, n_permutations, seed):
    """Recompute the printed permutation summary from the saved real-PET profiles."""
    ps = collapse_profile(rows)
    if ps.empty or not n_permutations:
        return None
    sample_conditions = ps.groupby('sample_id', sort=True)['Condition'].first()
    sample_ids, conditions = sample_conditions.index.to_numpy(), sample_conditions.to_numpy()
    n_layers = int(ps['Layer'].max())
    available_metrics = [metric for metric in METRICS if metric in ps.columns]
    n_tests = len(available_metrics) * n_layers
    rng = np.random.default_rng(seed)
    significant_counts = []
    layer_counts = {layer: [] for layer in (10, 11)}

    for _ in range(n_permutations):
        shuffled = rng.permutation(conditions)
        condition_map = dict(zip(sample_ids, shuffled))
        permuted = ps.assign(shuffled_condition=ps['sample_id'].map(condition_map))
        total_sig = 0
        per_layer_sig = {layer: 0 for layer in layer_counts}
        for layer in range(1, n_layers + 1):
            subset = permuted[permuted['Layer'] == layer]
            for metric in available_metrics:
                euphemistic = subset.loc[subset['shuffled_condition'] == 'Euphemistic', metric]
                non_euphemistic = subset.loc[subset['shuffled_condition'] == 'Non-Euphemistic', metric]
                p = mannwhitneyu(euphemistic, non_euphemistic, alternative='two-sided').pvalue
                if p * n_tests < 0.05:
                    total_sig += 1
                    if layer in per_layer_sig:
                        per_layer_sig[layer] += 1
        significant_counts.append(total_sig)
        for layer, count in per_layer_sig.items():
            layer_counts[layer].append(count)

    real_layer_significant = {}
    for layer in layer_counts:
        real_layer = ps[ps['Layer'] == layer]
        real_significant = 0
        for metric in available_metrics:
            euphemistic = real_layer.loc[real_layer['Condition'] == 'Euphemistic', metric]
            non_euphemistic = real_layer.loc[real_layer['Condition'] == 'Non-Euphemistic', metric]
            p = mannwhitneyu(euphemistic, non_euphemistic, alternative='two-sided').pvalue
            real_significant += p * n_tests < 0.05
        real_layer_significant[layer] = int(real_significant)

    result = {
        'n_permutations': n_permutations,
        'n_metrics': len(available_metrics),
        'n_tests_per_permutation': n_tests,
        'significant_tests_per_permutation': significant_counts,
        'mean_significant_tests': float(np.mean(significant_counts)),
        'max_significant_tests': max(significant_counts),
    }
    for layer, counts in layer_counts.items():
        result[f'layer{layer}_significant_tests_per_permutation'] = counts
        result[f'permutations_with_layer{layer}_significance'] = sum(count > 0 for count in counts)
        result[f'real_layer{layer}_significant_metrics'] = real_layer_significant[layer]
    return result


def compact_evaluation(data):
    report = data.get('classification_report', {})
    return {
        'split': data.get('split'),
        'n_samples': data.get('n_samples'),
        'accuracy': data.get('accuracy'),
        'macro_f1': report.get('macro avg', {}).get('f1-score'),
        'weighted_f1': report.get('weighted avg', {}).get('f1-score'),
        'per_class': {
            label: report.get(label) for label in ('Euphemistic', 'Non-Euphemistic')
            if label in report
        },
        'confusion_matrix': data.get('confusion_matrix'),
    }


def compact_significance(data):
    """Drop retired metrics and reapply Bonferroni using the current family."""
    significance = dict(data)
    results = dict(significance.get('results') or {})
    layer_family = len(METRICS) * config.NUM_LAYERS
    head_family = len(METRICS) * config.NUM_HEADS

    for result_name in (
        'layer_condition_means', 'finetuned_layer_tests', 'finetuned_head_tests',
        'base_vs_finetuned', 'finetuning_difference_tests',
    ):
        filtered = []
        for source in results.get(result_name, []):
            if source.get('metric') not in METRICS:
                continue
            row = dict(source)
            if result_name in ('finetuned_layer_tests', 'finetuning_difference_tests'):
                p = row.get('p_raw')
                if p is not None:
                    row['p_bonferroni'] = min(float(p) * layer_family, 1.0)
                    row['significant_bonferroni_0_05'] = row['p_bonferroni'] < 0.05
            elif result_name == 'finetuned_head_tests':
                p = row.get('p_raw')
                if p is not None:
                    row['p_bonferroni'] = min(float(p) * head_family, 1.0)
                    row['significant_bonferroni_0_05'] = row['p_bonferroni'] < 0.05
            filtered.append(row)
        if result_name in results:
            results[result_name] = filtered

    # Recompute this descriptive contrast from saved profiles below because
    # older significance files do not retain its raw p-values.
    results.pop('base_vs_finetuned', None)
    significance['results'] = results
    significance['metrics'] = list(METRICS)
    significance['bonferroni'] = {
        'alpha': 0.05,
        'layer_tests': layer_family,
        'head_tests': head_family,
    }
    return significance


def base_vs_finetuned_tests(ft_rows, base_rows):
    """Recompute base/finetuned class contrasts from cached per-sample profiles."""
    ft = collapse_profile(ft_rows)
    base = collapse_profile(base_rows)
    if ft.empty or base.empty:
        return []
    family_size = len(METRICS) * config.NUM_LAYERS
    tests = []
    for layer in range(1, config.NUM_LAYERS + 1):
        for metric in METRICS:
            ft_layer = ft[ft['Layer'] == layer]
            base_layer = base[base['Layer'] == layer]
            ft_e = ft_layer.loc[ft_layer['Condition'] == 'Euphemistic', metric].to_numpy()
            ft_n = ft_layer.loc[ft_layer['Condition'] == 'Non-Euphemistic', metric].to_numpy()
            base_e = base_layer.loc[base_layer['Condition'] == 'Euphemistic', metric].to_numpy()
            base_n = base_layer.loc[base_layer['Condition'] == 'Non-Euphemistic', metric].to_numpy()
            if not all(len(values) for values in (ft_e, ft_n, base_e, base_n)):
                continue
            p_base = float(mannwhitneyu(base_e, base_n, alternative='two-sided').pvalue)
            p_ft = float(mannwhitneyu(ft_e, ft_n, alternative='two-sided').pvalue)
            d_base, d_ft = cohens_d(base_e, base_n), cohens_d(ft_e, ft_n)
            tests.append({
                'layer': layer,
                'metric': metric,
                'cohens_d_base': d_base,
                'cohens_d_finetuned': d_ft,
                'delta_cohens_d': d_ft - d_base if d_base is not None and d_ft is not None else None,
                'p_base_bonferroni': min(p_base * family_size, 1.0),
                'p_finetuned_bonferroni': min(p_ft * family_size, 1.0),
            })
    return tests


def collect_run(run_dir, manifest_index):
    files = {name: run_dir / f'{name}.json'
             for name in ('evaluate', 'significance', 'sanity_checks')}
    available = {name: path.is_file() for name, path in files.items()}
    entry = {'run_name': run_dir.name, 'available': available,
             'missing': [name for name, ok in available.items() if not ok]}

    evaluate = read_json(files['evaluate']) if available['evaluate'] else None
    if evaluate:
        entry['evaluation'] = compact_evaluation(evaluate)
        model_dir = Path(evaluate.get('model_dir', ''))
        manifest_path = model_dir.parent / 'run_manifest.json'
        indexed = manifest_index.get(model_dir.parent.name)
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            source_run_dir = model_dir.parent
        elif indexed:
            manifest, source_run_dir = indexed
        else:
            manifest, source_run_dir = {}, None
        if manifest:
            # Use the current run-folder naming to normalize aliases such as
            # split_id="42" versus "seed_42" across freeze regimes.
            folder_name = source_run_dir.name if source_run_dir else ''
            if '_train-' in folder_name:
                split_label = folder_name.split('_train-', 1)[0].removeprefix('split-')
            else:
                split_label = manifest.get('split_id')
            entry['metadata'] = {key: manifest.get(key) for key in (
                'model_name', 'split_id', 'split_seed', 'train_seed',
                'frozen_transformer_layers', 'epochs', 'batch_size',
                'learning_rate', 'device', 'gpu_name', 'versions', 'split_sizes',
            ) if key in manifest}
            entry['metadata']['split_id'] = split_label
            entry['metadata']['original_split_id'] = manifest.get('split_id')
            if source_run_dir:
                entry['source_run_dir'] = str(source_run_dir)
        if 'metadata' not in entry:
            entry['metadata'] = {'split_id': run_dir.name, 'frozen_transformer_layers': None}

    if available['significance']:
        significance = read_json(files['significance'])
        entry['significance'] = compact_significance(significance)
        entry['significance']['n_test_samples'] = significance.get('n_test_samples')

    if available['sanity_checks']:
        checks = read_json(files['sanity_checks'])
        n = checks.get('n_profiled_samples', 0)
        flips = checks.get('n_prediction_flips_after_masking_pet', 0)
        profile_data = checks.get('profiles', {})
        if 'significance' in entry and profile_data.get('real_pet') and profile_data.get('base_real_pet'):
            entry['significance']['results']['base_vs_finetuned'] = base_vs_finetuned_tests(
                profile_data['real_pet'], profile_data['base_real_pet'],
            )
        correlations = checks.get('context_length_correlations')
        context_adjustment = checks.get('context_length_adjustment')
        joint_length_adjustment = checks.get('pet_context_length_adjustment')
        sample_lengths = pd.DataFrame(checks['lengths']) if checks.get('lengths') else pd.DataFrame()
        if not sample_lengths.empty and 'context_len' not in sample_lengths:
            sink_count = sample_lengths.get('sink_count', 2)
            sample_lengths['context_len'] = (
                sample_lengths['seq_len'] - sample_lengths['pet_len'] - sink_count
            )
        if profile_data.get('real_pet') and not sample_lengths.empty:
            # Recompute from saved sample profiles so newer metric selections
            # (including PET-internal flow) are reflected without inference.
            correlations = context_length_correlations(
                collapse_profile(profile_data['real_pet']), sample_lengths,
            )
            context_adjustment = context_length_adjusted_attention(
                collapse_profile(profile_data['real_pet']), sample_lengths,
            )
            joint_length_adjustment = pet_context_length_adjusted_attention(
                collapse_profile(profile_data['real_pet']), sample_lengths,
            )
        if context_adjustment is not None and 'class_effect_tests' not in context_adjustment:
            context_adjustment = dict(context_adjustment)
            context_adjustment['class_effect_tests'] = context_length_adjusted_tests(
                context_adjustment.get('data_points', []),
                bonferroni_tests=FULL_ATTENTION_TEST_FAMILY_SIZE,
            )
            context_adjustment['multiple_testing_correction'] = (
                'Bonferroni across the existing 7-metric × 12-layer family '
                f'({FULL_ATTENTION_TEST_FAMILY_SIZE} tests per run)'
            )
        entry['sanity_checks'] = {
            'n_test_samples': checks.get('n_test_samples'),
            'n_profiled_samples': n,
            'prediction_flips_after_masking_pet': flips,
            'prediction_flip_rate': flips / n if n else None,
            'seed': checks.get('seed'),
            'n_permutations': checks.get('n_permutations'),
            'layer10_11_control_effects': {
                key: control_effects(profile_data[key])
                for key in PROFILE_KEYS if key in profile_data
            },
            'paired_pet_random_contrasts': paired_pet_random_contrasts(
                profile_data.get('real_pet', []),
                profile_data.get('random_span', []),
            ),
            'layer10_11_correctness_effects': misclassification_effects(
                profile_data.get('real_pet', [])),
            'label_shuffle': label_shuffle_summary(
                profile_data.get('real_pet', []), checks.get('n_permutations'), checks.get('seed')),
            'context_length_correlations': correlations,
            'context_length_adjustment': context_adjustment,
            'pet_context_length_adjustment': joint_length_adjustment,
            'raw_profile_rows_omitted': {
                key: len(value) for key, value in profile_data.items()
            },
        }

    return entry


def mean_range(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {'mean': None, 'min': None, 'max': None}
    return {'mean': float(np.mean(values)), 'min': min(values), 'max': max(values)}


def group_key(run):
    metadata = run.get('metadata', {})
    return (str(metadata.get('split_id') or run['run_name']),
            metadata.get('frozen_transformer_layers'))


def build_comparisons(runs):
    """Summarize across training runs, but keep split protocols separate."""
    groups = {}
    for run in runs:
        groups.setdefault(group_key(run), []).append(run)

    comparisons = []
    for (split_id, frozen_layers), members in sorted(groups.items()):
        group = {
            'split_id': split_id,
            'frozen_transformer_layers': frozen_layers,
            'run_names': [run['run_name'] for run in members],
            'n_runs': len(members),
            'performance': {},
            'layer_effects': [],
            'attention_controls': [],
            'paired_pet_random_contrasts': [],
            'prediction_flip_rate': mean_range([
                run.get('sanity_checks', {}).get('prediction_flip_rate') for run in members
            ]),
            'label_shuffle': [],
            'layer10_11_correctness': [],
            'context_length_correlations': [],
            'context_length_adjustment': [],
            'pet_context_length_adjustment': [],
        }

        for field in ('accuracy', 'macro_f1', 'weighted_f1'):
            group['performance'][field] = mean_range([
                run.get('evaluation', {}).get(field) for run in members
            ])

        effect_map = {}
        for run in members:
            results = run.get('significance', {}).get('results', {})
            for key, value_name, p_name in (
                ('finetuned_layer_tests', 'cohens_d', 'p_bonferroni'),
                ('finetuning_difference_tests', 'cohens_d', 'p_bonferroni'),
            ):
                for test in results.get(key, []):
                    item = effect_map.setdefault(
                        (test['layer'], test['metric']),
                        {'absolute': [], 'finetuning_difference': []},
                    )
                    label = 'absolute' if key == 'finetuned_layer_tests' else 'finetuning_difference'
                    item[label].append({
                        'run': run['run_name'], 'd': test.get(value_name),
                        'p_bonferroni': test.get(p_name),
                    })

        for (layer, metric), tests in sorted(effect_map.items()):
            row = {'layer': layer, 'metric': metric}
            for label in ('absolute', 'finetuning_difference'):
                values = tests[label]
                ds = [test['d'] for test in values if test['d'] is not None]
                row[label] = {
                    **mean_range(ds),
                    'n_runs': len(values),
                    'positive_direction_runs': sum(d > 0 for d in ds),
                    'negative_direction_runs': sum(d < 0 for d in ds),
                    'bonferroni_significant_runs': sum(
                        test['p_bonferroni'] is not None and test['p_bonferroni'] < 0.05
                        for test in values
                    ),
                }
            group['layer_effects'].append(row)

        control_map = {}
        paired_span_map = {}
        correctness_map = {}
        correlation_map = {}
        adjustment_map = {}
        joint_adjustment_map = {}
        for run in members:
            sanity = run.get('sanity_checks', {})
            shuffle = sanity.get('label_shuffle')
            if shuffle is not None:
                group['label_shuffle'].append({
                    'run': run['run_name'],
                    'mean_significant_tests': shuffle['mean_significant_tests'],
                    'max_significant_tests': shuffle['max_significant_tests'],
                    'permutations_with_layer10_significance': shuffle['permutations_with_layer10_significance'],
                    'real_layer10_significant_metrics': shuffle['real_layer10_significant_metrics'],
                    'permutations_with_layer11_significance': shuffle['permutations_with_layer11_significance'],
                    'n_permutations': shuffle['n_permutations'],
                    'n_metrics': shuffle.get('n_metrics'),
                    'real_layer11_significant_metrics': shuffle['real_layer11_significant_metrics'],
                })
            for profile_name, layers in sanity.get('layer10_11_control_effects', {}).items():
                for layer, metrics in layers.items():
                    for metric, result in metrics.items():
                        control_map.setdefault((profile_name, int(layer), metric), []).append({
                            'run': run['run_name'], 'd': result.get('cohens_d'),
                            'p_bonferroni': result.get('p_bonferroni'),
                        })
            for result in sanity.get('paired_pet_random_contrasts', []):
                key = (result['layer'], result['metric'])
                paired_span_map.setdefault(key, []).append({
                    'run': run['run_name'],
                    'd': result.get('cohens_d'),
                    'p_bonferroni': result.get('p_bonferroni'),
                    'mean_difference_euphemistic': result.get(
                        'mean_difference_euphemistic'),
                    'mean_difference_non_euphemistic': result.get(
                        'mean_difference_non_euphemistic'),
                })
            for layer, metrics in sanity.get('layer10_11_correctness_effects', {}).items():
                for metric, outcomes in metrics.items():
                    for outcome, result in outcomes.items():
                        correctness_map.setdefault((int(layer), metric, outcome), []).append({
                            'run': run['run_name'], 'd': result.get('cohens_d'),
                        })
            correlation_data = sanity.get('context_length_correlations') or {}
            for result in correlation_data.get('results', []):
                key = (result['condition'], result['layer'], result['metric'])
                correlation_map.setdefault(key, []).append({
                    'run': run['run_name'],
                    'rho': result.get('spearman_rho'),
                    'p_bh': result.get('p_bh'),
                })
            adjustment_data = sanity.get('context_length_adjustment') or {}
            points = adjustment_data.get('data_points', [])
            adjusted_tests = {
                (test['layer'], test['metric']): test
                for test in adjustment_data.get('class_effect_tests', [])
            }
            for layer in sorted({point['layer'] for point in points}):
                for metric in LENGTH_ADJUSTED_METRICS:
                    metric_points = [point for point in points
                                     if point['layer'] == layer and point['metric'] == metric]
                    raw_e = [p['raw_value'] for p in metric_points
                             if p['condition'] == 'Euphemistic' and p['raw_value'] is not None]
                    raw_n = [p['raw_value'] for p in metric_points
                             if p['condition'] == 'Non-Euphemistic' and p['raw_value'] is not None]
                    adj_e = [p['length_adjusted_value'] for p in metric_points
                             if p['condition'] == 'Euphemistic' and p['length_adjusted_value'] is not None]
                    adj_n = [p['length_adjusted_value'] for p in metric_points
                             if p['condition'] == 'Non-Euphemistic' and p['length_adjusted_value'] is not None]
                    key = (layer, metric)
                    adjustment_map.setdefault(key, []).append({
                        'run': run['run_name'],
                        'raw_d': cohens_d(raw_e, raw_n),
                        'adjusted_d': cohens_d(adj_e, adj_n),
                        'class_effect_test': adjusted_tests.get(key),
                    })
            joint_data = sanity.get('pet_context_length_adjustment') or {}
            joint_points = joint_data.get('data_points', [])
            joint_tests = {
                (test['layer'], test['metric']): test
                for test in joint_data.get('class_effect_tests', [])
            }
            joint_results = {
                (result['layer'], result['metric']): result
                for result in joint_data.get('results', [])
                if result['condition'] == 'Euphemistic'
            }
            for layer in sorted({point['layer'] for point in joint_points}):
                for metric in LENGTH_ADJUSTED_METRICS:
                    metric_points = [point for point in joint_points
                                     if point['layer'] == layer and point['metric'] == metric]
                    raw_e = [p['raw_value'] for p in metric_points
                             if p['condition'] == 'Euphemistic' and p['raw_value'] is not None]
                    raw_n = [p['raw_value'] for p in metric_points
                             if p['condition'] == 'Non-Euphemistic' and p['raw_value'] is not None]
                    adj_e = [p['length_adjusted_value'] for p in metric_points
                             if p['condition'] == 'Euphemistic' and p['length_adjusted_value'] is not None]
                    adj_n = [p['length_adjusted_value'] for p in metric_points
                             if p['condition'] == 'Non-Euphemistic' and p['length_adjusted_value'] is not None]
                    key = (layer, metric)
                    slopes = joint_results.get(key, {})
                    joint_adjustment_map.setdefault(key, []).append({
                        'run': run['run_name'],
                        'raw_d': cohens_d(raw_e, raw_n),
                        'adjusted_d': cohens_d(adj_e, adj_n),
                        'class_effect_test': joint_tests.get(key),
                        'pet_length_slope': slopes.get('pet_length_slope'),
                        'context_length_slope': slopes.get('context_length_slope'),
                    })

        for (profile_name, layer, metric), values in sorted(control_map.items()):
            ds = [v['d'] for v in values if v['d'] is not None]
            group['attention_controls'].append({
                'profile': profile_name, 'layer': layer, 'metric': metric,
                **mean_range(ds), 'n_runs': len(values),
                'bonferroni_significant_runs': sum(
                    v['p_bonferroni'] is not None and v['p_bonferroni'] < 0.05
                    for v in values
                ),
            })
        for (layer, metric), values in sorted(paired_span_map.items()):
            ds = [value['d'] for value in values if value['d'] is not None]
            group['paired_pet_random_contrasts'].append({
                'layer': layer,
                'metric': metric,
                **mean_range(ds),
                'n_runs': len(values),
                'positive_direction_runs': sum(d > 0 for d in ds),
                'negative_direction_runs': sum(d < 0 for d in ds),
                'bonferroni_significant_runs': sum(
                    value['p_bonferroni'] is not None
                    and value['p_bonferroni'] < 0.05 for value in values
                ),
                'mean_difference_euphemistic': mean_range([
                    value['mean_difference_euphemistic'] for value in values
                ]),
                'mean_difference_non_euphemistic': mean_range([
                    value['mean_difference_non_euphemistic'] for value in values
                ]),
            })
        for (layer, metric, outcome), values in sorted(correctness_map.items()):
            group['layer10_11_correctness'].append({
                'layer': layer, 'metric': metric, 'outcome': outcome,
                **mean_range([v['d'] for v in values]), 'n_runs': len(values),
            })
        for (condition, layer, metric), values in sorted(correlation_map.items()):
            rhos = [value['rho'] for value in values if value['rho'] is not None]
            group['context_length_correlations'].append({
                'condition': condition, 'layer': layer, 'metric': metric,
                **mean_range(rhos),
                'n_runs': len(values),
                'positive_direction_runs': sum(rho > 0 for rho in rhos),
                'negative_direction_runs': sum(rho < 0 for rho in rhos),
                'fdr_significant_runs': sum(
                    value['p_bh'] is not None and value['p_bh'] < 0.05
                    for value in values
                ),
            })
        for (layer, metric), values in sorted(adjustment_map.items()):
            raw_ds = [value['raw_d'] for value in values if value['raw_d'] is not None]
            adjusted_ds = [value['adjusted_d'] for value in values if value['adjusted_d'] is not None]
            tests = [value['class_effect_test'] for value in values
                     if value['class_effect_test'] is not None]
            group['context_length_adjustment'].append({
                'layer': layer, 'metric': metric, 'n_runs': len(values),
                'raw_d': mean_range(raw_ds),
                'length_adjusted_d': mean_range(adjusted_ds),
                'adjusted_mean_difference': mean_range([
                    test['adjusted_mean_difference'] for test in tests
                ]),
                'bonferroni_significant_runs': sum(
                    test['p_bonferroni'] < 0.05 for test in tests
                ),
                'n_tested_runs': len(tests),
            })
        for (layer, metric), values in sorted(joint_adjustment_map.items()):
            raw_ds = [value['raw_d'] for value in values if value['raw_d'] is not None]
            adjusted_ds = [value['adjusted_d'] for value in values
                           if value['adjusted_d'] is not None]
            tests = [value['class_effect_test'] for value in values
                     if value['class_effect_test'] is not None]
            group['pet_context_length_adjustment'].append({
                'layer': layer, 'metric': metric, 'n_runs': len(values),
                'raw_d': mean_range(raw_ds),
                'length_adjusted_d': mean_range(adjusted_ds),
                'pet_length_slope': mean_range([
                    value['pet_length_slope'] for value in values
                ]),
                'context_length_slope': mean_range([
                    value['context_length_slope'] for value in values
                ]),
                'adjusted_mean_difference': mean_range([
                    test['adjusted_mean_difference'] for test in tests
                ]),
                'bonferroni_significant_runs': sum(
                    test['p_bonferroni'] < 0.05 for test in tests
                ),
                'n_tested_runs': len(tests),
            })
        comparisons.append(group)
    return comparisons


def build_direction_recurrence(runs):
    """Count repeated directions/significance without pooling split-level d values."""
    by_freeze = {}
    for run in runs:
        freeze = run.get('metadata', {}).get('frozen_transformer_layers')
        by_freeze.setdefault(freeze, []).append(run)
    summaries = []
    for freeze, members in sorted(by_freeze.items(), key=lambda item: str(item[0])):
        tests = {}
        for run in members:
            results = run.get('significance', {}).get('results', {})
            for source, label in (('finetuned_layer_tests', 'absolute'),
                                  ('finetuning_difference_tests', 'finetuning_difference')):
                for test in results.get(source, []):
                    if test['layer'] not in (10, 11, 12):
                        continue
                    cell = tests.setdefault((test['layer'], test['metric']),
                                            {'absolute': [], 'finetuning_difference': []})
                    cell[label].append(test)
        rows = []
        for (layer, metric), data in sorted(tests.items()):
            row = {'layer': layer, 'metric': metric}
            for label in ('absolute', 'finetuning_difference'):
                vals = data[label]
                ds = [v.get('cohens_d') for v in vals if v.get('cohens_d') is not None]
                row[label] = {
                    'n_runs': len(vals),
                    'positive_direction_runs': sum(d > 0 for d in ds),
                    'negative_direction_runs': sum(d < 0 for d in ds),
                    'bonferroni_significant_runs': sum(
                        v.get('p_bonferroni') is not None and v['p_bonferroni'] < 0.05
                        for v in vals
                    ),
                }
            rows.append(row)
        summaries.append({
            'frozen_transformer_layers': freeze,
            'run_names': [run['run_name'] for run in members],
            'rows': rows,
        })
    return summaries


def fmt_range(summary, digits=2, signed=False):
    if summary.get('mean') is None:
        return '-'
    sign = '+' if signed else ''
    return (f"{summary['mean']:{sign}.{digits}f} "
            f"[{summary['min']:{sign}.{digits}f}, {summary['max']:{sign}.{digits}f}]")


def fmt_num(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else '-'


def render_table(headers, rows, right_align=()):
    """Render a compact table that remains legible in an unformatted text viewer."""
    values = [[str(value) for value in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in values:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(row):
        cells = []
        for index, value in enumerate(row):
            cells.append(value.rjust(widths[index]) if index in right_align
                         else value.ljust(widths[index]))
        return '  '.join(cells).rstrip()

    header = format_row([str(value) for value in headers])
    separator = '  '.join('-' * width for width in widths).rstrip()
    return [header, separator, *(format_row(row) for row in values)]


def render_plaintext(runs, comparisons, recurrence):
    lines = [
        'CROSS-RUN RESULTS SUMMARY',
        '==========================',
        '',
        f"Generated from {len(runs)} run(s). Runs are summarized within each split and freeze regime; "
        'different test splits are not pooled.',
        '',
        'RUN INVENTORY AND TASK PERFORMANCE',
        '-----------------------------------',
        '',
    ]
    inventory_rows = []
    for run in runs:
        meta = run.get('metadata', {})
        ev = run.get('evaluation', {})
        inventory_rows.append([
            meta.get('split_id', run['run_name']), meta.get('train_seed', '-'),
            meta.get('frozen_transformer_layers', '-'), ev.get('n_samples', '-'),
            fmt_num(ev.get('accuracy')), fmt_num(ev.get('macro_f1')),
            ', '.join(name for name, ok in run['available'].items() if ok),
        ])
    lines.extend(render_table(
        ['Split', 'Train seed', 'Frozen layers', 'N test', 'Accuracy', 'Macro-F1', 'Stats available'],
        inventory_rows, right_align=(1, 2, 3, 4, 5),
    ))

    for regime in recurrence:
        lines += [
            '',
            f"DIRECTION/SIGNIFICANCE RECURRENCE ACROSS SPLITS "
            f"(FROZEN LAYERS: {regime['frozen_transformer_layers']})",
            '-' * (51 + len(str(regime['frozen_transformer_layers']))),
            '',
            f"Descriptive counts over {len(regime['run_names'])} runs; no effect sizes or p-values "
            'are pooled across split protocols.',
            '',
        ]
        rows = []
        for row in regime['rows']:
            absolute, difference = row['absolute'], row['finetuning_difference']
            rows.append([
                row['layer'], row['metric'],
                f"+{absolute['positive_direction_runs']}/-{absolute['negative_direction_runs']}",
                f"{absolute['bonferroni_significant_runs']}/{absolute['n_runs']}",
                f"+{difference['positive_direction_runs']}/-{difference['negative_direction_runs']}",
                f"{difference['bonferroni_significant_runs']}/{difference['n_runs']}",
            ])
        lines.extend(render_table(
            ['Layer', 'Metric', 'Absolute d direction', 'Absolute significant',
             'Fine-tuning difference direction', 'Difference significant'],
            rows, right_align=(0, 3, 5),
        ))

    for group in comparisons:
        lines += [
            '',
            f"SPLIT {group['split_id']}; FROZEN LAYERS: {group['frozen_transformer_layers']} "
            f"(N={group['n_runs']} TRAINING RUN(S))",
            '=' * (39 + len(str(group['split_id'])) + len(str(group['frozen_transformer_layers']))),
            '',
        ]
        lines.extend(render_table(
            ['Metric', 'Accuracy mean [range]', 'Macro-F1 mean [range]'],
            [['Task performance', fmt_range(group['performance']['accuracy'], 3),
              fmt_range(group['performance']['macro_f1'], 3)]],
            right_align=(1, 2),
        ))
        lines += [
            '',
            "Layer effects are Cohen's d (euphemistic - non-euphemistic), summarized as "
            'mean [min, max] across training runs. Significance is per-run Bonferroni p < .05.',
            '',
        ]
        rows = []
        for row in group['layer_effects']:
            if row['layer'] not in (10, 11, 12):
                continue
            absolute, difference = row['absolute'], row['finetuning_difference']
            rows.append([
                row['layer'], row['metric'], fmt_range(absolute, signed=True),
                f"{absolute['bonferroni_significant_runs']}/{absolute['n_runs']}",
                fmt_range(difference, signed=True),
                f"{difference['bonferroni_significant_runs']}/{difference['n_runs']}",
            ])
        lines.extend(render_table(
            ['Layer', 'Metric', 'Absolute d mean [range]', 'Absolute sig.',
             'Fine-tuning difference d mean [range]', 'Difference sig.'],
            rows, right_align=(0, 2, 3, 4, 5),
        ))

        lines += [
            '',
            f"PET-mask prediction-flip rate: {fmt_range(group['prediction_flip_rate'], 3)}.",
            '',
        ]
        rows = []
        for shuffle in group['label_shuffle']:
            rows.append([
                shuffle['run'], f"{shuffle['mean_significant_tests']:.2f}",
                shuffle['max_significant_tests'],
                    f"{shuffle['permutations_with_layer10_significance']}/{shuffle['n_permutations']}",
                    f"{shuffle['real_layer10_significant_metrics']}/{shuffle.get('n_metrics', 6)}",
                f"{shuffle['permutations_with_layer11_significance']}/{shuffle['n_permutations']}",
                f"{shuffle['real_layer11_significant_metrics']}/{shuffle.get('n_metrics', 6)}",
            ])
        lines.extend(render_table(
            ['Label-shuffle control', 'Mean significant tests / permutation', 'Max',
             'Layer-10-hit permutations', 'Real layer-10 metrics significant',
             'Layer-11-hit permutations', 'Real layer-11 metrics significant'],
            rows, right_align=(1, 2, 3, 4, 5, 6),
        ))
        lines += [
            '',
        ]
        rows = []
        for row in group['attention_controls']:
            if row['profile'] in ('masked_pet', 'random_span'):
                rows.append([
                    row['profile'], row['layer'], row['metric'],
                    fmt_range(row, signed=True),
                    f"{row['bonferroni_significant_runs']}/{row['n_runs']}",
                ])
        lines.extend(render_table(
            ['Attention control', 'Layer', 'Metric', "Cohen's d mean [range]",
             'Bonferroni-significant runs'],
            rows, right_align=(1, 3, 4),
        ))
        lines += [
            '',
        ]
        if group['paired_pet_random_contrasts']:
            lines += [
                'PAIRED PET-MINUS-RANDOM-SPAN CLASS CONTRASTS',
                'Per-example difference is PET attention minus matched random-span attention.',
                'Cohen\'s d compares these differences between classes; significance uses 84-test Bonferroni correction per run.',
                '',
            ]
            rows = []
            for item in group['paired_pet_random_contrasts']:
                if item['layer'] not in (10, 11) or item['metric'] not in (
                        'Inward (Context → PET)', 'Internal (PET → PET)'):
                    continue
                rows.append([
                    item['layer'], item['metric'],
                    fmt_range(item, signed=True),
                    f"{item['positive_direction_runs']}/{item['n_runs']}",
                    f"{item['bonferroni_significant_runs']}/{item['n_runs']}",
                ])
            lines.extend(render_table(
                ['Layer', 'Metric', 'Paired-contrast d mean [range]',
                 'Positive runs', 'Bonferroni-significant runs'],
                rows, right_align=(0, 2, 3, 4),
            ))
            lines += ['', '']
        correct = {
            (r['layer'], r['metric']): r for r in group['layer10_11_correctness']
            if r['outcome'] == 'correct'
        }
        incorrect = {
            (r['layer'], r['metric']): r for r in group['layer10_11_correctness']
            if r['outcome'] == 'misclassified'
        }
        rows = []
        for layer in (10, 11):
            for metric in ('Inward (Context → PET)', 'Internal (PET → PET)',
                           'Sink (PET → Sink)', 'Sink (Sink → PET)'):
                key = (layer, metric)
                if key in correct or key in incorrect:
                    rows.append([
                        layer, metric, fmt_range(correct.get(key, {}), signed=True),
                        fmt_range(incorrect.get(key, {}), signed=True),
                    ])
        lines.extend(render_table(
            ['Layer', 'Metric', 'Correctly classified d', 'Misclassified d'], rows,
            right_align=(0, 2, 3),
        ))
        if group['context_length_correlations']:
            lines += [
                '',
                'CONTEXT-LENGTH / ATTENTION CORRELATIONS',
                'Spearman rho summarized across runs; FDR significance is counted per run.',
                '',
            ]
            rows = []
            for row in group['context_length_correlations']:
                rows.append([
                    row['condition'], row['layer'], row['metric'],
                    fmt_range(row, digits=3, signed=True),
                    f"{row['positive_direction_runs']}/{row['n_runs']}",
                    f"{row['negative_direction_runs']}/{row['n_runs']}",
                    f"{row['fdr_significant_runs']}/{row['n_runs']}",
                ])
            lines.extend(render_table(
                ['Condition', 'Layer', 'Metric', 'rho mean [range]', 'Positive',
                 'Negative', 'FDR-significant'], rows,
                right_align=(1, 3, 4, 5, 6),
            ))
        if group['context_length_adjustment']:
            lines += [
                '',
                'CONTEXT-LENGTH-ADJUSTED ATTENTION EFFECTS',
                'Cohen\'s d across conditions before and after linear adjustment to the run mean context length.',
                'Significance tests the class coefficient in attention ~ class + context length using HC3 robust OLS.',
                f"Per-run Bonferroni correction uses the existing {FULL_ATTENTION_TEST_FAMILY_SIZE}-test family.",
                'Positive d and adjusted mean difference indicate greater Euphemistic attention.',
                '',
            ]
            rows = []
            for row in group['context_length_adjustment']:
                raw = row['raw_d']
                adjusted = row['length_adjusted_d']
                delta = None
                if raw.get('mean') is not None and adjusted.get('mean') is not None:
                    delta = adjusted['mean'] - raw['mean']
                rows.append([
                    row['layer'], row['metric'], fmt_range(raw, digits=3, signed=True),
                    fmt_range(adjusted, digits=3, signed=True),
                    '-' if delta is None else f'{delta:+.3f}',
                    fmt_range(row['adjusted_mean_difference'], digits=4, signed=True),
                    f"{row['bonferroni_significant_runs']}/{row['n_tested_runs']}",
                ])
            lines.extend(render_table(
                ['Layer', 'Metric', 'Raw d mean [range]', 'Adjusted d mean [range]',
                 'Δ mean d', 'Adjusted mean diff.', 'Adjusted sig.'], rows,
                right_align=(0, 2, 3, 4, 5, 6),
            ))
        if group['pet_context_length_adjustment']:
            lines += [
                '',
                'JOINT PET- AND CONTEXT-LENGTH-ADJUSTED ATTENTION EFFECTS',
                'Cohen\'s d before and after adjustment to each run\'s mean PET and context lengths.',
                'Significance tests the class coefficient in attention ~ class + PET length + context length using HC3 robust OLS.',
                f"Per-run Bonferroni correction uses the existing {FULL_ATTENTION_TEST_FAMILY_SIZE}-test family.",
                'Positive d and adjusted mean difference indicate greater Euphemistic attention.',
                '',
            ]
            rows = []
            for row in group['pet_context_length_adjustment']:
                raw = row['raw_d']
                adjusted = row['length_adjusted_d']
                delta = None
                if raw.get('mean') is not None and adjusted.get('mean') is not None:
                    delta = adjusted['mean'] - raw['mean']
                rows.append([
                    row['layer'], row['metric'], fmt_range(raw, digits=3, signed=True),
                    fmt_range(adjusted, digits=3, signed=True),
                    '-' if delta is None else f'{delta:+.3f}',
                    fmt_range(row['adjusted_mean_difference'], digits=4, signed=True),
                    f"{row['bonferroni_significant_runs']}/{row['n_tested_runs']}",
                ])
            lines.extend(render_table(
                ['Layer', 'Metric', 'Raw d mean [range]', 'Adjusted d mean [range]',
                 'Δ mean d', 'Adjusted mean diff.', 'Adjusted sig.'], rows,
                right_align=(0, 2, 3, 4, 5, 6),
            ))

    lines += [
        '',
        'INTERPRETATION NOTES',
        '--------------------',
        '',
        'Ranges describe observed run-to-run variation; with few runs they are not confidence intervals.',
        'Effects are summarized within matching split/freeze groups, not averaged across distinct splits.',
        'Layers 10-12 are shown above; the JSON output retains the summaries for every layer.',
        "Label-shuffle outcomes are recomputed from saved real-PET profiles using each run's recorded seed.",
        '',
    ]
    return '\n'.join(lines)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description='Summarize cross-run performance, effect stability, and controls.'
    )
    parser.add_argument('--stats-dir', type=Path, default=root / 'output' / 'stats',
                        help='Directory containing one subdirectory per run.')
    parser.add_argument('--output-json', type=Path, default=None,
                        help='Structured summary JSON (default: <stats-dir>/aggregate_summary.json).')
    parser.add_argument('--output-txt', type=Path, default=None,
                        help='Plain-text report (default: <stats-dir>/aggregate_summary.txt).')
    args = parser.parse_args()
    stats_dir = args.stats_dir.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve() if args.output_json else stats_dir / 'aggregate_summary.json'
    output_txt = args.output_txt.expanduser().resolve() if args.output_txt else stats_dir / 'aggregate_summary.txt'

    run_root = root / 'runs'
    manifest_index = {}
    if run_root.is_dir():
        for manifest_path in run_root.glob('*/run_manifest.json'):
            manifest = read_json(manifest_path)
            run_name = manifest.get('run_name')
            if run_name:
                manifest_index[run_name] = (manifest, manifest_path.parent)

    run_dirs = sorted(path for path in stats_dir.iterdir()
                      if path.is_dir() and any(
                          (path / f'{name}.json').is_file()
                          for name in ('evaluate', 'significance', 'sanity_checks')
                      ))
    runs = [collect_run(path, manifest_index) for path in run_dirs]
    comparisons = build_comparisons(runs)
    recurrence = build_direction_recurrence(runs)
    aggregate = {
        'script': 'aggregate_stats.py',
        'stats_dir': str(stats_dir),
        'n_runs': len(runs),
        'runs': runs,
        'comparisons_by_split_and_freeze': comparisons,
        'direction_significance_recurrence_by_freeze': recurrence,
        'notes': [
            'Training runs are summarized within matching split and freeze groups.',
            'Raw sanity-check profile rows are summarized and omitted from this file.',
            'Label-shuffle summaries are recomputed from saved real-PET profiles and recorded seeds.',
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open('w', encoding='utf-8') as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(render_plaintext(runs, comparisons, recurrence), encoding='utf-8')
    logger.info("Summarized %s run(s) to:\n  %s\n  %s", len(runs), output_txt, output_json)


if __name__ == '__main__':
    main()
