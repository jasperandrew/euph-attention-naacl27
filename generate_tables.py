"""Generate paper-ready main and appendix tables from aggregate statistics.

This script only reads the saved aggregate JSON; it does not run model inference
or recompute statistical tests. Generated LaTeX is written to a separate file
and can be included from the paper with ``\\input{...}``.
"""
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AGGREGATE_PATH = ROOT / 'output' / 'stats' / 'aggregate_summary.json'
OUTPUT_PATH = ROOT / 'output' / 'paper' / 'paper_tables.tex'
SPLITS_DIR = ROOT / 'datasets' / 'splits'
DATA_SPLIT_NAMES = {
    '42': 'Stratified (seed 42)',
    'lrec': 'PET-disjoint (Poh et al.)',
}
CLASS_DISPLAY_NAMES = {
    'Euphemistic': 'Euph.',
    'Non-Euphemistic': 'Non-Euph.',
}
CLASS_CONTRAST_LABEL = (
    f"{CLASS_DISPLAY_NAMES['Euphemistic']} $-$ "
    f"{CLASS_DISPLAY_NAMES['Non-Euphemistic']}"
)
DATA_COLUMN_HEADERS = (
    'Split/Partition',
    'Euphemistic',
    'Non-Euph.',
)
PERFORMANCE_SPLIT_NAMES = {
    '42': 'Stratified',
    'lrec': 'PET-disjoint',
}
PERFORMANCE_COLUMN_HEADERS = ('Split', 'Accuracy', 'Macro-F1')
PER_CLASS_SPLIT_NAMES = {
    '42': 'Stratified',
    'lrec': 'PET-disjoint',
}
PER_CLASS_COLUMN_HEADERS = ('Split', 'Class', 'Precision', 'Recall', 'F1')
ATTENTION_EFFECT_SPLIT_NAMES = {
    '42': 'Stratified',
    'lrec': 'PET-disjoint',
}
ATTENTION_MASKING_COLUMN_HEADERS = (
    'Layer', 'Metric', r'Raw $d$', r'Joint-adjusted $d$',
    r'Masked $d$ (sig.)', r'Random-span $d$ (sig.)',
)
TABLE_METRICS = ('inward', 'internal')
TABLE_LAYERS = (10, 11)
FINETUNING_DIFF_METRICS = (
    'inward', 'internal', 'outward', 'pet-sink', 'sink-pet', 'entropy', 'distance',
)

METRIC_LABELS = {
    'Inward (Context → PET)': r'Inward ($C\to P$)',
    'Internal (PET → PET)': r'Internal ($P\to P$)',
    'Outward (PET → Context)': r'Outward ($P\to C$)',
    'Sink (PET → Sink)': r'To-Sink ($P\to S$)',
    'Sink (Sink → PET)': r'From-Sink ($S\to P$)',
    'Entropy': 'Entropy',
    'Distance': 'Distance',
}


def fmt_metric(value):
    return METRIC_LABELS.get(value, value)


def header_row(headers):
    return ' & '.join(headers) + r' \\'


def fmt_range(stats, digits=2):
    """Format a mean and observed min/max range without implying a CI."""
    if not stats or stats.get('mean') is None:
        return '--'
    return (f"{stats['mean']:.{digits}f} "
            f"[{stats['min']:.{digits}f}, {stats['max']:.{digits}f}]")


def fmt_signed_range(stats, digits=2):
    """Format a mean and observed range, retaining minus signs when needed."""
    if not stats or stats.get('mean') is None:
        return '--'
    return (f"{stats['mean']:.{digits}f} "
            f"[{stats['min']:.{digits}f}, {stats['max']:.{digits}f}]")


def fmt_mean(stats, digits=2):
    if not stats or stats.get('mean') is None:
        return '--'
    return f"{stats['mean']:.{digits}f}"


def fmt_split(split_id, split_names):
    return split_names.get(str(split_id), str(split_id))


def read_class_counts(split_dir):
    """Read class counts from a train/validation/test split using the CSV stdlib."""
    counts = {}
    for partition in ('train', 'val', 'test'):
        csv_path = split_dir / f'{partition}.csv'
        if not csv_path.is_file():
            raise FileNotFoundError(f'Missing dataset partition: {csv_path}')
        with csv_path.open(encoding='utf-8-sig', newline='') as source:
            reader = csv.DictReader(source)
            if 'label' not in (reader.fieldnames or []):
                raise ValueError(f'{csv_path} has no label column.')
            partition_counts = {'euphemistic': 0, 'non_euphemistic': 0}
            for row in reader:
                label = str(row['label']).strip().lower()
                if label in {'1', '1.0', 'true', 'euph', 'euphemistic'}:
                    partition_counts['euphemistic'] += 1
                elif label in {'0', '0.0', 'false', 'non-euph', 'non-euphemistic'}:
                    partition_counts['non_euphemistic'] += 1
                else:
                    raise ValueError(f'Unrecognized label {row["label"]!r} in {csv_path}.')
            counts[partition] = partition_counts
    return counts


def fmt_count(count, total):
    percentage = 100 * count / total if total else 0
    return f'{count:,} ({percentage:.1f}\\%)'


def dataset_table(split_dirs):
    groups = []
    for split_index, (split_name, split_dir) in enumerate(split_dirs):
        counts = read_class_counts(split_dir)
        rows = []
        for partition in ('train', 'val', 'test'):
            classes = counts[partition]
            total = classes['euphemistic'] + classes['non_euphemistic']
            rows.append(
                f"        {partition.title()} ($N={total:,}$) & "
                f"{fmt_count(classes['euphemistic'], total)} & "
                f"{fmt_count(classes['non_euphemistic'], total)} \\\\")
        groups.append((split_name, rows))

    body = []
    for split_index, (split_name, rows) in enumerate(groups):
        if split_index:
            body.append('        \\midrule')
        body.append(f"        \\multicolumn{{3}}{{l}}{{\\textbf{{{split_name}}}}} \\\\")
        body.extend(rows)
    return rf'''\begin{{table}}[h]
    \centering\small
    \begin{{tabular}}{{lrr}}
        \toprule
        {header_row(DATA_COLUMN_HEADERS)}
        \midrule
{chr(10).join(body)}
        \bottomrule
    \end{{tabular}}
    \caption{{Class distribution across training, validation, and test partitions. Each split covers the same base corpus.}}
    \label{{tab:data}}
\end{{table}}'''


def per_class_summaries(runs):
    grouped = {}
    for run in runs:
        if run.get('metadata', {}).get('frozen_transformer_layers') != 0:
            continue
        evaluation = run.get('evaluation') or {}
        for class_name in ('Non-Euphemistic', 'Euphemistic'):
            scores = (evaluation.get('per_class') or {}).get(class_name)
            if not scores:
                continue
            key = str(run.get('metadata', {}).get('split_id', 'unknown'))
            group = grouped.setdefault(key, {}).setdefault(class_name, {
                'precision': [], 'recall': [], 'f1-score': [],
            })
            for metric in group:
                value = scores.get(metric)
                if value is not None:
                    group[metric].append(float(value))

    return grouped


def performance_table(groups, split_names):
    rows = []
    for group_index, group in enumerate(groups):
        if group_index:
            rows.append('        \\midrule')
        perf = group['performance']
        rows.append(
            f"        {fmt_split(group['split_id'], split_names)} & "
            f"{fmt_mean(perf['accuracy'], 3)} & "
            f"{fmt_mean(perf['macro_f1'], 3)} \\\\")
    body = '\n'.join(rows)
    return rf'''\begin{{table}}[h]
    \centering\small
    \begin{{tabular}}{{lrr}}
        \toprule
        {header_row(PERFORMANCE_COLUMN_HEADERS)}
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
    \caption{{Mean overall evaluation performance, summarized separately by held-out split.}}
    \label{{tab:perf}}
\end{{table}}'''


def per_class_performance_table(groups, runs, split_names):
    per_class = per_class_summaries(runs)
    rows = []
    for group_index, group in enumerate(groups):
        if group_index:
            rows.append('        \\midrule')
        split_id = str(group['split_id'])
        class_groups = per_class.get(split_id, {})
        for class_index, class_name in enumerate(('Non-Euphemistic', 'Euphemistic')):
            scores = class_groups.get(class_name, {})
            split_name = fmt_split(split_id, split_names) if class_index == 0 else ''
            rows.append(
                f"        {split_name} & {CLASS_DISPLAY_NAMES[class_name]} & "
                f"{fmt_mean(summary(scores.get('precision', [])), 3)} & "
                f"{fmt_mean(summary(scores.get('recall', [])), 3)} & "
                f"{fmt_mean(summary(scores.get('f1-score', [])), 3)} \\\\")
    body = '\n'.join(rows)
    return rf'''\begin{{table}}[h]
    \centering\small
    \begin{{tabular}}{{llrrr}}
        \toprule
        {header_row(PER_CLASS_COLUMN_HEADERS)}
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
    \caption{{Mean performance for each class, summarized separately by held-out split.}}
    \label{{tab:class-perf}}
\end{{table}}'''


def summary(values):
    if not values:
        return None
    return {
        'mean': sum(values) / len(values),
        'min': min(values),
        'max': max(values),
    }


def attention_masking_table(groups, metrics, layers, split_names):
    rows = []
    for group_index, group in enumerate(groups):
        if group_index:
            rows.append('        \\midrule')
        adjusted_index = {
            (row['layer'], row['metric']): row
            for row in group.get('pet_context_length_adjustment', [])
        }
        controls = {
            (row['profile'], row['layer'], row['metric']): row
            for row in group.get('attention_controls', [])
        }
        split_name = fmt_split(group['split_id'], split_names)
        rows.append(
            f"        \\multicolumn{{6}}{{l}}{{\\textbf{{{split_name}}}}} \\\\")
        for layer in layers:
            for metric in metrics:
                adjusted = adjusted_index.get((layer, metric))
                real = controls.get(('real_pet', layer, metric))
                masked = controls.get(('masked_pet', layer, metric))
                random_span = controls.get(('random_span', layer, metric))
                if not adjusted and not real and not masked and not random_span:
                    continue
                masked_sig = (masked or {}).get('bonferroni_significant_runs', 0)
                masked_n = (masked or {}).get('n_runs', 0)
                random_sig = (random_span or {}).get('bonferroni_significant_runs', 0)
                random_n = (random_span or {}).get('n_runs', 0)
                rows.append(
                    f"        {layer} & {metric.split(' (', 1)[0]} & "
                    f"{fmt_signed_range((adjusted or {}).get('raw_d'))} & "
                    f"{fmt_signed_range((adjusted or {}).get('length_adjusted_d'))} & "
                    f"{fmt_signed_range(masked)} ({masked_sig}/{masked_n}) & "
                    f"{fmt_signed_range(random_span)} ({random_sig}/{random_n}) \\\\")
    body = '\n'.join(rows)
    return rf'''\begin{{table*}}[t]
    \centering\small
    \setlength{{\tabcolsep}}{{6pt}}
    \begin{{tabular}}{{@{{}}clllll@{{}}}}
        \toprule
        {header_row(ATTENTION_MASKING_COLUMN_HEADERS)}
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
    \caption{{Class effect sizes ({CLASS_CONTRAST_LABEL}) for the primary PET attention flows in layers 10--11. Effect-size ranges show observed run-to-run variation within each split. Joint-adjusted values account for PET and context length. Raw and jointly adjusted effects are Bonferroni-significant in every run for all listed layer-metric pairs. Masked and random-span show the number of runs with a Bonferroni-significant class contrast (84 metric-by-layer tests per run).}}
    \label{{tab:attn-effects}}
\end{{table*}}'''


def label_shuffle_table(groups):
    rows = []
    for group in groups:
        split_name = fmt_split(group['split_id'], ATTENTION_EFFECT_SPLIT_NAMES)
        for shuffle in group.get('label_shuffle', []):
            run_name = shuffle.get('run', '')
            seed = run_name.split('_train-', 1)[-1].split('_freeze-', 1)[0]
            rows.append(
                f"        {split_name} & {seed} & "
                f"{shuffle['mean_significant_tests']:.2f} & "
                f"{shuffle['max_significant_tests']} & "
                f"{shuffle['permutations_with_layer10_significance']}/"
                f"{shuffle['n_permutations']} & "
                f"{shuffle['real_layer10_significant_metrics']}/"
                f"{shuffle.get('n_metrics', 7)} & "
                f"{shuffle['permutations_with_layer11_significance']}/"
                f"{shuffle['n_permutations']} & "
                f"{shuffle['real_layer11_significant_metrics']}/"
                f"{shuffle.get('n_metrics', 7)} \\\\")
    body = '\n'.join(rows)
    return rf'''\begin{{table*}}[t]
    \centering\small
    \setlength{{\tabcolsep}}{{4pt}}
    \begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
        \toprule
        Split & Seed & Mean / shuffle & Max / shuffle & L10 shuffle hits & L10 real sig. & L11 shuffle hits & L11 real sig. \\
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
    \caption{{Label-shuffle results over the full 84-test family (seven metrics across 12 layers), with layer-specific permutation hits and real-data significant metric counts.}}
    \label{{tab:shuffle}}
\end{{table*}}'''


def appendix_control_table(groups, profile, caption, label, layers=TABLE_LAYERS):
    rows = []
    flip_rates = []
    for group in groups:
        split_name = fmt_split(group['split_id'], ATTENTION_EFFECT_SPLIT_NAMES)
        if profile == 'masked_pet':
            flip_rate = group.get('prediction_flip_rate') or {}
            if flip_rate.get('mean') is not None:
                flip_rates.append(
                    f"{split_name}: {100 * flip_rate['mean']:.1f}\\% "
                    f"[{100 * flip_rate['min']:.1f}, {100 * flip_rate['max']:.1f}]"
                )
        controls = {
            (row['layer'], row['metric']): row
            for row in group.get('attention_controls', [])
            if row['profile'] == profile
        }
        for layer in layers:
            for metric in FINETUNING_DIFF_METRICS:
                metric_name = metric_names_for_label(metric)
                result = controls.get((layer, metric_name))
                if not result:
                    continue
                rows.append(
                    f"        {split_name} & {layer} & {fmt_metric(metric_name)} & "
                    f"{fmt_signed_range(result)} & "
                    f"{result['bonferroni_significant_runs']}/{result['n_runs']} \\\\")
    body = '\n'.join(rows)
    if flip_rates:
        caption += ' PET-mask prediction-flip rates: ' + '; '.join(flip_rates) + '.'
    return rf'''\begin{{table*}}[t]
    \centering\small
    \setlength{{\tabcolsep}}{{5pt}}
    \begin{{tabular}}{{@{{}}lllrr@{{}}}}
        \toprule
        Split & Layer & Metric & Cohen's $d$ mean [range] & Significant runs \\
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
    \caption{{{caption}}}
    \label{{{label}}}
\end{{table*}}'''


def metric_names_for_label(key):
    return {
        'inward': 'Inward (Context → PET)',
        'internal': 'Internal (PET → PET)',
        'outward': 'Outward (PET → Context)',
        'pet-sink': 'Sink (PET → Sink)',
        'sink-pet': 'Sink (Sink → PET)',
        'entropy': 'Entropy',
        'distance': 'Distance',
    }[key]


def paired_pet_random_table(groups, layers=TABLE_LAYERS):
    rows = []
    for group in groups:
        split_name = fmt_split(group['split_id'], ATTENTION_EFFECT_SPLIT_NAMES)
        effects = {
            (item['layer'], item['metric']): item
            for item in group.get('paired_pet_random_contrasts', [])
        }
        for layer in layers:
            for metric in (metric_names_for_label('inward'),
                           metric_names_for_label('internal')):
                item = effects.get((layer, metric))
                if not item:
                    continue
                rows.append(
                    f"        {split_name} & {layer} & {fmt_metric(metric)} & "
                    f"{fmt_signed_range(item)} & "
                    f"{item['bonferroni_significant_runs']}/{item['n_runs']} \\\\")
    body = '\n'.join(rows)
    return rf'''\begin{{table*}}[t]
    \centering\small
    \setlength{{\tabcolsep}}{{5pt}}
    \begin{{tabular}}{{@{{}}lllrr@{{}}}}
        \toprule
        Split & Layer & Metric & Paired-contrast $d$ mean [range] & Significant runs \\
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
    \caption{{For each example, the paired contrast is the real-PET attention value minus the length-matched random-span value. Cohen's $d$ measures the Euphemistic--Non-Euphemistic difference in these per-example contrasts; ranges show run-to-run variation. Significance counts use Mann--Whitney tests with Bonferroni correction over 84 metric-by-layer tests per run.}}
    \label{{tab:pet-rand-diff}}
\end{{table*}}'''


def joint_length_adjustment_table(groups, layers=TABLE_LAYERS):
    rows = []
    for group in groups:
        split_name = fmt_split(group['split_id'], ATTENTION_EFFECT_SPLIT_NAMES)
        adjustments = {
            (row['layer'], row['metric']): row
            for row in group.get('pet_context_length_adjustment', [])
        }
        for layer in layers:
            for metric_key in ('inward', 'internal', 'outward', 'sink-pet'):
                metric = metric_names_for_label(metric_key)
                result = adjustments.get((layer, metric))
                if not result:
                    continue
                tested_runs = result.get('n_tested_runs', result.get('n_runs', 0))
                rows.append(
                    f"        {split_name} & {layer} & {fmt_metric(metric)} & "
                    f"{fmt_signed_range(result.get('raw_d'))} $\\to$ "
                    f"{fmt_signed_range(result.get('length_adjusted_d'))} & "
                    f"{result.get('bonferroni_significant_runs', 0)}/{tested_runs} \\\\")
    body = '\n'.join(rows)
    return rf'''\begin{{table*}}[t]
    \centering\small
    \setlength{{\tabcolsep}}{{4pt}}
    \begin{{tabular}}{{@{{}}lllrr@{{}}}}
        \toprule
        Split & Layer & Metric & Raw $d$ $\to$ adjusted $d$ [range] & Adjusted significant runs \\
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
    \caption{{Joint PET- and context-length adjustment for the four context-related metrics in layers 10--11. Effects summarize four runs within each split. Significance is the class coefficient from HC3 robust regression on class, PET length, and context length, with Bonferroni correction over 84 tests per run.}}
    \label{{tab:length-joint}}
\end{{table*}}'''


def correctness_table(groups, layers=TABLE_LAYERS):
    selected_metrics = ('inward', 'internal', 'pet-sink', 'sink-pet')
    rows = []
    for group in groups:
        split_name = fmt_split(group['split_id'], ATTENTION_EFFECT_SPLIT_NAMES)
        effects = {
            (row['layer'], row['metric'], row['outcome']): row
            for row in group.get('layer10_11_correctness', [])
        }
        for layer in layers:
            for metric_key in selected_metrics:
                metric = metric_names_for_label(metric_key)
                correct = effects.get((layer, metric, 'correct'))
                misclassified = effects.get((layer, metric, 'misclassified'))
                if not correct and not misclassified:
                    continue
                rows.append(
                    f"        {split_name} & {layer} & {fmt_metric(metric)} & "
                    f"{fmt_signed_range(correct)} & {fmt_signed_range(misclassified)} \\\\")
    body = '\n'.join(rows)
    return rf'''\begin{{table*}}[t]
    \centering\small
    \setlength{{\tabcolsep}}{{5pt}}
    \begin{{tabular}}{{@{{}}lllrr@{{}}}}
        \toprule
        Split & Layer & Metric & Correctly classified $d$ [range] & Misclassified $d$ [range] \\
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
    \caption{{Classification-outcome stratification for selected late-layer metrics. Effect sizes are descriptive euphemistic--non-euphemistic contrasts within each correctness stratum and summarize four runs per split.}}
    \label{{tab:misclass}}
\end{{table*}}'''


def finetuning_difference_table(groups, metrics, layers):
    effects_by_group = []
    for group in groups:
        effects_by_group.append({
            (row['layer'], row['metric']): row.get('finetuning_difference')
            for row in group.get('layer_effects', [])
        })

    pooled_effects = {}
    for layer in layers:
        for metric in metrics:
            split_effects = [
                group_effects[(layer, metric)]
                for group_effects in effects_by_group
                if group_effects.get((layer, metric))
            ]
            if split_effects:
                pooled_effects[(layer, metric)] = {
                    'mean': sum(effect['mean'] for effect in split_effects) / len(split_effects),
                    'min': min(effect['min'] for effect in split_effects),
                    'max': max(effect['max'] for effect in split_effects),
                }

    rows = []
    for metric in metrics:
        metric_label = {
            'Sink (PET → Sink)': 'To-Sink',
            'Sink (Sink → PET)': 'From-Sink',
        }.get(metric, metric.split(' (', 1)[0])
        layer_values = [
            fmt_signed_range(pooled_effects[(layer, metric)])
            if (layer, metric) in pooled_effects else '--'
            for layer in layers
        ]
        rows.append(
            f"        {metric_label} & "
            f"{layer_values[0]} & {layer_values[1]} \\\\")
    body = '\n'.join(rows)
    return rf'''\begin{{table}}[t]
    \centering\small
    \setlength{{\tabcolsep}}{{4pt}}
    \begin{{tabular}}{{@{{}}lcc@{{}}}}
        \toprule
        Metric & Layer 10 diff. $d$ & Layer 11 diff. $d$ \\
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
    \caption{{Finetuning difference effect sizes for all seven metrics in layers 10--11. Each effect is Cohen's $d$ for the class contrast in per-sample finetuned-minus-base changes. Means equally weight the two splits; brackets show the observed minimum and maximum across all eight runs.}}
    \label{{tab:ft-diff-effects}}
\end{{table}}'''


def main():
    with AGGREGATE_PATH.open(encoding='utf-8') as source:
        aggregate = json.load(source)

    # The fully unfrozen model is the paper's primary model. Keep split
    # protocols separate; the aggregation deliberately does not pool them.
    full_finetune_groups = [
        group for group in aggregate['comparisons_by_split_and_freeze']
        if group.get('frozen_transformer_layers') == 0
    ]
    if not full_finetune_groups:
        raise ValueError('No fully unfrozen groups found in aggregate summary.')

    metric_names = {
        'inward': 'Inward (Context → PET)',
        'internal': 'Internal (PET → PET)',
        'outward': 'Outward (PET → Context)',
        'pet-sink': 'Sink (PET → Sink)',
        'sink-pet': 'Sink (Sink → PET)',
        'entropy': 'Entropy',
        'distance': 'Distance',
    }
    selected_metrics = [metric_names[name] for name in TABLE_METRICS]
    tables = [
        '% Generated by generate_tables.py; edit the source data, not this file.',
        dataset_table([(DATA_SPLIT_NAMES['42'], SPLITS_DIR / '42'),
                       (DATA_SPLIT_NAMES['lrec'], SPLITS_DIR / 'lrec')]),
        performance_table(full_finetune_groups, PERFORMANCE_SPLIT_NAMES),
        per_class_performance_table(
            full_finetune_groups, aggregate['runs'], PER_CLASS_SPLIT_NAMES,
        ),
        attention_masking_table(
            full_finetune_groups, selected_metrics, TABLE_LAYERS,
            ATTENTION_EFFECT_SPLIT_NAMES,
        ),
        '% Appendix robustness-control tables',
        label_shuffle_table(full_finetune_groups),
        appendix_control_table(
            full_finetune_groups, 'masked_pet',
            'PET-token masking results by split and layer. Each entry is the mean [minimum, maximum] across training runs; significance counts use the per-run Bonferroni correction over 84 tests.',
            'tab:masking',
        ),
        appendix_control_table(
            full_finetune_groups, 'random_span',
            'Length-matched random-span results by split and layer. Each entry is the mean [minimum, maximum] across training runs; significance counts use the per-run Bonferroni correction over 84 tests.',
            'tab:randspan',
        ),
        paired_pet_random_table(full_finetune_groups),
        joint_length_adjustment_table(full_finetune_groups),
        correctness_table(full_finetune_groups),
        finetuning_difference_table(
            full_finetune_groups,
            [metric_names[name] for name in FINETUNING_DIFF_METRICS], TABLE_LAYERS,
        ),
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text('\n\n'.join(tables) + '\n', encoding='utf-8')
    print(f'Wrote {len(tables) - 1} LaTeX tables to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
