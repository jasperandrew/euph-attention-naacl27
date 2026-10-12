"""Profile one held-out sample and check the PET attention partition.

By default, the script profiles row 0 of the selected run's held-out test
split. Select another row with ``--sample-index`` or select by dataset ID with
``--sample-id``. It writes one row per layer/head to CSV and a JSON report with
the same profile plus the conservation check.

Example:
  python profile_sample.py --model-dir runs/split-42_train-44_freeze-0
  python profile_sample.py --model-dir runs/split-42_train-44_freeze-0 --sample-id 814
"""
import argparse
import os

from src.cli import add_json_arg, add_model_arg, init_model, write_json


_PARTITION_METRICS = [
    'Outward (PET → Context)',
    'Internal (PET → PET)',
    'Sink (PET → Sink)',
]


def main(args):
    # Import model/data code after selecting the run, so its tokenizer and split
    # are resolved from that run rather than from a global default.
    from src import config
    from src.data_utils import load_splits
    from src.profiler.analysis import profile_single_sample
    from src.stats.models import load_classifier

    _, test_df = load_splits()
    if args.sample_id is not None:
        matches = test_df[test_df['id'].astype(str) == args.sample_id]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one test row with id {args.sample_id!r}; found {len(matches)}."
            )
        row = matches.iloc[0]
        row_index = int(test_df.index.get_loc(row.name))
    else:
        if not 0 <= args.sample_index < len(test_df):
            raise IndexError(
                f"sample-index {args.sample_index} is outside the test split "
                f"(0 to {len(test_df) - 1})."
            )
        row_index = args.sample_index
        row = test_df.iloc[row_index]

    model = load_classifier(config.MODEL_DIR)
    profiles, prediction = profile_single_sample(
        model,
        str(row[config.DATA_COLUMN]),
        str(row['pet_instance']),
    )
    if profiles is None:
        raise ValueError(
            f"Could not locate PET {row['pet_instance']!r} in test row "
            f"{row['id']!r}."
        )

    profiles['PET partition sum'] = profiles[_PARTITION_METRICS].sum(axis=1)
    profiles['Absolute error from 1'] = (profiles['PET partition sum'] - 1.0).abs()
    max_error = float(profiles['Absolute error from 1'].max())
    passed = max_error <= args.tolerance

    csv_path = args.csv_out or os.path.splitext(args.json_out)[0] + '.csv'
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    profiles.to_csv(csv_path, index=False)

    label = 'Euphemistic' if str(row['label']) in {
        '1', '1.0', 'True', 'euph', 'Euphemistic'
    } else 'Non-Euphemistic'
    result = {
        'script': 'profile_sample.py',
        'model_dir': config.MODEL_DIR,
        'split_dir': config.MODEL_RUN_DIR,
        'sample': {
            'test_row_index': row_index,
            'id': row['id'],
            'pet': row['pet_instance'],
            'gold_label': label,
            'prediction': prediction,
            'text': row[config.DATA_COLUMN],
        },
        'partition_check': {
            'metrics': _PARTITION_METRICS,
            'identity': 'outward + internal + sink = 1',
            'tolerance': args.tolerance,
            'profile_rows_checked': int(len(profiles)),
            'layer_count': int(profiles['Layer'].nunique()),
            'head_count': int(profiles['Head'].nunique()),
            'sum_min': float(profiles['PET partition sum'].min()),
            'sum_max': float(profiles['PET partition sum'].max()),
            'max_absolute_error': max_error,
            'passed': passed,
        },
        'profiles': profiles,
        'csv_out': os.path.abspath(csv_path),
    }
    write_json(args.json_out, result)

    print(f"Sample {row['id']} | PET={row['pet_instance']!r} | "
          f"gold={label} | prediction={prediction}")
    print(f"Checked {len(profiles)} layer/head rows; partition sum range "
          f"{result['partition_check']['sum_min']:.9f}–"
          f"{result['partition_check']['sum_max']:.9f}.")
    print(f"Maximum absolute error: {max_error:.3g} "
          f"(tolerance {args.tolerance:g}) — {'PASS' if passed else 'FAIL'}")
    print(f"CSV: {os.path.abspath(csv_path)}")
    print(f"JSON: {os.path.abspath(args.json_out)}")

    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Profile one held-out sample and verify its PET attention partition.'
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--sample-id', help='Dataset id of a row in the held-out test split.')
    selection.add_argument('--sample-index', type=int, default=0,
                           help='Zero-based row index in the held-out test split (default: 0).')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                        help='Maximum allowed absolute difference of the partition sum from 1.')
    parser.add_argument('--csv-out', metavar='PATH',
                        help='CSV output path (default: beside the JSON output).')
    add_model_arg(parser)
    add_json_arg(parser)
    args = parser.parse_args()
    init_model(args)
    main(args)
