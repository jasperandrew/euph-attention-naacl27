"""
Single-sentence profiler entry point.

Runs the unpaired attention profiler across the held-out test split, aggregating metrics
per layer and condition. By default also profiles the base XLM-R model and emits
a ft−base difference plot. Always produces the layer-11 per-head lollipop plot.

Profiling data is cached under output/plots/<run-name>/data/ as parquet files
and reused on subsequent runs. Use --no-cache to force re-running inference.

Usage:
  python plot_profile.py --model-dir runs/en/<run-name>
  python plot_profile.py --model-dir runs/en/<run-name>
  python plot_profile.py --model-dir runs/en/<run-name> --layers unfrozen
  python plot_profile.py --model-dir runs/en/<run-name> --skip-base
  python plot_profile.py --model-dir runs/en/<run-name> --max-samples 50
  python plot_profile.py --model-dir runs/en/<run-name> --no-cache
"""
import argparse
import logging
import os

from src.cli import add_json_arg, add_model_arg, init_model, write_json
from src.stats.models import load_classifier

logger = logging.getLogger(__name__)


_METRIC_COLS = [
    'Outward (PET → Context)',
    'Internal (PET → PET)',
    'Sink (PET → Sink)',
    'Sink (Sink → PET)',
    'Inward (Context → PET)',
    'Entropy',
    'Distance',
    'Head Variance',
]


def _data_dir():
    return os.path.join(config.PLOTS_DIR, config.MODEL_RUN_NAME, 'data')


def _cache_path(name):
    return os.path.join(_data_dir(), f'{name}.parquet')


def _load_cache(name):
    import pandas as pd
    path = _cache_path(name)
    if os.path.exists(path):
        logger.info(f"  Loading cached {name} from {path}")
        cached = pd.read_parquet(path)
        missing = set(_METRIC_COLS).difference(cached.columns)
        if missing:
            logger.info("  Cached profile lacks new metric(s) %s; recomputing", sorted(missing))
            return None
        return cached
    return None


def _save_cache(df, name):
    os.makedirs(_data_dir(), exist_ok=True)
    path = _cache_path(name)
    df.to_parquet(path, index=False)
    logger.info(f"  Cached {name} to {path}")


def load_model(model_id_or_dir):
    return load_classifier(model_id_or_dir)


def profile(model, test_df, max_samples):
    return profile_dataframe(
        model, test_df,
        text_col=config.DATA_COLUMN,
        pet_col='pet_instance',
        label_col='label',
        max_samples=max_samples,
    )


def main(args):
    profile_dir = os.path.join(config.PLOTS_DIR, config.MODEL_RUN_NAME)
    os.makedirs(profile_dir, exist_ok=True)
    use_cache = not args.no_cache and args.max_samples is None

    logger.info("Loading held-out test split...")
    _, test_df = load_splits()
    logger.info(f"  {len(test_df)} samples")

    ft_df = _load_cache('ft_df') if use_cache else None
    if ft_df is None:
        logger.info("Profiling finetuned model...")
        ft_model = load_model(config.MODEL_DIR)
        ft_df = profile(ft_model, test_df, args.max_samples)
        del ft_model
        torch.cuda.empty_cache()
        if ft_df is None:
            logger.info("No valid profiles extracted.")
            return
        if use_cache:
            _save_cache(ft_df, 'ft_df')

    unfrozen_only = (args.layers == 'unfrozen')
    results = {
        'script': 'plot_profile.py',
        'model_dir': config.MODEL_DIR,
        'split_dir': config.MODEL_RUN_DIR,
        'n_profile_rows': len(ft_df),
        'finetuned_profiles': ft_df,
    }
    plot_aggregate_profiles(ft_df, output_dir=profile_dir, unfrozen_only=unfrozen_only)
    plot_layer11_lollipop(ft_df, output_dir=profile_dir)
    if 'Predicted' in ft_df.columns:
        plot_correct_vs_misclassified(ft_df, output_dir=profile_dir)

    mask_df = _load_cache('mask_df') if use_cache else None
    if mask_df is not None and 'Sample' in mask_df.columns:
        plot_real_vs_masked(ft_df, mask_df, output_dir=profile_dir)

    if not args.skip_base:
        base_df = _load_cache('base_df') if use_cache else None
        if base_df is None:
            logger.info("Profiling base model...")
            base_model = load_model(config.MODEL_NAME)
            base_df = profile(base_model, test_df, args.max_samples)
            del base_model
            torch.cuda.empty_cache()
            if use_cache:
                _save_cache(base_df, 'base_df')

        if len(ft_df) != len(base_df):
            raise RuntimeError(f"Row count mismatch: ft={len(ft_df)} base={len(base_df)}")
        ft_df = ft_df.reset_index(drop=True)
        base_df = base_df.reset_index(drop=True)
        diff_df = ft_df.copy()
        for col in _METRIC_COLS:
            diff_df[col] = ft_df[col].values - base_df[col].values
        results['base_profiles'] = base_df
        results['finetuned_minus_base_profiles'] = diff_df
        plot_finetuning_diff_profile(diff_df, output_dir=profile_dir, unfrozen_only=unfrozen_only)
        plot_base_vs_finetuned(ft_df, base_df, output_dir=profile_dir, unfrozen_only=unfrozen_only)

    logger.info(f"Plots saved to {profile_dir}/")
    write_json(args.json_out, results)
    if args.json_out:
        logger.info(f"JSON results written to {args.json_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single-sentence euphemism attention profiler")
    parser.add_argument('--max-samples', type=int, default=None,
                        help="Limit number of test samples (useful for quick testing); disables cache")
    parser.add_argument('--layers', choices=['all', 'unfrozen'], default='all',
                        help="Which layers to include in plots (default: all)")
    parser.add_argument('--skip-base', action='store_true',
                        help="Skip the base-model pass and the ft−base diff plot.")
    parser.add_argument('--no-cache', action='store_true',
                        help="Force re-running inference and overwrite cached parquet files.")
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed profiling and cache messages.')
    add_model_arg(parser)
    add_json_arg(parser)
    args = parser.parse_args()
    init_model(args)

    from src import config
    from src.data_utils import load_splits
    from src.profiler.analysis import profile_dataframe
    from src.profiler.plots import plot_aggregate_profiles, plot_finetuning_diff_profile, plot_base_vs_finetuned, plot_layer11_lollipop, plot_correct_vs_misclassified, plot_real_vs_masked

    main(args)
