"""
Single-sentence profiler entry point.

Runs the unpaired attention profiler across the eval split, aggregating metrics
per layer and condition. By default also profiles the base XLM-R model and emits
a ft−base difference plot. Always produces the layer-11 per-head lollipop plot.

Usage:
  python plot_profile.py
  python plot_profile.py --layers unfrozen
  python plot_profile.py --skip-base       # finetuned only, no diff plot
  python plot_profile.py --max-samples 50
"""
import argparse
import os


_METRIC_COLS = [
    'Inward (Context → PET)',
    'Outward (PET → Context)',
    'Internal (PET → PET)',
    'Entropy',
    'Sink Fraction',
    'Attn Distance',
    'Head Variance',
]


def load_model(model_id_or_dir):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id_or_dir,
        output_attentions=True,
        attn_implementation="eager",
        num_labels=2,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model


def profile(model, eval_df, max_samples):
    return profile_dataframe(
        model, eval_df,
        text_col=config.DATA_COLUMN,
        pet_col='pet_instance',
        label_col='label',
        max_samples=max_samples,
    )


def main(args):
    profile_dir = os.path.join(config.PLOTS_DIR, 'profiler')
    os.makedirs(profile_dir, exist_ok=True)

    print("Loading eval split...")
    _, eval_df = load_splits()
    print(f"  {len(eval_df)} samples")

    print("Profiling finetuned model...")
    ft_model = load_model(config.MODEL_DIR)
    ft_df = profile(ft_model, eval_df, args.max_samples)
    del ft_model
    torch.cuda.empty_cache()
    if ft_df is None:
        print("No valid profiles extracted.")
        return

    unfrozen_only = (args.layers == 'unfrozen')
    plot_aggregate_profiles(ft_df, output_dir=profile_dir, unfrozen_only=unfrozen_only)
    plot_layer11_lollipop(ft_df, output_dir=profile_dir)

    if not args.skip_base:
        print("Profiling base model...")
        base_model = load_model(config.MODEL_NAME)
        base_df = profile(base_model, eval_df, args.max_samples)
        del base_model
        torch.cuda.empty_cache()

        if len(ft_df) != len(base_df):
            raise RuntimeError(f"Row count mismatch: ft={len(ft_df)} base={len(base_df)}")
        ft_df = ft_df.reset_index(drop=True)
        base_df = base_df.reset_index(drop=True)
        diff_df = ft_df.copy()
        for col in _METRIC_COLS:
            diff_df[col] = ft_df[col].values - base_df[col].values
        plot_finetuning_diff_profile(diff_df, output_dir=profile_dir, unfrozen_only=unfrozen_only)

    print(f"Plots saved to {profile_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single-sentence euphemism attention profiler")
    parser.add_argument('--max-samples', type=int, default=None,
                        help="Limit number of eval samples (useful for quick testing)")
    parser.add_argument('--layers', choices=['all', 'unfrozen'], default='all',
                        help="Which layers to include in plots (default: all)")
    parser.add_argument('--skip-base', action='store_true',
                        help="Skip the base-model pass and the ft−base diff plot.")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification
    from src import config
    from src.data_utils import load_splits
    from src.profiler.analysis import profile_dataframe
    from src.profiler.plots import plot_aggregate_profiles, plot_finetuning_diff_profile, plot_layer11_lollipop

    main(args)
