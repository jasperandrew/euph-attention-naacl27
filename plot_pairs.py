"""
Paired analysis entry point.

Runs all analyses that compare a euphemistic sentence against its non-euphemistic
counterpart: attention heatmaps, entropy, sink attention, and base-vs-finetuned diffs.

Usage:
  python plot_pairs.py
  python plot_pairs.py --compare-base
  python plot_pairs.py --compare-base --single-head-plots --finetune-head-plots
"""
import argparse
import os

from src.cli import add_lang_arg, init_lang


def load_model(model_dir, output_attentions=True):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        output_attentions=output_attentions,
        attn_implementation="eager",
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model


def main(args):
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    print("Loading finetuned model...")
    ft_model = load_model(config.MODEL_DIR)
    print(f"  Device: {next(ft_model.parameters()).device}")

    base_model = None
    if args.compare_base:
        print("Loading base model for comparison...")
        base_model = load_model(config.MODEL_NAME)

    tokenized_n = tokenize(config.EXAMPLE_EUPH_N)
    tokenized_y = tokenize(config.EXAMPLE_EUPH_Y)
    tokens_n = get_tokens(tokenized_n)
    tokens_y = get_tokens(tokenized_y)
    pet_tokens = config.EXAMPLE_PET_TOKENS

    ex = dict(
        tokens_n=tokens_n, tokenized_n=tokenized_n,
        tokens_y=tokens_y, tokenized_y=tokenized_y,
    )

    # Run forward passes once for the example pair; reused across all analyses below.
    print("\nRunning forward passes for example pair...")
    ft_attn_n = get_attn_weights(ft_model, tokenized_n)
    ft_attn_y = get_attn_weights(ft_model, tokenized_y)
    base_attn_n = get_attn_weights(base_model, tokenized_n) if base_model else None
    base_attn_y = get_attn_weights(base_model, tokenized_y) if base_model else None

    # Per-layer/head heatmaps
    if args.single_head_plots:
        print(f"Saving single-head plots for layers {config.FROZEN_LAYERS+1}–{config.NUM_LAYERS}...")
        out = os.path.join(config.PLOTS_DIR, 'pairs', 'single_head')
        for layer_i in range(config.FROZEN_LAYERS, config.NUM_LAYERS):
            for head_i in range(config.NUM_HEADS):
                plot_euph_attention_analysis(
                    ft_model, layer_i, head_i, pet_tokens,
                    attn_n=ft_attn_n, attn_y=ft_attn_y,
                    output_dir=out, **ex,
                )
        print(f"  Saved to {out}/")

    # Entropy
    print("\nRunning entropy analysis...")
    entropy_dir = os.path.join(config.PLOTS_DIR, 'pairs', 'entropy')
    plot_entropy_by_layer(
        ft_model, pet_tokens, **ex,
        label="Finetuned",
        compare_model=base_model,
        compare_label="Base" if base_model else None,
        attn_n=ft_attn_n, attn_y=ft_attn_y,
        output_dir=entropy_dir,
    )
    print("  Entropy line plot saved.")
    plot_entropy_diff_heatmap(
        ft_model, pet_tokens, **ex,
        label="Finetuned",
        attn_n=ft_attn_n, attn_y=ft_attn_y,
        output_dir=entropy_dir,
    )
    print("  Entropy diff heatmap saved.")

    # Sink attention
    print("\nRunning sink attention analysis...")
    sink_dir = os.path.join(config.PLOTS_DIR, 'pairs', 'sink')
    _, sample_diffs = plot_all_sink_diffs(ft_model, config.SAMPLES, output_dir=sink_dir)

    print("\n--- Layer 12 Summary ---")
    for sample in config.SAMPLES:
        result = sample_diffs.get(sample['pet_phrase'])
        if result is None:
            continue
        diff, _ = result
        print(f"  {sample['pet_phrase']}: {diff[diff['Layer'] == 12]['Sink Fraction'].mean():+.4f}")

    heads_of_interest = [5, 6, 7]
    print(f"\n--- Layer 12, Heads {heads_of_interest} ---")
    for sample in config.SAMPLES:
        result = sample_diffs.get(sample['pet_phrase'])
        if result is None:
            continue
        diff, _ = result
        filtered = diff[(diff['Layer'] == 12) & (diff['Head'].isin(heads_of_interest))]
        print(f"  {sample['pet_phrase']}: {filtered['Sink Fraction'].mean():+.4f}")

    # Base vs. finetuned diffs
    if base_model is not None:
        print("\nSaving base vs. finetuned diff plots...")
        diff_dir = os.path.join(config.PLOTS_DIR, 'pairs', 'finetune_diff')
        plot_avg_diff(base_model, ft_model, pet_tokens, **ex, output_dir=diff_dir)
        print("  Average diff bar chart saved.")

        if args.finetune_head_plots:
            for layer_i in range(config.FROZEN_LAYERS, config.NUM_LAYERS):
                for head_i in range(config.NUM_HEADS):
                    plot_finetune_attn_diff(
                        base_model, ft_model, layer_i, head_i,
                        og_attn_n=base_attn_n, og_attn_y=base_attn_y,
                        ft_attn_n=ft_attn_n, ft_attn_y=ft_attn_y,
                        output_dir=diff_dir, **ex,
                    )
            print(f"  Per-head finetune diff plots saved to {diff_dir}/")

    print(f"\nAll plots saved to {config.PLOTS_DIR}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Paired euphemism attention analysis")
    parser.add_argument('--all', action='store_true',
                        help="Run all analyses (implies --compare-base, --single-head-plots, --finetune-head-plots)")
    parser.add_argument('--compare-base', action='store_true',
                        help="Also load the base model for comparison plots")
    parser.add_argument('--single-head-plots', action='store_true',
                        help="Save per-head attention plots for all non-frozen layers")
    parser.add_argument('--finetune-head-plots', action='store_true',
                        help="Save per-head base-vs-finetuned diff plots (requires --compare-base)")
    add_lang_arg(parser)
    args = parser.parse_args()
    init_lang(args)
    if args.all:
        args.compare_base = True
        args.single_head_plots = True
        args.finetune_head_plots = True

    import torch
    from transformers import AutoModelForSequenceClassification
    from src import config

    if config.LANG not in config.PAIR_EXAMPLES:
        raise SystemExit(
            f"plot_pairs.py needs hand-picked example sentences; add a '{config.LANG}' "
            f"entry to PAIR_EXAMPLES in src/config.py."
        )
    from src.data_utils import tokenize, get_tokens
    from src.attention_utils import get_attn_weights
    from src.pair.plots import (
        plot_euph_attention_analysis,
        plot_finetune_attn_diff,
        plot_avg_diff,
        plot_entropy_by_layer,
        plot_entropy_diff_heatmap,
        plot_all_sink_diffs,
    )

    main(args)
