"""Consolidated significance and verification tests.

Profiles the finetuned and base XLM-R models on the eval split once each, then runs:

  Section 1 — Per-layer mean values by condition; frozen vs unfrozen |Δ| ratios.
  Section 2 — Per-(metric, layer) significance on finetuned model.
  Section 3 — Per-(metric, head) significance at layers 11 and 12, finetuned model.
  Section 4 — Side-by-side base vs finetuned d per (metric, layer); reports Δd.
  Section 5 — Per-(metric, layer) significance on (ft − base) diffs (interaction).

Usage:
  python significance.py
  python significance.py --max-samples 100
  python significance.py --skip-base   # sections 1–3 only
"""
import argparse

import numpy as np
import scipy.stats
import torch
from transformers import AutoModelForSequenceClassification

from src import config
from src.data_utils import load_splits
from src.profiler.analysis import profile_dataframe


_METRICS = ['Inward (Context → PET)', 'Outward (PET → Context)',
            'Internal (PET → PET)', 'Entropy', 'Sink Fraction', 'Attn Distance']


def load_model(model_id_or_dir):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id_or_dir, output_attentions=True, attn_implementation="eager",
        num_labels=2,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / s if s > 0 else 0.0


def sig_marker(p_bonf):
    return '***' if p_bonf < 0.001 else '**' if p_bonf < 0.01 else '*' if p_bonf < 0.05 else ''


def profile_with_sample_id(model, eval_df, max_samples=None):
    df = profile_dataframe(model, eval_df, text_col=config.DATA_COLUMN,
                           pet_col='pet_instance', label_col='label',
                           max_samples=max_samples)
    df = df.copy().reset_index(drop=True)
    df['sample_id'] = df.index // (config.NUM_LAYERS * config.NUM_HEADS)
    return df


def per_sample_collapse(df):
    return df.groupby(['sample_id', 'Layer', 'Condition'])[_METRICS].mean().reset_index()


def section(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def section_1_means(ft_df):
    section("Section 1 — Mean per-layer values by condition (verify_claims)")
    e = ft_df[ft_df['Condition'] == 'Euphemistic']
    n = ft_df[ft_df['Condition'] == 'Non-Euphemistic']
    print(f"\n  {'Layer':>6} {'Metric':>26} {'Euph':>10} {'Non':>10} {'Diff':>10} {'%Δ':>8}")
    for layer in range(1, config.NUM_LAYERS + 1):
        for m in _METRICS:
            em = e[e['Layer'] == layer][m].mean()
            nm = n[n['Layer'] == layer][m].mean()
            diff = em - nm
            pct = 100 * diff / nm if nm != 0 else float('nan')
            mark = '  [frozen]' if layer <= config.FROZEN_LAYERS else ''
            print(f"  {layer:>6} {m:>26} {em:>10.4f} {nm:>10.4f} {diff:>+10.4f} {pct:>+7.1f}%{mark}")
        print()

    print(f"\n  Frozen vs unfrozen mean |Δ| per metric")
    print(f"  {'Metric':>26} {'frozen |Δ|':>12} {'unfrozen |Δ|':>14} {'ratio':>8}")
    for m in _METRICS:
        fz, uf = [], []
        for layer in range(1, config.NUM_LAYERS + 1):
            em = e[e['Layer'] == layer][m].mean()
            nm = n[n['Layer'] == layer][m].mean()
            (fz if layer <= config.FROZEN_LAYERS else uf).append(abs(em - nm))
        fz_m, uf_m = float(np.mean(fz)), float(np.mean(uf))
        ratio = uf_m / fz_m if fz_m > 0 else float('inf')
        print(f"  {m:>26} {fz_m:>12.4f} {uf_m:>14.4f} {ratio:>8.2f}x")


def section_2_ft_layer(ft_ps):
    section("Section 2 — Per-(metric, layer) significance, finetuned model")
    n_tests = len(_METRICS) * config.NUM_LAYERS
    thresh = 0.05 / n_tests
    print(f"  {n_tests} tests; Bonferroni p < {thresh:.2e}\n")
    print(f"  {'Metric':>26} {'Layer':>6} {'mean Euph':>10} {'mean Non':>10} "
          f"{'d':>8} {'p (raw)':>11} {'p (bonf)':>11} {'sig':>5}")
    for m in _METRICS:
        for layer in range(1, config.NUM_LAYERS + 1):
            sub = ft_ps[ft_ps['Layer'] == layer]
            e = sub[sub['Condition'] == 'Euphemistic'][m].values
            n = sub[sub['Condition'] == 'Non-Euphemistic'][m].values
            _, p = scipy.stats.mannwhitneyu(e, n, alternative='two-sided')
            p_b = min(p * n_tests, 1.0)
            d = cohens_d(e, n)
            mark = '  [frozen]' if layer <= config.FROZEN_LAYERS else ''
            print(f"  {m:>26} {layer:>6} {e.mean():>10.4f} {n.mean():>10.4f} "
                  f"{d:>+8.3f} {p:>11.2e} {p_b:>11.2e} {sig_marker(p_b):>5}{mark}")
        print()


def section_3_heads(ft_df):
    section("Section 3 — Per-(metric, head) significance at layers 11, 12, finetuned model")
    n_tests = len(_METRICS) * config.NUM_HEADS
    thresh = 0.05 / n_tests
    for layer in [11, 12]:
        print(f"\n  Layer {layer} — {n_tests} tests, Bonferroni p < {thresh:.2e}\n")
        print(f"  {'Metric':>26} {'Head':>4} {'d':>8} {'p (raw)':>11} {'p (bonf)':>11} {'sig':>5}")
        ldf = ft_df[ft_df['Layer'] == layer]
        for m in _METRICS:
            for head in range(1, config.NUM_HEADS + 1):
                sub = ldf[ldf['Head'] == head]
                e = sub[sub['Condition'] == 'Euphemistic'][m].values
                n = sub[sub['Condition'] == 'Non-Euphemistic'][m].values
                _, p = scipy.stats.mannwhitneyu(e, n, alternative='two-sided')
                p_b = min(p * n_tests, 1.0)
                d = cohens_d(e, n)
                print(f"  {m:>26} {head:>4} {d:>+8.3f} {p:>11.2e} {p_b:>11.2e} {sig_marker(p_b):>5}")
            print()


def section_4_base_vs_ft(base_ps, ft_ps):
    section("Section 4 — Base vs finetuned d per (metric, layer); Δd = d_ft − d_base")
    n_tests = len(_METRICS) * config.NUM_LAYERS
    thresh = 0.05 / n_tests
    print(f"  Bonferroni p < {thresh:.2e}\n")
    print(f"  {'Metric':>26} {'Layer':>6} {'d_base':>8} {'d_ft':>8} {'Δd':>8} "
          f"{'p_base':>13} {'p_ft':>13}")
    for m in _METRICS:
        for layer in range(1, config.NUM_LAYERS + 1):
            sb = base_ps[base_ps['Layer'] == layer]
            sf = ft_ps[ft_ps['Layer'] == layer]
            eb = sb[sb['Condition'] == 'Euphemistic'][m].values
            nb = sb[sb['Condition'] == 'Non-Euphemistic'][m].values
            ef = sf[sf['Condition'] == 'Euphemistic'][m].values
            nf = sf[sf['Condition'] == 'Non-Euphemistic'][m].values
            _, pb = scipy.stats.mannwhitneyu(eb, nb, alternative='two-sided')
            _, pf = scipy.stats.mannwhitneyu(ef, nf, alternative='two-sided')
            pb_b = min(pb * n_tests, 1.0)
            pf_b = min(pf * n_tests, 1.0)
            d_b = cohens_d(eb, nb)
            d_f = cohens_d(ef, nf)
            frozen = '  [frozen]' if layer <= config.FROZEN_LAYERS else ''
            print(f"  {m:>26} {layer:>6} {d_b:>+8.3f} {d_f:>+8.3f} {(d_f - d_b):>+8.3f} "
                  f"{pb_b:>10.2e}{sig_marker(pb_b):>3} {pf_b:>10.2e}{sig_marker(pf_b):>3}{frozen}")
        print()


def section_5_diff(ft_df, base_df):
    section("Section 5 — Per-(metric, layer) significance on (ft − base) diffs (interaction)")
    if len(ft_df) != len(base_df):
        print(f"  Skipped — row counts differ (ft={len(ft_df)}, base={len(base_df)}).")
        return
    diff_df = ft_df.copy()
    for col in _METRICS:
        diff_df[col] = ft_df[col].values - base_df[col].values
    diff_ps = per_sample_collapse(diff_df)
    n_tests = len(_METRICS) * config.NUM_LAYERS
    thresh = 0.05 / n_tests
    print(f"  {n_tests} tests; Bonferroni p < {thresh:.2e}\n")
    print(f"  {'Metric':>26} {'Layer':>6} {'mean Δ_E':>11} {'mean Δ_N':>11} "
          f"{'d':>8} {'p (raw)':>11} {'p (bonf)':>11} {'sig':>5}")
    for m in _METRICS:
        for layer in range(1, config.NUM_LAYERS + 1):
            sub = diff_ps[diff_ps['Layer'] == layer]
            e = sub[sub['Condition'] == 'Euphemistic'][m].values
            n = sub[sub['Condition'] == 'Non-Euphemistic'][m].values
            _, p = scipy.stats.mannwhitneyu(e, n, alternative='two-sided')
            p_b = min(p * n_tests, 1.0)
            d = cohens_d(e, n)
            frozen = '  [frozen]' if layer <= config.FROZEN_LAYERS else ''
            print(f"  {m:>26} {layer:>6} {e.mean():>+11.4f} {n.mean():>+11.4f} "
                  f"{d:>+8.3f} {p:>11.2e} {p_b:>11.2e} {sig_marker(p_b):>5}{frozen}")
        print()


def main(args):
    _, eval_df = load_splits()
    print(f"Eval split: {len(eval_df)} samples")

    print("\nProfiling finetuned model...")
    ft_model = load_model(config.MODEL_DIR)
    ft_df = profile_with_sample_id(ft_model, eval_df, args.max_samples)
    del ft_model
    torch.cuda.empty_cache()
    ft_ps = per_sample_collapse(ft_df)

    base_df, base_ps = None, None
    if not args.skip_base:
        print("Profiling base model...")
        base_model = load_model(config.MODEL_NAME)
        base_df = profile_with_sample_id(base_model, eval_df, args.max_samples)
        del base_model
        torch.cuda.empty_cache()
        base_ps = per_sample_collapse(base_df)

    section_1_means(ft_df)
    section_2_ft_layer(ft_ps)
    section_3_heads(ft_df)
    if base_ps is not None:
        section_4_base_vs_ft(base_ps, ft_ps)
        section_5_diff(ft_df, base_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--skip-base', action='store_true',
                        help='Skip base-model passes (sections 1–3 only).')
    args = parser.parse_args()
    main(args)
