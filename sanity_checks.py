"""Six sanity checks for the layer-11 reconfiguration claim.

Each test sample is profiled three ways in a single pass (real PET span, random
non-overlapping span of equal length, and PET tokens replaced with <mask>), then:

  1. Label-shuffle permutation control — should produce ~0 Bonferroni survivors.
  2. PET / sequence length confound — residualize each layer-11 metric on
     (pet_len, seq_len) and report d before vs after.
  3. Random-span (positional PET-specificity) — same input sentence, metrics
     computed on a random non-overlapping span. Tests whether the configuration
     is tied to the PET *location*.
  4. Correct- vs misclassified-sample stratification at layer 11.
  5. PET-mask (lexical PET-specificity) — PET tokens replaced with <mask>,
     model re-run, metrics computed at the (former) PET span. Tests whether the
     configuration depends on PET *content*.
  6. Finetuned − base difference test — base model profiled on the same test
     set; row-wise subtraction isolates finetuning-induced changes. Confirms
     how many layer-11 metrics are significant in the difference test.

Usage:
  python sanity_checks.py --model-dir runs/en/<run-name>
  python sanity_checks.py --model-dir runs/en/<run-name> --max-samples 50
  python sanity_checks.py --model-dir runs/en/<run-name> --n-perms 20 --seed 0
"""
import argparse
import logging
import os
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model
import torch

from src.cli import add_json_arg, add_model_arg, init_model, write_json
from src.stats.models import load_classifier
from src.stats.profiling import METRIC_COLUMNS, profile_attention_span
from src import config
from src.attention_utils import get_model_output, _find_unbounded_pet
from src.data_utils import get_tokenizer, load_splits, tokenize, get_tokens

_METRICS = METRIC_COLUMNS
logger = logging.getLogger(__name__)


def load_model(model_dir):
    return load_classifier(model_dir)


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / s if s > 0 else 0.0


def random_non_overlapping_span(seq_len, pet_start, pet_end, rng):
    pet_len = pet_end - pet_start
    valid = []
    for s in range(1, seq_len - pet_len):  # excludes sinks at 0 and seq_len-1
        e = s + pet_len
        if e <= pet_start or s >= pet_end:
            valid.append(s)
    if not valid:
        return None, None
    chosen = int(rng.choice(valid))
    return chosen, chosen + pet_len


def metrics_for_span(attn_weights, seq_len, span_start, span_end, sink_indices):
    """Compute per-(layer, head) profile records treating [span_start, span_end) as the PET."""
    return profile_attention_span(
        attn_weights, seq_len, span_start, span_end, sink_indices,
    )


def profile_pass(model, test_df, rng, max_samples=None, real_only=False):
    """Iterate test_df, profile each sample up to three ways in a single forward pass.

    Real PET and masked PET are run for every sample where the PET is found.
    Random span is only run when a non-overlapping span of equal length exists;
    samples without a valid random span are included in real/mask but not rand.

    If real_only=True, only the real PET pass is run (used for the base model).

    Returns (real_df, rand_df, mask_df, length_df, n_pred_flipped, n_total) where
    n_total counts samples included in real/mask (PET found), and rand_df covers
    the subset that also had a valid random span. When real_only=True, rand_df and
    mask_df are empty DataFrames.
    """
    if max_samples:
        test_df = test_df.head(max_samples)
    mask_id = get_tokenizer().mask_token_id
    real_dfs, rand_dfs, mask_dfs, lengths = [], [], [], []
    n_pred_flipped = 0
    for _, row in test_df.iterrows():
        text = str(row[config.DATA_COLUMN])
        pet = str(row['pet_instance'])
        cond = 'Euphemistic' if str(row['label']) in ['1', '1.0', 'True', 'euph'] else 'Non-Euphemistic'
        tokenized = tokenize(text)
        tokens = get_tokens(tokenized)
        ps, pe = _find_unbounded_pet(tokens, pet)
        if ps is None:
            continue
        seq_len = len(tokens)
        sink_idx = [i for i, t in enumerate(tokens) if t in ('<s>', '</s>')]

        attn, predicted = get_model_output(model, tokenized)
        df_real = metrics_for_span(attn, seq_len, ps, pe, sink_idx)
        df_real['Condition'] = cond
        df_real['Sample'] = pet
        # The base checkpoint has a randomly initialized classification head,
        # so its predicted class is meaningless. Base-model profiles are used
        # only for attention comparisons; keep that field out of the output.
        if not real_only:
            df_real['Predicted'] = predicted
        real_dfs.append(df_real)
        lengths.append({'pet_len': pe - ps, 'seq_len': seq_len, 'Condition': cond,
                        'pred_real': predicted, 'pred_mask': None})

        if not real_only:
            masked = {'input_ids': tokenized['input_ids'].clone(),
                      'attention_mask': tokenized['attention_mask']}
            masked['input_ids'][0, ps:pe] = mask_id
            attn_mask, pred_mask = get_model_output(model, masked)
            df_mask = metrics_for_span(attn_mask, seq_len, ps, pe, sink_idx)
            df_mask['Condition'] = cond
            df_mask['Sample'] = pet
            df_mask['Predicted'] = pred_mask
            mask_dfs.append(df_mask)
            lengths[-1]['pred_mask'] = pred_mask
            if predicted != pred_mask:
                n_pred_flipped += 1

            rs, re_ = random_non_overlapping_span(seq_len, ps, pe, rng)
            if rs is not None:
                df_rand = metrics_for_span(attn, seq_len, rs, re_, sink_idx)
                df_rand['Condition'] = cond
                df_rand['Sample'] = pet
                df_rand['Predicted'] = predicted
                rand_dfs.append(df_rand)

    empty = pd.DataFrame(columns=real_dfs[0].columns if real_dfs else _METRICS)
    return (pd.concat(real_dfs, ignore_index=True),
            pd.concat(rand_dfs, ignore_index=True) if rand_dfs else empty,
            pd.concat(mask_dfs, ignore_index=True) if mask_dfs else empty,
            pd.DataFrame(lengths),
            n_pred_flipped,
            len(lengths))


def to_per_sample(df):
    df = df.copy().reset_index(drop=True)
    df['sample_id'] = df.index // (config.NUM_LAYERS * config.NUM_HEADS)
    group_columns = ['sample_id', 'Layer', 'Condition']
    if 'Predicted' in df.columns:
        group_columns.append('Predicted')
    return df.groupby(group_columns)[_METRICS].mean().reset_index()


def check_permutation(per_sample, n_perms, seed):
    logger.debug("\n" + "=" * 78)
    logger.debug(" Check 1 — Label-shuffle permutation control")
    logger.debug("=" * 78)
    rng = np.random.default_rng(seed)
    sc = per_sample.groupby('sample_id')['Condition'].first()
    sample_ids = sc.index.values
    conds = sc.values
    n_tests = len(_METRICS) * config.NUM_LAYERS
    thresh = 0.05 / n_tests

    logger.debug(f"  {n_perms} shuffles · {n_tests} tests each · Bonferroni p < {thresh:.2e}\n")
    n_sigs, n_l11_sigs = [], []
    for _ in range(n_perms):
        shuffled = rng.permutation(conds)
        cmap = dict(zip(sample_ids, shuffled))
        ps = per_sample.copy()
        ps['Condition'] = ps['sample_id'].map(cmap)
        ns, nl11 = 0, 0
        for m in _METRICS:
            for layer in range(1, config.NUM_LAYERS + 1):
                sub = ps[ps['Layer'] == layer]
                e = sub[sub['Condition'] == 'Euphemistic'][m].values
                n = sub[sub['Condition'] == 'Non-Euphemistic'][m].values
                _, p = scipy.stats.mannwhitneyu(e, n, alternative='two-sided')
                if p * n_tests < 0.05:
                    ns += 1
                    if layer == 11:
                        nl11 += 1
        n_sigs.append(ns)
        n_l11_sigs.append(nl11)
    logger.debug(f"  Bonferroni-significant tests per shuffle (out of {n_tests}):")
    logger.debug(f"    mean={np.mean(n_sigs):.2f}, max={max(n_sigs)}, all={n_sigs}")
    logger.debug(f"  Layer-11 significant per shuffle (out of {len(_METRICS)}):")
    logger.debug(f"    mean={np.mean(n_l11_sigs):.2f}, max={max(n_l11_sigs)}, all={n_l11_sigs}")
    l11_real = per_sample[per_sample['Layer'] == 11]
    n_real_sig = sum(
        scipy.stats.mannwhitneyu(
            l11_real[l11_real['Condition'] == 'Euphemistic'][m].values,
            l11_real[l11_real['Condition'] == 'Non-Euphemistic'][m].values,
            alternative='two-sided',
        )[1] * n_tests < 0.05
        for m in _METRICS
    )
    logger.debug(f"  Real (unshuffled): {n_real_sig}/{len(_METRICS)} layer-11 metrics significant.")


def check_length(per_sample, lengths):
    logger.debug("\n" + "=" * 78)
    logger.debug(" Check 2 — PET / sequence length confound at layer 11")
    logger.debug("=" * 78)
    lengths = lengths.reset_index(drop=True).copy()
    lengths['sample_id'] = lengths.index
    e = lengths[lengths['Condition'] == 'Euphemistic']
    n = lengths[lengths['Condition'] == 'Non-Euphemistic']
    logger.debug(f"  PET length: Euph={e['pet_len'].mean():.2f} ({e['pet_len'].std():.2f}), "
          f"Non={n['pet_len'].mean():.2f} ({n['pet_len'].std():.2f}), "
          f"d={cohens_d(e['pet_len'], n['pet_len']):+.3f}")
    logger.debug(f"  Seq length: Euph={e['seq_len'].mean():.2f} ({e['seq_len'].std():.2f}), "
          f"Non={n['seq_len'].mean():.2f} ({n['seq_len'].std():.2f}), "
          f"d={cohens_d(e['seq_len'], n['seq_len']):+.3f}")

    l11 = per_sample[per_sample['Layer'] == 11].merge(
        lengths[['sample_id', 'pet_len', 'seq_len']], on='sample_id')
    logger.debug(f"\n  Per-metric d at layer 11, before and after residualizing on (pet_len, seq_len):")
    logger.debug(f"  {'Metric':>26} {'d_raw':>9} {'d_resid':>9} {'Δ':>7}")
    for m in _METRICS:
        X = l11[['pet_len', 'seq_len']].values
        y = l11[m].values
        reg = sklearn.linear_model.LinearRegression().fit(X, y)
        resid = y - reg.predict(X)
        e_ = l11[l11['Condition'] == 'Euphemistic'].index
        n_ = l11[l11['Condition'] == 'Non-Euphemistic'].index
        d_raw = cohens_d(y[l11['Condition'].values == 'Euphemistic'],
                         y[l11['Condition'].values == 'Non-Euphemistic'])
        d_res = cohens_d(resid[l11['Condition'].values == 'Euphemistic'],
                         resid[l11['Condition'].values == 'Non-Euphemistic'])
        logger.debug(f"  {m:>26} {d_raw:>+9.3f} {d_res:>+9.3f} {(d_res - d_raw):>+7.3f}")


def check_random_span(real_ps, rand_ps):
    logger.debug("\n" + "=" * 78)
    logger.debug(" Check 3 — Random-span baseline at layer 11")
    logger.debug("=" * 78)
    n_tests = len(_METRICS) * config.NUM_LAYERS
    thresh = 0.05 / n_tests
    logger.debug(f"  Bonferroni threshold p < {thresh:.2e}\n")
    logger.debug(f"  {'Metric':>26} {'d_real':>9} {'d_rand':>9} {'p_rand (raw)':>14} {'p_rand (bonf)':>14} {'sig':>4}")
    for m in _METRICS:
        rl = real_ps[real_ps['Layer'] == 11]
        rd = rand_ps[rand_ps['Layer'] == 11]
        e_r = rl[rl['Condition'] == 'Euphemistic'][m].values
        n_r = rl[rl['Condition'] == 'Non-Euphemistic'][m].values
        e_d = rd[rd['Condition'] == 'Euphemistic'][m].values
        n_d = rd[rd['Condition'] == 'Non-Euphemistic'][m].values
        d_real = cohens_d(e_r, n_r)
        d_rand = cohens_d(e_d, n_d)
        _, p = scipy.stats.mannwhitneyu(e_d, n_d, alternative='two-sided')
        p_b = min(p * n_tests, 1.0)
        sig = '***' if p_b < 0.001 else '**' if p_b < 0.01 else '*' if p_b < 0.05 else ''
        logger.debug(f"  {m:>26} {d_real:>+9.3f} {d_rand:>+9.3f} {p:>14.2e} {p_b:>14.2e} {sig:>4}")


def check_misclassified(per_sample):
    logger.debug("\n" + "=" * 78)
    logger.debug(" Check 4 — Correct vs misclassified at layer 11")
    logger.debug("=" * 78)
    l11 = per_sample[per_sample['Layer'] == 11].copy()
    l11['correct'] = l11['Condition'] == l11['Predicted']
    n_cor = int(l11['correct'].sum())
    n_mis = int((~l11['correct']).sum())
    n_e_cor = int(((l11['Condition'] == 'Euphemistic') & l11['correct']).sum())
    n_n_cor = int(((l11['Condition'] == 'Non-Euphemistic') & l11['correct']).sum())
    n_e_mis = int(((l11['Condition'] == 'Euphemistic') & ~l11['correct']).sum())
    n_n_mis = int(((l11['Condition'] == 'Non-Euphemistic') & ~l11['correct']).sum())
    logger.debug(f"  Correct: {n_cor} (E={n_e_cor}, N={n_n_cor})  Misclassified: {n_mis} (E={n_e_mis}, N={n_n_mis})\n")

    logger.debug(f"  {'Metric':>26} {'d_correct':>11} {'d_misclass':>12} {'Δ':>7}")
    for m in _METRICS:
        c = l11[l11['correct']]
        i = l11[~l11['correct']]
        e_c = c[c['Condition'] == 'Euphemistic'][m].values
        n_c = c[c['Condition'] == 'Non-Euphemistic'][m].values
        e_i = i[i['Condition'] == 'Euphemistic'][m].values
        n_i = i[i['Condition'] == 'Non-Euphemistic'][m].values
        d_c = cohens_d(e_c, n_c)
        d_i = cohens_d(e_i, n_i) if len(e_i) > 1 and len(n_i) > 1 else float('nan')
        delta = (abs(d_i) - abs(d_c)) if not np.isnan(d_i) else float('nan')
        logger.debug(f"  {m:>26} {d_c:>+11.3f} {d_i:>+12.3f} {delta:>+7.3f}")


def check_pet_mask(real_ps, mask_ps, n_pred_flipped, n_total):
    logger.debug("\n" + "=" * 78)
    logger.debug(" Check 5 — PET-mask baseline at layer 11 (lexical PET-specificity)")
    logger.debug("=" * 78)
    flip_pct = 100.0 * n_pred_flipped / n_total if n_total else 0.0
    logger.debug(f"  Prediction flip rate after masking PET: {n_pred_flipped}/{n_total} = {flip_pct:.1f}%")

    n_tests = len(_METRICS) * config.NUM_LAYERS
    thresh = 0.05 / n_tests
    logger.debug(f"  Bonferroni threshold p < {thresh:.2e}\n")
    logger.debug(f"  {'Metric':>26} {'d_real':>9} {'d_mask':>9} {'p_mask (raw)':>14} {'p_mask (bonf)':>14} {'sig':>4}")
    for m in _METRICS:
        rl = real_ps[real_ps['Layer'] == 11]
        ml = mask_ps[mask_ps['Layer'] == 11]
        e_r = rl[rl['Condition'] == 'Euphemistic'][m].values
        n_r = rl[rl['Condition'] == 'Non-Euphemistic'][m].values
        e_m = ml[ml['Condition'] == 'Euphemistic'][m].values
        n_m = ml[ml['Condition'] == 'Non-Euphemistic'][m].values
        d_real = cohens_d(e_r, n_r)
        d_mask = cohens_d(e_m, n_m)
        _, p = scipy.stats.mannwhitneyu(e_m, n_m, alternative='two-sided')
        p_b = min(p * n_tests, 1.0)
        sig = '***' if p_b < 0.001 else '**' if p_b < 0.01 else '*' if p_b < 0.05 else ''
        logger.debug(f"  {m:>26} {d_real:>+9.3f} {d_mask:>+9.3f} {p:>14.2e} {p_b:>14.2e} {sig:>4}")


def check_diff_test(real_df, base_df):
    logger.debug("\n" + "=" * 78)
    logger.debug(" Check 6 — Finetuned − base difference test at layer 11")
    logger.debug("=" * 78)
    n_tests = len(_METRICS) * config.NUM_LAYERS
    thresh = 0.05 / n_tests
    logger.debug(f"  Bonferroni threshold p < {thresh:.2e}\n")

    diff_df = real_df.copy().reset_index(drop=True)
    base_reset = base_df.reset_index(drop=True)
    for m in _METRICS:
        diff_df[m] = real_df[m].values - base_reset[m].values
    diff_ps = to_per_sample(diff_df)

    logger.debug(f"  {'Metric':>26} {'d_ft':>9} {'d_base':>9} {'d_diff':>9} {'p_diff (raw)':>14} {'p_diff (bonf)':>14} {'sig':>4}")
    real_ps = to_per_sample(real_df)
    base_ps = to_per_sample(base_df)
    n_sig = 0
    for m in _METRICS:
        rl = real_ps[real_ps['Layer'] == 11]
        bl = base_ps[base_ps['Layer'] == 11]
        dl = diff_ps[diff_ps['Layer'] == 11]
        d_ft   = cohens_d(rl[rl['Condition'] == 'Euphemistic'][m].values,
                           rl[rl['Condition'] == 'Non-Euphemistic'][m].values)
        d_base = cohens_d(bl[bl['Condition'] == 'Euphemistic'][m].values,
                           bl[bl['Condition'] == 'Non-Euphemistic'][m].values)
        e_d = dl[dl['Condition'] == 'Euphemistic'][m].values
        n_d = dl[dl['Condition'] == 'Non-Euphemistic'][m].values
        d_diff = cohens_d(e_d, n_d)
        _, p = scipy.stats.mannwhitneyu(e_d, n_d, alternative='two-sided')
        p_b = min(p * n_tests, 1.0)
        sig = '***' if p_b < 0.001 else '**' if p_b < 0.01 else '*' if p_b < 0.05 else ''
        if p_b < 0.05:
            n_sig += 1
        logger.debug(f"  {m:>26} {d_ft:>+9.3f} {d_base:>+9.3f} {d_diff:>+9.3f} {p:>14.2e} {p_b:>14.2e} {sig:>4}")
    logger.debug(f"\n  {n_sig}/{len(_METRICS)} layer-11 metrics significant in the difference test.")


def main(args):
    logger.info("Loading finetuned model...")
    ft_model = load_model(config.MODEL_DIR)
    _, test_df = load_splits()
    logger.info(f"Test split: {len(test_df)} samples")

    rng = np.random.default_rng(args.seed)
    logger.info("Profiling (real PET + random span + masked PET in single pass)...")
    real_df, rand_df, mask_df, lengths, n_flipped, n_total = profile_pass(
        ft_model, test_df, rng, max_samples=args.max_samples)
    n_samples = len(real_df) // (config.NUM_LAYERS * config.NUM_HEADS)
    logger.info(f"  {n_samples} samples kept (after PET / random-span availability)")
    del ft_model
    torch.cuda.empty_cache()

    real_ps = to_per_sample(real_df)
    rand_ps = to_per_sample(rand_df)
    mask_ps = to_per_sample(mask_df)

    if args.max_samples is None:
        data_dir = os.path.join(config.PLOTS_DIR, config.MODEL_RUN_NAME, 'data')
        os.makedirs(data_dir, exist_ok=True)
        mask_df.to_parquet(os.path.join(data_dir, 'mask_df.parquet'), index=False)
        logger.info(f"  Cached mask_df to {data_dir}/mask_df.parquet")

    check_permutation(real_ps, args.n_perms, args.seed)
    check_length(real_ps, lengths)
    check_random_span(real_ps, rand_ps)
    check_misclassified(real_ps)
    check_pet_mask(real_ps, mask_ps, n_flipped, n_total)

    logger.info("\nLoading base model for difference test...")
    base_model = load_model(config.MODEL_NAME)
    logger.info("Profiling base model (real PET only)...")
    base_df, _, _, _, _, _ = profile_pass(
        base_model, test_df, rng, max_samples=args.max_samples, real_only=True)
    del base_model
    torch.cuda.empty_cache()
    check_diff_test(real_df, base_df)
    write_json(args.json_out, {
        'script': 'sanity_checks.py',
        'model_dir': config.MODEL_DIR,
        'split_dir': config.MODEL_RUN_DIR,
        'n_test_samples': len(test_df),
        'n_profiled_samples': int(n_samples),
        'n_prediction_flips_after_masking_pet': int(n_flipped),
        'seed': args.seed,
        'n_permutations': args.n_perms,
        'profiles': {'real_pet': real_df, 'random_span': rand_df,
                     'masked_pet': mask_df, 'base_real_pet': base_df},
        'lengths': lengths,
    })
    if args.json_out:
        logger.info(f"JSON profile data written to {args.json_out}")
    return {
        'real_ft': real_df,
        'random_ft': rand_df,
        'masked_ft': mask_df,
        'base_real': base_df,
        'lengths': lengths,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Five sanity checks for the layer-11 reconfiguration claim: permutation control, length confound, random-span baseline, misclassification stratification, and PET-mask lexical control."
    )
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--n-perms', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed per-metric statistical reports.')
    add_model_arg(parser)
    add_json_arg(parser)
    args = parser.parse_args()
    init_model(args)

    main(args)
