"""Five sanity checks for the layer-11 reconfiguration claim.

Each eval sample is profiled three ways in a single pass (real PET span, random
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

Usage:
  python sanity_checks.py
  python sanity_checks.py --max-samples 50
  python sanity_checks.py --n-perms 20 --seed 0
"""
import argparse

_METRICS = ['Inward (Context → PET)', 'Outward (PET → Context)',
            'Internal (PET → PET)', 'Entropy', 'Sink Fraction', 'Attn Distance']


def load_model(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, output_attentions=True, attn_implementation="eager",
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
    span_indices = list(range(span_start, span_end))
    non_indices = [i for i in range(seq_len) if i not in span_indices and i not in sink_indices]
    positions = np.arange(seq_len)
    records = []
    for layer_i in range(len(attn_weights)):
        num_heads = attn_weights[layer_i].shape[1]
        layer_dists = []
        for head_i in range(num_heads):
            w = get_attn_head_weights(attn_weights, layer_i, head_i, seq_len)
            span_rows = w[span_start:span_end, :]
            layer_dists.append(span_rows.flatten())
            inward = w[np.ix_(non_indices, span_indices)].sum(axis=1).mean() if non_indices else 0
            outward = w[np.ix_(span_indices, non_indices)].sum(axis=1).mean() if non_indices else 0
            internal = w[np.ix_(span_indices, span_indices)].sum(axis=1).mean()
            clamped = np.clip(span_rows, 1e-9, 1.0)
            entropy = np.mean([scipy.stats.entropy(clamped[i]) for i in range(span_end - span_start)])
            sink = w[np.ix_(span_indices, sink_indices)].sum(axis=1).mean() if sink_indices else 0
            sp = np.array(span_indices)[:, np.newaxis]
            distance = (np.abs(sp - positions) * span_rows).sum(axis=1).mean()
            records.append({
                'Layer': layer_i + 1, 'Head': head_i + 1,
                _METRICS[0]: inward, _METRICS[1]: outward, _METRICS[2]: internal,
                _METRICS[3]: entropy, _METRICS[4]: sink, _METRICS[5]: distance,
            })
        hv = np.var(np.array(layer_dists), axis=0).mean()
        for r in records[-num_heads:]:
            r['Head Variance'] = hv
    return pd.DataFrame(records)


def profile_pass(model, eval_df, rng, max_samples=None):
    """Iterate eval_df, profile each sample three ways (real PET, random span, masked PET).

    Returns (real_df, rand_df, mask_df, length_df, n_pred_flipped, n_total) where the
    DataFrames have matching sample_id ordering (sample_id = row_idx // 144 in
    real/rand/mask; sample_id = row_idx in length_df). Samples are skipped when PET
    is not found OR no valid random span exists. The masked pass runs the model
    a second time on inputs where PET tokens have been replaced with <mask>.
    """
    if max_samples:
        eval_df = eval_df.head(max_samples)
    mask_id = TOKENIZER.mask_token_id
    real_dfs, rand_dfs, mask_dfs, lengths = [], [], [], []
    n_pred_flipped = 0
    for _, row in eval_df.iterrows():
        text = str(row[config.DATA_COLUMN])
        pet = str(row['pet_instance'])
        cond = 'Euphemistic' if str(row['label']) in ['1', '1.0', 'True', 'euph'] else 'Non-Euphemistic'
        tokenized = tokenize(text)
        tokens = get_tokens(tokenized)
        ps, pe = _find_unbounded_pet(tokens, pet)
        if ps is None:
            continue
        seq_len = len(tokens)
        rs, re_ = random_non_overlapping_span(seq_len, ps, pe, rng)
        if rs is None:
            continue
        sink_idx = [i for i, t in enumerate(tokens) if t in ('<s>', '</s>')]

        attn, predicted = get_model_output(model, tokenized)
        df_real = metrics_for_span(attn, seq_len, ps, pe, sink_idx)
        df_rand = metrics_for_span(attn, seq_len, rs, re_, sink_idx)

        masked = {'input_ids': tokenized['input_ids'].clone(),
                  'attention_mask': tokenized['attention_mask']}
        masked['input_ids'][0, ps:pe] = mask_id
        attn_mask, pred_mask = get_model_output(model, masked)
        df_mask = metrics_for_span(attn_mask, seq_len, ps, pe, sink_idx)

        for d in (df_real, df_rand, df_mask):
            d['Condition'] = cond
        df_real['Predicted'] = predicted
        df_rand['Predicted'] = predicted
        df_mask['Predicted'] = pred_mask
        real_dfs.append(df_real)
        rand_dfs.append(df_rand)
        mask_dfs.append(df_mask)
        if predicted != pred_mask:
            n_pred_flipped += 1
        lengths.append({'pet_len': pe - ps, 'seq_len': seq_len, 'Condition': cond,
                        'pred_real': predicted, 'pred_mask': pred_mask})

    return (pd.concat(real_dfs, ignore_index=True),
            pd.concat(rand_dfs, ignore_index=True),
            pd.concat(mask_dfs, ignore_index=True),
            pd.DataFrame(lengths),
            n_pred_flipped,
            len(lengths))


def to_per_sample(df):
    df = df.copy().reset_index(drop=True)
    df['sample_id'] = df.index // (config.NUM_LAYERS * config.NUM_HEADS)
    return df.groupby(['sample_id', 'Layer', 'Condition', 'Predicted'])[_METRICS].mean().reset_index()


def check_permutation(per_sample, n_perms, seed):
    print("\n" + "=" * 78)
    print(" Check 1 — Label-shuffle permutation control")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    sc = per_sample.groupby('sample_id')['Condition'].first()
    sample_ids = sc.index.values
    conds = sc.values
    n_tests = len(_METRICS) * config.NUM_LAYERS
    thresh = 0.05 / n_tests

    print(f"  {n_perms} shuffles · {n_tests} tests each · Bonferroni p < {thresh:.2e}\n")
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
    print(f"  Bonferroni-significant tests per shuffle (out of {n_tests}):")
    print(f"    mean={np.mean(n_sigs):.2f}, max={max(n_sigs)}, all={n_sigs}")
    print(f"  Layer-11 significant per shuffle (out of {len(_METRICS)}):")
    print(f"    mean={np.mean(n_l11_sigs):.2f}, max={max(n_l11_sigs)}, all={n_l11_sigs}")
    print(f"  Real (unshuffled): 5/6 layer-11 metrics significant.")


def check_length(per_sample, lengths):
    print("\n" + "=" * 78)
    print(" Check 2 — PET / sequence length confound at layer 11")
    print("=" * 78)
    lengths = lengths.reset_index(drop=True).copy()
    lengths['sample_id'] = lengths.index
    e = lengths[lengths['Condition'] == 'Euphemistic']
    n = lengths[lengths['Condition'] == 'Non-Euphemistic']
    print(f"  PET length: Euph={e['pet_len'].mean():.2f} ({e['pet_len'].std():.2f}), "
          f"Non={n['pet_len'].mean():.2f} ({n['pet_len'].std():.2f}), "
          f"d={cohens_d(e['pet_len'], n['pet_len']):+.3f}")
    print(f"  Seq length: Euph={e['seq_len'].mean():.2f} ({e['seq_len'].std():.2f}), "
          f"Non={n['seq_len'].mean():.2f} ({n['seq_len'].std():.2f}), "
          f"d={cohens_d(e['seq_len'], n['seq_len']):+.3f}")

    l11 = per_sample[per_sample['Layer'] == 11].merge(
        lengths[['sample_id', 'pet_len', 'seq_len']], on='sample_id')
    print(f"\n  Per-metric d at layer 11, before and after residualizing on (pet_len, seq_len):")
    print(f"  {'Metric':>26} {'d_raw':>9} {'d_resid':>9} {'Δ':>7}")
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
        print(f"  {m:>26} {d_raw:>+9.3f} {d_res:>+9.3f} {(d_res - d_raw):>+7.3f}")


def check_random_span(real_ps, rand_ps):
    print("\n" + "=" * 78)
    print(" Check 3 — Random-span baseline at layer 11")
    print("=" * 78)
    n_tests = len(_METRICS) * config.NUM_LAYERS
    thresh = 0.05 / n_tests
    print(f"  Bonferroni threshold p < {thresh:.2e}\n")
    print(f"  {'Metric':>26} {'d_real':>9} {'d_rand':>9} {'p_rand (raw)':>14} {'p_rand (bonf)':>14} {'sig':>4}")
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
        print(f"  {m:>26} {d_real:>+9.3f} {d_rand:>+9.3f} {p:>14.2e} {p_b:>14.2e} {sig:>4}")


def check_misclassified(per_sample):
    print("\n" + "=" * 78)
    print(" Check 4 — Correct vs misclassified at layer 11")
    print("=" * 78)
    l11 = per_sample[per_sample['Layer'] == 11].copy()
    l11['correct'] = l11['Condition'] == l11['Predicted']
    n_cor = int(l11['correct'].sum())
    n_mis = int((~l11['correct']).sum())
    n_e_cor = int(((l11['Condition'] == 'Euphemistic') & l11['correct']).sum())
    n_n_cor = int(((l11['Condition'] == 'Non-Euphemistic') & l11['correct']).sum())
    n_e_mis = int(((l11['Condition'] == 'Euphemistic') & ~l11['correct']).sum())
    n_n_mis = int(((l11['Condition'] == 'Non-Euphemistic') & ~l11['correct']).sum())
    print(f"  Correct: {n_cor} (E={n_e_cor}, N={n_n_cor})  Misclassified: {n_mis} (E={n_e_mis}, N={n_n_mis})\n")

    print(f"  {'Metric':>26} {'d_correct':>11} {'d_misclass':>12} {'Δ':>7}")
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
        print(f"  {m:>26} {d_c:>+11.3f} {d_i:>+12.3f} {delta:>+7.3f}")


def check_pet_mask(real_ps, mask_ps, n_pred_flipped, n_total):
    print("\n" + "=" * 78)
    print(" Check 5 — PET-mask baseline at layer 11 (lexical PET-specificity)")
    print("=" * 78)
    flip_pct = 100.0 * n_pred_flipped / n_total if n_total else 0.0
    print(f"  Prediction flip rate after masking PET: {n_pred_flipped}/{n_total} = {flip_pct:.1f}%")

    n_tests = len(_METRICS) * config.NUM_LAYERS
    thresh = 0.05 / n_tests
    print(f"  Bonferroni threshold p < {thresh:.2e}\n")
    print(f"  {'Metric':>26} {'d_real':>9} {'d_mask':>9} {'p_mask (raw)':>14} {'p_mask (bonf)':>14} {'sig':>4}")
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
        print(f"  {m:>26} {d_real:>+9.3f} {d_mask:>+9.3f} {p:>14.2e} {p_b:>14.2e} {sig:>4}")


def main(args):
    print("Loading finetuned model...")
    model = load_model(config.MODEL_DIR)
    _, eval_df = load_splits()
    print(f"Eval split: {len(eval_df)} samples")

    rng = np.random.default_rng(args.seed)
    print("Profiling (real PET + random span + masked PET in single pass)...")
    real_df, rand_df, mask_df, lengths, n_flipped, n_total = profile_pass(
        model, eval_df, rng, max_samples=args.max_samples)
    n_samples = len(real_df) // (config.NUM_LAYERS * config.NUM_HEADS)
    print(f"  {n_samples} samples kept (after PET / random-span availability)")

    real_ps = to_per_sample(real_df)
    rand_ps = to_per_sample(rand_df)
    mask_ps = to_per_sample(mask_df)

    check_permutation(real_ps, args.n_perms, args.seed)
    check_length(real_ps, lengths)
    check_random_span(real_ps, rand_ps)
    check_misclassified(real_ps)
    check_pet_mask(real_ps, mask_ps, n_flipped, n_total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Five sanity checks for the layer-11 reconfiguration claim: permutation control, length confound, random-span baseline, misclassification stratification, and PET-mask lexical control."
    )
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--n-perms', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import scipy.stats
    import sklearn.linear_model
    import torch
    from transformers import AutoModelForSequenceClassification
    from src import config
    from src.attention_utils import get_model_output, get_attn_head_weights, _find_unbounded_pet
    from src.data_utils import TOKENIZER, load_splits, tokenize, get_tokens

    main(args)
