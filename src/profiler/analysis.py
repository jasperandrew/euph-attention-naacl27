"""
Single-sentence attention profiler.

Computes 5 unpaired metrics for each sentence independently, then
profile_dataframe aggregates across a DataFrame of examples.
"""
import numpy as np
import pandas as pd
import scipy.stats

from .. import config
from ..attention_utils import get_model_output, get_attn_head_weights, _find_unbounded_pet
from ..data_utils import tokenize, get_tokens


def profile_single_sample(model, text, pet_phrase, sink_tokens=('<s>', '</s>')):
    """Return (df, predicted_label) for one sentence.

    df has per-layer/head rows with metrics (all from the PET phrase's perspective):
      Inward (Context → PET)  : mean attention other tokens pay to PET
      Outward (PET → Context) : mean attention PET pays to other tokens
      Internal (PET → PET)    : mean self-attention within PET
      Entropy                 : mean outward entropy (dispersion of PET rows)
      Sink Fraction           : fraction of PET attention going to <s>/</s>
      Attn Distance           : expected token distance PET attends over
      Head Variance           : variance across heads (layer-level, duplicated per row)

    predicted_label is 'Euphemistic' or 'Non-Euphemistic'.
    Returns (None, None) if the PET phrase is not found.
    """
    tokenized = tokenize(text)
    tokens = get_tokens(tokenized)
    attn_weights, predicted_label = get_model_output(model, tokenized)

    pet_start, pet_end = _find_unbounded_pet(tokens, pet_phrase)
    if pet_start is None:
        print(f"PET phrase '{pet_phrase}' not found in '{text}'")
        return None, None

    sink_indices = [i for i, t in enumerate(tokens) if t in sink_tokens]
    seq_len = len(tokens)
    non_pet_indices = [i for i in range(seq_len) if (i < pet_start or i >= pet_end) and i not in sink_indices]
    pet_indices = list(range(pet_start, pet_end))

    positions = np.arange(seq_len)

    records = []
    for layer_i in range(len(attn_weights)):
        num_heads = attn_weights[layer_i].shape[1]
        layer_pet_dists = []

        for head_i in range(num_heads):
            head_weights = get_attn_head_weights(attn_weights, layer_i, head_i, seq_len)
            pet_to_all = head_weights[pet_start:pet_end, :]
            layer_pet_dists.append(pet_to_all.flatten())

            inward = head_weights[np.ix_(non_pet_indices, pet_indices)].sum(axis=1).mean() if non_pet_indices else 0
            outward = head_weights[np.ix_(pet_indices, non_pet_indices)].sum(axis=1).mean() if non_pet_indices else 0
            internal = head_weights[np.ix_(pet_indices, pet_indices)].sum(axis=1).mean()

            clamped = np.clip(pet_to_all, 1e-9, 1.0)
            entropy = np.mean([scipy.stats.entropy(clamped[i]) for i in range(pet_end - pet_start)])

            sink = head_weights[np.ix_(pet_indices, sink_indices)].sum(axis=1).mean() if sink_indices else 0

            pet_pos = np.array(pet_indices)[:, np.newaxis]      # (pet_len, 1)
            dist_matrix = np.abs(pet_pos - positions)           # (pet_len, seq_len)
            distance = (dist_matrix * pet_to_all).sum(axis=1).mean()

            records.append({
                'Layer': layer_i + 1,
                'Head': head_i + 1,
                'Inward (Context → PET)': inward,
                'Outward (PET → Context)': outward,
                'Internal (PET → PET)': internal,
                'Entropy': entropy,
                'Sink Fraction': sink,
                'Attn Distance': distance,
            })

        head_variance = np.var(np.array(layer_pet_dists), axis=0).mean()
        for r in records[-num_heads:]:
            r['Head Variance'] = head_variance

    return pd.DataFrame(records), predicted_label


def profile_dataframe(model, df, text_col, pet_col, label_col, max_samples=None):
    """Run profile_single_sample over every row of df, tagging each with Condition and Sample."""
    if max_samples is not None:
        df = df.head(max_samples)

    if pet_col not in df.columns:
        raise ValueError(f"Column '{pet_col}' not found. Available: {list(df.columns)}")

    all_dfs = []
    for _, row in df.iterrows():
        text = str(row[text_col])
        pet = str(row[pet_col])
        condition = 'Euphemistic' if str(row[label_col]) in ['1', '1.0', 'True', 'euph'] else 'Non-Euphemistic'

        df_profile, predicted = profile_single_sample(model, text, pet)
        if df_profile is not None:
            df_profile['Condition'] = condition
            df_profile['Predicted'] = predicted
            df_profile['Sample'] = pet
            all_dfs.append(df_profile)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None
