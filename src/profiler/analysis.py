"""
Single-sentence attention profiler.

Computes eight attention metrics for each sentence independently, then
profile_dataframe aggregates across a DataFrame of examples.
"""
import logging
import pandas as pd

from .. import config
from ..attention_utils import get_model_output, _find_unbounded_pet
from ..data_utils import tokenize, get_tokens
from ..stats.profiling import profile_attention_span

logger = logging.getLogger(__name__)


def profile_single_sample(model, text, pet_phrase, sink_tokens=('<s>', '</s>')):
    """Return (df, predicted_label) for one sentence.

    df has per-layer/head rows with metrics (all from the PET phrase's perspective):
      Inward (Context → PET)  : mean attention other tokens pay to PET
      Outward (PET → Context) : mean attention PET pays to other tokens
      Internal (PET → PET)    : mean self-attention within PET
      Entropy                 : mean outward entropy (dispersion of PET rows)
      Sink (PET → Sink)       : fraction of PET attention going to <s>/</s>
      Sink (Sink → PET)       : mean sink-token attention directed to PET tokens
      All (All → PET)         : mean attention from all non-padding queries to PET tokens
      Distance           : expected token distance PET attends over
      Head Variance           : variance across heads (layer-level, duplicated per row)

    predicted_label is 'Euphemistic' or 'Non-Euphemistic'.
    Returns (None, None) if the PET phrase is not found.
    """
    tokenized = tokenize(text)
    tokens = get_tokens(tokenized)
    attn_weights, predicted_label = get_model_output(model, tokenized)

    pet_start, pet_end = _find_unbounded_pet(tokens, pet_phrase)
    if pet_start is None:
        logger.warning("PET phrase '%s' not found in '%s'", pet_phrase, text)
        return None, None

    seq_len = len(tokens)
    sink_indices = [i for i, token in enumerate(tokens) if token in sink_tokens]
    return profile_attention_span(
        attn_weights, seq_len, pet_start, pet_end, sink_indices,
    ), predicted_label


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
