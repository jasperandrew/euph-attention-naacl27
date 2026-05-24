"""
Analysis functions that operate on euph/non-euph sentence pairs.

Each function here requires two sentences — one euphemistic, one not — and
computes some difference metric between them.
"""
from ..attention_utils import get_attn_weights, pet_sink_attention
from ..data_utils import tokenize, get_tokens


def compute_sink_diff_for_sample(model, sample):
    """Return (layer×head diff DataFrame, scalar mean diff) for one sample pair.

    sample keys: 'euph_n', 'euph_y', 'pet_phrase'
    Returns None if the PET phrase is not found in either sentence.
    """
    pet_phrase = sample['pet_phrase']

    tok_n = tokenize(sample['euph_n'])
    tok_y = tokenize(sample['euph_y'])
    tokens_n = get_tokens(tok_n)
    tokens_y = get_tokens(tok_y)

    df_n = pet_sink_attention(get_attn_weights(model, tok_n), tokens_n, pet_phrase)
    df_y = pet_sink_attention(get_attn_weights(model, tok_y), tokens_y, pet_phrase)

    if df_n is None or df_y is None:
        print(f"  Skipping '{pet_phrase}' — PET phrase not found.")
        return None

    diff = df_y.copy()
    diff['Sink (PET → Sink)'] = df_y['Sink (PET → Sink)'].values - df_n['Sink (PET → Sink)'].values
    return diff, diff['Sink (PET → Sink)'].mean()
