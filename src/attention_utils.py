import numpy as np
import torch
import pandas as pd
import scipy.stats

from . import config


# ---------------------------------------------------------------------------
# Attention weight extraction
# ---------------------------------------------------------------------------

def get_attn_weights(model, tokenized_input):
    input_ids = tokenized_input['input_ids'].to(model.device)
    attn_mask = tokenized_input['attention_mask'].to(model.device)
    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attn_mask)
    return output.attentions


def get_model_output(model, tokenized_input):
    """Run one forward pass; return (attentions, predicted_label_str).

    predicted_label_str is 'Euphemistic' (class 1) or 'Non-Euphemistic' (class 0).
    """
    input_ids = tokenized_input['input_ids'].to(model.device)
    attn_mask = tokenized_input['attention_mask'].to(model.device)
    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attn_mask)
    pred_id = int(torch.argmax(output.logits, dim=-1).item())
    pred_label = 'Euphemistic' if pred_id == 1 else 'Non-Euphemistic'
    return output.attentions, pred_label


def get_attn_head_weights(attn_weights, layer_idx, head_idx, seq_len):
    return attn_weights[layer_idx][0, head_idx, :seq_len, :seq_len].cpu().numpy()


# ---------------------------------------------------------------------------
# PET region finders
# ---------------------------------------------------------------------------

def _find_unbounded_pet(tokens, pet_phrase):
    """Return (start, end) token indices for pet_phrase via substring match.

    pet_phrase may be a list of token strings (e.g. ['▁passed', '▁on']) or a
    plain string (e.g. 'passed on').  Matching is case-insensitive and ignores
    the ▁ delimiter and whitespace.
    """
    if isinstance(pet_phrase, list):
        target_str = "".join(pet_phrase).replace(config.TOK_DELIM, "").replace(" ", "").lower()
        display_name = " ".join(pet_phrase)
    else:
        target_str = pet_phrase.replace(config.TOK_DELIM, "").replace(" ", "").lower()
        display_name = pet_phrase

    norm_tokens = [t.replace(config.TOK_DELIM, "").lower() for t in tokens]
    target_len = len(target_str)

    for i in range(len(tokens)):
        acc = ""
        for j in range(i, len(tokens)):
            acc += norm_tokens[j]
            if acc == target_str:
                return i, j + 1
            if len(acc) >= target_len:
                break

    print(f"Warning: PET sequence '{display_name}' not found in the input sequence.")
    return None, None


def _find_bounded_pet(tokens, include_bounds=False):
    """Return (start, end) indices for the region between BOUND_TOKENs.

    Only usable when config.BOUNDED is True.
    """
    try:
        pet_start = tokens.index(config.BOUND_TOKEN) + 1
        pet_end = tokens.index(config.BOUND_TOKEN, pet_start)
    except ValueError:
        print(f"Warning: {config.BOUND_TOKEN} tokens not found twice in the input sequence.")
        return None, None

    if include_bounds:
        pet_start -= 1
        pet_end += 1

    if pet_start >= pet_end:
        print(f"Warning: No tokens found strictly between the {config.BOUND_TOKEN} tokens.")
        return None, None

    return pet_start, pet_end


# ---------------------------------------------------------------------------
# Attention region extractors
# ---------------------------------------------------------------------------

def get_pet_attention(attn_head_weights, tokens, pet_indices):
    pet_start, pet_end = pet_indices
    if pet_start is None or pet_end is None:
        return None, None
    return attn_head_weights[pet_start:pet_end, pet_start:pet_end], tokens[pet_start:pet_end]


def get_non_pet_attention(attn_head_weights, tokens, pet_indices):
    pet_start, pet_end = pet_indices
    if pet_start is None or pet_end is None:
        return None, None

    non_pet_weights = []
    non_pet_tokens = []

    if pet_start > 0:
        non_pet_weights.append(attn_head_weights[0:pet_start, 0:pet_start])
        non_pet_tokens.append(tokens[0:pet_start])

    if pet_end < len(tokens):
        non_pet_weights.append(attn_head_weights[pet_end:len(tokens), pet_end:len(tokens)])
        non_pet_tokens.append(tokens[pet_end:len(tokens)])

    if not non_pet_weights:
        print("Warning: No tokens found outside the defined PET region.")
        return None, None

    return non_pet_weights, non_pet_tokens


def get_bound_attention(attn_head_weights, tokens):
    """Extract attention between the two BOUND_TOKEN markers. Requires bounded mode."""
    try:
        b1 = tokens.index(config.BOUND_TOKEN)
        b2 = tokens.index(config.BOUND_TOKEN, b1 + 1)
    except ValueError:
        print(f"Warning: {config.BOUND_TOKEN} tokens not found twice in the input sequence.")
        return None, None
    bound_indices = [b1, b2]
    return attn_head_weights[np.ix_(bound_indices, bound_indices)], [config.BOUND_TOKEN, config.BOUND_TOKEN]


def get_collapsed_attention(attn_head_weights, tokens, pet_phrase, aggregation_method='mean'):
    pet_start, pet_end = _find_unbounded_pet(tokens, pet_phrase)
    if pet_start is None:
        return None, None

    actual_pet_tokens = tokens[pet_start:pet_end]
    pre_pet_indices = list(range(0, pet_start))
    pet_indices = list(range(pet_start, pet_end))
    post_pet_indices = list(range(pet_end, len(tokens)))

    collapsed_labels = ['(pre_pet)'] + [t.replace(config.TOK_DELIM, '') for t in actual_pet_tokens] + ['(post_pet)']
    n = len(collapsed_labels)
    collapsed_attn = np.zeros((n, n))

    if aggregation_method == 'mean':
        agg_func = np.mean
    elif aggregation_method == 'sum':
        agg_func = np.sum
    else:
        raise ValueError(f"aggregation_method must be 'mean' or 'sum', got '{aggregation_method}'")

    region_map = {0: pre_pet_indices, n - 1: post_pet_indices}
    for k, pet_idx in enumerate(pet_indices):
        region_map[k + 1] = [pet_idx]

    for i in range(n):
        for j in range(n):
            q_idx = region_map.get(i, [])
            k_idx = region_map.get(j, [])
            if not q_idx or not k_idx:
                continue
            sub = attn_head_weights[np.ix_(q_idx, k_idx)]
            collapsed_attn[i, j] = agg_func(sub) if sub.size > 0 else 0.0

    return collapsed_attn, collapsed_labels


# ---------------------------------------------------------------------------
# Entropy analysis
# ---------------------------------------------------------------------------

def get_attention_entropy(attn_weights, tokens, pet_tokens, model_label="Model"):
    """Compute inward entropy: how broadly is each PET token attended to by the sequence?"""
    pet_start, pet_end = _find_unbounded_pet(tokens, pet_tokens)
    if pet_start is None:
        return None

    seq_len = len(tokens)
    records = []

    for layer_i, layer_attn in enumerate(attn_weights):
        num_heads = layer_attn.shape[1]
        for head_i in range(num_heads):
            head_weights = get_attn_head_weights(attn_weights, layer_i, head_i, seq_len)
            for pet_tok_i in range(pet_start, pet_end):
                col = np.clip(head_weights[:, pet_tok_i], 1e-9, 1.0)
                records.append({
                    'Layer': layer_i + 1,
                    'Head': head_i + 1,
                    'PET Token': tokens[pet_tok_i].replace(config.TOK_DELIM, ''),
                    'Entropy': scipy.stats.entropy(col),
                    'Condition': model_label,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Sink attention analysis
# ---------------------------------------------------------------------------

def pet_sink_attention(attn_weights, tokens, pet_tokens, sink_tokens=('<s>', '</s>')):
    pet_start, pet_end = _find_unbounded_pet(tokens, pet_tokens)
    if pet_start is None:
        return None

    sink_indices = [i for i, t in enumerate(tokens) if t in sink_tokens]
    seq_len = len(tokens)
    records = []

    for layer_i in range(len(attn_weights)):
        for head_i in range(attn_weights[layer_i].shape[1]):
            head_weights = get_attn_head_weights(attn_weights, layer_i, head_i, seq_len)
            pet_rows = head_weights[pet_start:pet_end]
            sink_attn = pet_rows[:, sink_indices].sum(axis=1).mean()
            records.append({
                'Layer': layer_i + 1,
                'Head': head_i + 1,
                'Sink (PET → Sink)': float(sink_attn),
            })

    return pd.DataFrame(records)
