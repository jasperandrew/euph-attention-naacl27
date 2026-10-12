"""Shared per-span attention profile calculations."""

import numpy as np
import pandas as pd
import scipy.stats

from ..attention_utils import get_attn_head_weights


METRIC_COLUMNS = [
    'Inward (Context → PET)', 'Outward (PET → Context)',
    'Internal (PET → PET)', 'Entropy', 'Sink (PET → Sink)',
    'Sink (Sink → PET)', 'Distance',
]


def profile_attention_span(attn_weights, seq_len, span_start, span_end,
                           sink_indices, include_head_variance=True):
    """Calculate per-layer/head metrics for a span treated as the PET.

    The span uses half-open, sample-wide token indices. All other non-sink
    positions are context. Attention matrices are trimmed to ``seq_len`` so
    padded positions do not enter any measure.
    """
    if not 0 <= span_start < span_end <= seq_len:
        raise ValueError(
            f'Invalid span [{span_start}, {span_end}) for sequence length {seq_len}'
        )

    span_indices = list(range(span_start, span_end))
    sink_indices = list(sink_indices)
    context_indices = [
        i for i in range(seq_len)
        if i not in span_indices and i not in sink_indices
    ]
    positions = np.arange(seq_len)
    records = []

    for layer_i, layer_weights in enumerate(attn_weights):
        num_heads = layer_weights.shape[1]
        layer_span_attention = []
        layer_records = []

        for head_i in range(num_heads):
            weights = get_attn_head_weights(attn_weights, layer_i, head_i, seq_len)
            span_to_all = weights[span_start:span_end, :]
            layer_span_attention.append(span_to_all.flatten())

            inward = (weights[np.ix_(context_indices, span_indices)].sum(axis=1).mean()
                      if context_indices else 0)
            outward = (weights[np.ix_(span_indices, context_indices)].sum(axis=1).mean()
                       if context_indices else 0)
            internal = weights[np.ix_(span_indices, span_indices)].sum(axis=1).mean()

            clipped = np.clip(span_to_all, 1e-9, 1.0)
            entropy = np.mean([
                scipy.stats.entropy(clipped[i]) for i in range(span_end - span_start)
            ])
            sink = (weights[np.ix_(span_indices, sink_indices)].sum(axis=1).mean()
                    if sink_indices else 0)
            sink_to_pet = (weights[np.ix_(sink_indices, span_indices)].sum(axis=1).mean()
                           if sink_indices else 0)
            span_positions = np.array(span_indices)[:, np.newaxis]
            distance = (
                np.abs(span_positions - positions) * span_to_all
            ).sum(axis=1).mean()

            record = {
                'Layer': layer_i + 1,
                'Head': head_i + 1,
                'Inward (Context → PET)': inward,
                'Outward (PET → Context)': outward,
                'Internal (PET → PET)': internal,
                'Entropy': entropy,
                'Sink (PET → Sink)': sink,
                'Sink (Sink → PET)': sink_to_pet,
                'Distance': distance,
            }
            records.append(record)
            layer_records.append(record)

        if include_head_variance:
            head_variance = np.var(np.asarray(layer_span_attention), axis=0).mean()
            for record in layer_records:
                record['Head Variance'] = head_variance

    return pd.DataFrame(records)
