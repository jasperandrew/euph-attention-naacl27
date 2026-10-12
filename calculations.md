## Terminology
- PET token: part of the PET
- Sink token: `<s>`,`</s>`
- Context token: everything except PET and sink tokens

## Profile Measures (per layer x head x sample)

### Definitions
- X: an arbitrary set of tokens from the current sample
    - Defined sets: A (all tokens), P (PET tokens), S (sink tokens), and C (context tokens)
    - Attention matrices are trimmed to the sample's unpadded length before calculations
- I_X ⊆ {1,…,n}: the sample-wide indices of the tokens belonging to X
- a_ij: the attention weight from query token i to key token j
- H(X): Shannon entropy over X, in nats
    - The values of X are first clipped to [10^-9, 1] to avoid log(0) issues
    - `scipy.stats.entropy` defaults to base e, and normalizes its inputs to sum to 1

### Attention Flow (Inward, Outward, Internal, Sink)

Given query token set Q and key token set K, for i in I_Q, sum a_ij for all j in I_K, then take the mean over I_Q.

"The average attention flow from query tokens Q to key tokens K."

### Sink → PET Attention

For each sink-token query index `i ∈ I_S`, sum `a_ij` over PET key indices `j ∈ I_P`, then take the mean over sink-token queries:

`(1 / |I_S|) Σ_{i ∈ I_S} Σ_{j ∈ I_P} a_ij`

"The mean attention mass that the sample's `<s>` and `</s>` tokens direct to the PET span."

### All → PET Attention

For each non-padding query index `i ∈ I_A`, sum `a_ij` over PET key indices `j ∈ I_P`, then take the mean over all query tokens:

`(1 / |I_A|) Σ_{i ∈ I_A} Σ_{j ∈ I_P} a_ij`

"The mean attention mass directed to the PET span across all query tokens."

Because `I_A` includes context, PET, and sink queries, this measure is the token-count-weighted mean of Context → PET, PET → PET, and Sink → PET. The reverse measure, PET → All, is 1 by attention-row normalization (up to floating-point error), so it serves as a sanity check rather than a discriminating metric.

### Entropy

For i in I_P, define the attention-weight vector W_i = (a_ij)_{j ∈ I_A}, calculate its normalized entropy H(W_i), then take the mean over I_P.

"The average dispersal of attention flowing from PET tokens."

### Distance

For i in I_P, sum a_ij*|i-j| for all j in I_A, then take the average over I_P.

"The average expected distance of attention flowing from PET tokens."
