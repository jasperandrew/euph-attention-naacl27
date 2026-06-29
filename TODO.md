# TODO

## 1. `FROZEN_LAYERS` is global but should be per-model

`src/config.py` still has `FROZEN_LAYERS = 7`, a leftover from the original
frozen-7-layer EN model. The current models (EN unfrozen, SP) were trained with
all 12 layers trainable, so this value is wrong for them.

Where it leaks:
- `significance.py` Section 1/2/4/5 tag layers 1–7 with `[frozen]`, and the
  "Frozen vs unfrozen mean |Δ|" table header — for unfrozen models these are
  mislabels (they just mean "layers 1–7"). Existing `notes/outputs/*.out` for
  the unfrozen EN model carry these stale tags; don't quote them into the paper.
- `plot_profile.py --layers unfrozen` and `plot_pairs.py --single-head-plots`
  ("all non-frozen layers") select layers via `FROZEN_LAYERS`.

Fix: make it per-language (e.g. `FROZEN_LAYERS = {'en': 0, 'sp': 0, 'zh': 0}`
resolved via `config.LANG`), confirming the actual training regime of each
model first. If the frozen-7 EN model ever comes back as a comparison point it
would need its own entry (or model dir + metadata).

## 2. Verify profiler PET-token localization on non-EN tokenization

The PET-locating logic (`_find_unbounded_pet` in `src/attention_utils.py`,
used by the profiler) and `TOK_DELIM = '▁'` / `EXAMPLE_PET_TOKENS` in
`src/config.py` were built against XLM-R SentencePiece output for
space-delimited English.

- **SP**: probably works (space-delimited, same `▁` conventions), but spot-check
  a handful of samples — accents and the capitalized PETs ("Daños colaterales")
  may tokenize differently than their in-sentence occurrences.
- **ZH**: SentencePiece emits most Han tokens *without* the `▁` prefix (it only
  marks tokens that follow whitespace), so any matching that assumes
  `▁`-prefixed word starts will mislocate or miss PET spans. Must be verified
  before trusting any ZH attention metrics — a wrong span silently produces
  plausible-looking numbers.

Suggested check (once the ZH model lands, but SP is testable now): for ~20
random samples per language, print the tokenized sentence with the located PET
span highlighted next to `pet_instance`, and confirm alignment by eye. Worth
wiring as a small `--debug-pet` flag or standalone script so it can be re-run
per language/model.

## 3. Tests
- test using layer 11/10 as output
- check PET encoding values in euph/non-euph contexts
  - save encodings for 10/11/12 at training to compare

## 4. Color
- check paper for demographics
- sanity check for demog differences
- gender differences in LLM output