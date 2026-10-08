# v2 Review Recommendations

Additional perturbation controls — vzth, bDAg
- Mask context tokens.
- Delete the PET or use a context-only baseline.
- Compare PET masking with random-token replacement.

Broader replication/generalization — Area Chair, nSpM, vzth, bDAg
- Additional architectures, datasets, and languages.
- Multiple random seeds. ***
- Other fine-tuning regimes or freeze boundaries.

Align late-layer evidence with the revised claim — derived from Area Chair, vzth, bDAg
- Expand the detailed controls and appendix statistics beyond layer 11 to layers 10–12. ***

Improve classifier performance — vzth
- Train/select a stronger classifier, so the analysis describes a more capable decision strategy.

Ensure released artifacts match the paper — nSpM, bDAg
- The referenced finetuning notebook should be present in the released repository, alongside clear data-reproduction instructions.

NLP Lab Notes
- Framing: Frozen-first or unfrozen-first?
- Training: 1 new seed, 1 new split
  - check default seed
  - find split (Julia's?) on github
- Do the attn metrics depend on # of PETs (not instances)
- Libby Barak author!

Anna Notes
```
Hi all,

Jasper talked about his paper on Friday. My mind was blank and I was constantly distracted by emails shooting at me from every direction, but Libby was very sharp and helpful. Here's what I remember from that discussion: 

We should repeat both training regimes with multiple fixed, documented seeds - we are already using 20 epochs, right? (Section 3) Should we run both 20 epoch training regimes with 5 fixed seeds each?  The claim will be much stronger if the classification results and the main attention patterns hold across runs.

We also talked about reproducibility -> record the training and control seeds, save the actual data splits, document all the settings.

Some additional comments (based on the current paper in overleaf): 

Section 5.2 -> the 71.7% masking baseline matches the partial-freeze model, not the primary model. 
From Table 8: 330 + 84 = 414, and 414/618 = 67.0%. For the partial-freeze model: (347 + 96)/618 = 71.7%. This needs to be fixed. 

 The figures and tables do not match --  check Fig2 against Table 5, and Fig 3 against Table 8. The unmasked values in Fig 2 should also agree with Table 4. Fig. 4 shows absolute attention profiles, not masking results, so Fig. 4/Table 5 is not the corresponding pair. For example, Fig. 3 shows inward attention for correct predictions at roughly 0.5 but Table 8 reports 1.084. 

I think we should regenerate all figures and tables from the same identified results files. (I think the same happened with our color paper ... -- my fault)

Fig. 5 -> the caption and highlighted heads do not agree. 

Section 5.2 ->  Maybe: Inward attention remains significant after masking; internal and outward attention become non-significant, while entropy remains non-significant. --> it's just that entropy cannot disappear...

Also mention that the masked sink effect is significant (masking results)

Section 3 -> the evaluation set is also used to select the best checkpoint? As written, there is no separate validation set??? 

Masking results -> report masked accuracy, macro F1, and how many flips introduce errors versus correct existing errors. I think  86.1% agreement only tells us that most predicted labels stay the same -- it does not tell us whether accuracy is preserved.  so, the interpretation is more limited --  the intervention removes the original PET tokens, but preserves their location, token count, and surrounding contextual clues. 


Random-span control --- > directly test whether the class effect is stronger at PETs than at matched random spans. A significant PET effect and a non-significant random-span effect do not show that the two effects differ. Compute a PET-minus-random value for each example, using multiple random draws, and compare this difference across classes.


Label-shuffle control  --  twenty permutations are a limited check. Appendix A reports one permutation with 13 Bonferroni-significant tests, so we should inspect that result and run more permutations before saying the control _confirms_ that the findings are not due to chance. 

In the Data secton -- report the number of unique PETs, PET overlap across splits, and label distribution by PET. 

Section 5.5 -> be careful with the correct vs. misclassified comparison. In the error group, true and predicted labels are reversed, so an effect associated with the model's prediction can also reverse. Maybe report the four groups separately?

LAyer 12 -- specify how classification is done. If we use the standard <s> representation for classification, we should probably also examine <s> --> PET attention? We dont have <s> as a query, right? 


For the partial-freeze model, do we freeze the embeddings too? 
```

Meeting Notes
- how is attention/entropy computed?
- clarify equation variables
- verify attention summation -- sums to 1? how does both inward and outward attn increase?
  - why does entropy increase?
- absolute significance/cohen's d references in text body
- consistent use of d
- pet masking chart is not very readable
- Fig 2 - random span as well?
- story with examples. why is inward/internal attn increasing? why not outward?
- pet-only classification

# Other TODOs

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