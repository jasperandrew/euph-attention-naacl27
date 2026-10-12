# v2 Review Recommendations

Additional perturbation controls — vzth, bDAg
- Mask context tokens.
- Delete the PET or use a context-only baseline.
- Compare PET masking with random-token replacement.

Broader replication/generalization — Area Chair, nSpM, vzth, bDAg
- Additional architectures, datasets, and languages.
- Other fine-tuning regimes or freeze boundaries.

Align late-layer evidence with the revised claim — derived from Area Chair, vzth, bDAg
- Expand the detailed controls and appendix statistics beyond layer 11 to layers 10–12. ***

Improve classifier performance — vzth
- Train/select a stronger classifier, so the analysis describes a more capable decision strategy.

Ensure released artifacts match the paper — nSpM, bDAg
- The referenced finetuning notebook should be present in the released repository, alongside clear data-reproduction instructions.

# Anna Notes
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

# Meeting Notes
- absolute significance/cohen's d references in text body
- consistent use of d
- pet masking chart is not very readable
- Fig 2 - random span as well?
- story with examples. why is inward/internal attn increasing? why not outward?
- pet-only classification
- variance, SD across runs

# Other TODOs

## 1. `FROZEN_LAYERS` fallback for models without manifests

Current run directories record `frozen_transformer_layers` in
`run_manifest.json`, and `config.set_model_dir()` loads that value before
significance and plotting workflows use `FROZEN_LAYERS`. The current runs are
therefore labeled and filtered using their per-run freeze setting.

The remaining risk is the module default, `FROZEN_LAYERS = 7`: selecting a
model directory without a manifest, or with a manifest missing
`frozen_transformer_layers`, can silently use the wrong freeze count. Consider
requiring this metadata or warning when it is absent. The old `plot_pairs.py`
reference was removed; do not use existing `notes/outputs/*.out` as current
results without checking which model run produced them.

## 3. Tests
- test using layer 11/10 as output
- check PET encoding values in euph/non-euph contexts
  - save encodings for 10/11/12 at training to compare

## 4. Color
- check paper for demographics
- sanity check for demog differences
- gender differences in LLM output
