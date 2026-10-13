# Positional PETs: Mapping Attention Flow in Finetuned Euphemism Detection
Jasper Andrew (andrewj2@montclair.edu), Anna Feldman (feldmana@montclair.edu), Jing Peng (pengj@montclair.edu), Libby Barak (barakl@montclair.edu)

## Replication Instructions

### Environment

1. Install `python==3.12.12` (or use a conda environment, or whatever you like)
2. Run `python -m ensurepip --upgrade`
3. Run `pip install -r requirements.txt`
4. Install the correct version of [pytorch](https://pytorch.org/get-started/locally/)
5. All main scripts have help text, run `python <file> -h` for usage info

### Analysis scripts

Model-dependent scripts require an explicit fine-tuned model path; no run is
selected automatically. Pass either a run directory (containing `model/`) or
the model directory itself. For example:

```bash
python evaluate.py --model-dir runs/en/seed_42_freeze0_seed43_20261005_0847 --json-out output/eval.json
python plot_profile.py --model-dir runs/en/seed_42_freeze0_seed43_20261005_0847 --json-out output/profile.json
```

`evaluate.py`, `plot_profile.py`, `plot_pairs.py`, `significance.py`, and
`sanity_checks.py` write structured JSON results by default under
`output/stats/<run-name>/`; use `--json-out PATH` to choose another location.
Plots and profiler caches go under `output/plots/<run-name>/`. Console progress
is still produced. When a run has saved splits, these scripts evaluate its test
split. Outputs are separated by run name so results from different checkpoints
do not overwrite or reuse one another.

To run evaluation and statistics sequentially while loading each model only
once, use `all_stats.py`. It runs all three workflows by default; select a
subset with `--only`:

```bash
python all_stats.py --model-dir runs/en/<run-name>
python all_stats.py --model-dir runs/en/<run-name> --only evaluate significance
python all_stats.py --all-models
python all_stats.py --all-models --jobs 2
```

`--all-models` discovers run directories containing `model/config.json` under
`runs/`. Runs are serial by default. `--jobs N` processes up to N model runs at
once; each worker can hold both a fine-tuned and a base model, so increase N
only when device memory allows it.

### Euphemism Dataset (as of 05/2026)
1. Download [EN-euphs-2025.csv](https://github.com/NLPlabMSU/lrec-euph/blob/main/Datasets/final-datasets/csv/EN-euphs-2025.csv) from the prior study's project repo, and place it in the `datasets` folder.
2. Run `process-euphs.py`

### XLM-RoBERTa Finetuning
- Use [euph_finetune.ipynb](./model/euph_finetune.ipynb)
- We trained in Google Colab, using the L4 GPU accelerator runtime.
- The notebook may require modifications to run on different hardware.
