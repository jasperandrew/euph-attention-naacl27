# euph-attention-v2

## Replication Instructions

### Environment

1. Install `python==3.12.12` (or use a conda environment, or whatever you like)
2. Run `python -m ensurepip --upgrade`
3. Run `pip install -r requirements.txt`
4. Install the correct version of [pytorch](https://pytorch.org/get-started/locally/)
5. All main scripts have help text, run `python <file> -h` for usage info

### Euphemism Dataset (as of 05/2026)
1. Download [EN-euphs-2025.csv](https://github.com/NLPlabMSU/lrec-euph/blob/main/Datasets/final-datasets/csv/EN-euphs-2025.csv) from the prior study's project repo, and place it in the `datasets` folder.
2. Run `process-euphs.py`

### XLM-RoBERTa Finetuning
- Use [euph_finetune.ipynb](./model/euph_finetune.ipynb)
- We trained in Google Colab, using the L4 GPU accelerator runtime.
- The notebook may require modifications to run on different hardware.