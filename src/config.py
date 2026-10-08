import os
import json

# Project root (one level above src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')

DATA_COLUMN = 'text_raw'
CSV_NAME = 'EN-euphs-2025-processed.csv'
PLOTS_DIR = os.path.join(BASE_DIR, 'output', 'plots')
TEST_SIZE = 0.2
RANDOM_SEED = 42

MODEL_NAME = 'FacebookAI/xlm-roberta-base'
TOK_DELIM = '▁'
BOUNDED = False
BOUND_TOKEN = '[PET_BOUNDARY]'
FROZEN_LAYERS = 7
NUM_LAYERS = 12
NUM_HEADS = 12

# Hand-picked English example sentence pairs for plot_pairs.py.
# pet_tokens are in XLM-RoBERTa tokenizer format (▁ prefix for word-initial
# subwords). 'samples' drive the multi-sample sink-attention analyses.
EXAMPLE_EUPH_N = "They passed on to the next round."
EXAMPLE_EUPH_Y = "They passed on to the next life."
EXAMPLE_PET_TOKENS = ['▁passed', '▁on']
SAMPLES = [
    {
        'euph_n': "They passed on to the next round.",
        'euph_y': "They passed on to the next life.",
        'pet_phrase': 'passed on',
    },
    {
        'euph_n': "They let him go to the event.",
        'euph_y': "They let him go to save money.",
        'pet_phrase': 'let him go',
    },
    {
        'euph_n': "The acid was neutralized.",
        'euph_y': "The target was neutralized.",
        'pet_phrase': 'neutralized',
    },
    {
        'euph_n': "He is getting on in his career.",
        'euph_y': "He is getting on in years.",
        'pet_phrase': 'getting on',
    },
    {
        'euph_n': "She is expecting a package.",
        'euph_y': "She is expecting a baby.",
        'pet_phrase': 'expecting',
    },
]

MODEL_DIR = None
MODEL_RUN_DIR = None
MODEL_RUN_NAME = None


def set_model_dir(path):
    """Set the exact fine-tuned model to use; accepts a run dir or its model/ child."""
    global MODEL_DIR, MODEL_RUN_DIR, MODEL_RUN_NAME, FROZEN_LAYERS
    path = os.path.abspath(os.path.expanduser(path))
    nested_model = os.path.join(path, 'model')
    if os.path.isfile(os.path.join(nested_model, 'config.json')):
        MODEL_RUN_DIR, MODEL_DIR = path, nested_model
    else:
        MODEL_DIR = path
        MODEL_RUN_DIR = os.path.dirname(path) if os.path.basename(path) == 'model' else path
    if not os.path.isfile(os.path.join(MODEL_DIR, 'config.json')):
        raise FileNotFoundError(f"No model config.json found in --model-dir path: {path}")
    MODEL_RUN_NAME = os.path.basename(MODEL_RUN_DIR.rstrip(os.sep)) or 'model'
    manifest_path = os.path.join(MODEL_RUN_DIR, 'run_manifest.json')
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
        if 'frozen_transformer_layers' in manifest:
            FROZEN_LAYERS = int(manifest['frozen_transformer_layers'])
