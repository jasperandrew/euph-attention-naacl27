import os

# Project root (one level above src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')

LANGS = ['en', 'sp', 'zh']

DATA_COLUMN = 'text_raw'
TEST_SIZE = 0.2
RANDOM_SEED = 42

MODEL_NAME = 'FacebookAI/xlm-roberta-base'
TOK_DELIM = '▁'
BOUNDED = False
BOUND_TOKEN = '[PET_BOUNDARY]'
FROZEN_LAYERS = 7
NUM_LAYERS = 12
NUM_HEADS = 12

# Hand-picked example sentence pairs for plot_pairs.py, per language.
# pet_tokens are in XLM-RoBERTa tokenizer format (▁ prefix for word-initial
# subwords). plot_pairs.py rejects languages without an entry here.
# 'samples' drive the multi-sample sink-attention analyses.
PAIR_EXAMPLES = {
    'en': {
        'euph_n': "They passed on to the next round.",
        'euph_y': "They passed on to the next life.",
        'pet_tokens': ['▁passed', '▁on'],
        'samples': [
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
        ],
    },
}


def set_lang(lang):
    """Set the active language and re-derive everything language-scoped.

    Entry scripts call this (via src.cli.init_lang) right after argparse and
    before importing src.data_utils, which loads the tokenizer from MODEL_DIR.
    """
    global LANG, MODEL_DIR, PLOTS_DIR, CSV_NAME
    global EXAMPLE_EUPH_N, EXAMPLE_EUPH_Y, EXAMPLE_PET_TOKENS, SAMPLES
    LANG = lang.lower()
    MODEL_DIR = os.path.join(BASE_DIR, 'model', LANG)
    PLOTS_DIR = os.path.join(BASE_DIR, 'plots', LANG)
    CSV_NAME = f'{LANG.upper()}-euphs-2025-processed.csv'
    examples = PAIR_EXAMPLES.get(LANG, {})
    EXAMPLE_EUPH_N = examples.get('euph_n')
    EXAMPLE_EUPH_Y = examples.get('euph_y')
    EXAMPLE_PET_TOKENS = examples.get('pet_tokens', [])
    SAMPLES = examples.get('samples', [])


set_lang(os.environ.get('EUPH_LANG', 'en'))
