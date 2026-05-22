import os

# Project root (one level above src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

CSV_NAME = 'EN-euphs-2025-processed.csv'
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

# Example sentence pair for single-example attention plots.
# pet_tokens should be in XLM-RoBERTa tokenizer format (▁ prefix for non-initial subwords).
EXAMPLE_EUPH_N = "They passed on to the next round."
EXAMPLE_EUPH_Y = "They passed on to the next life."
EXAMPLE_PET_TOKENS = ['▁passed', '▁on']

# Multi-sample analysis for sink attention and related metrics.
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
