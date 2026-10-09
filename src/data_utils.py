import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from . import config

logger = logging.getLogger(__name__)

_TOKENIZER = None
_TOKENIZER_KEY = None


def get_tokenizer():
    """Load and cache the tokenizer for the currently selected model."""
    global _TOKENIZER, _TOKENIZER_KEY
    if not config.MODEL_DIR:
        raise ValueError('Select a model with --model-dir before loading the tokenizer.')

    key = (os.path.abspath(config.MODEL_DIR), config.BOUNDED, config.BOUND_TOKEN)
    if key != _TOKENIZER_KEY:
        if not os.path.isfile(os.path.join(config.MODEL_DIR, 'config.json')):
            raise FileNotFoundError(
                f"No fine-tuned model config found at {config.MODEL_DIR}; "
                'select a run with --model-dir.'
            )
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
        if config.BOUNDED:
            tokenizer.add_special_tokens(
                {'additional_special_tokens': [config.BOUND_TOKEN]}
            )
        _TOKENIZER, _TOKENIZER_KEY = tokenizer, key
    return _TOKENIZER


def tokenize(text):
    return get_tokenizer()(
        text, max_length=512, padding='max_length', truncation=True,
        return_tensors='pt',
    )


def get_tokens(tokenized_input):
    tokens = get_tokenizer().convert_ids_to_tokens(
        tokenized_input['input_ids'][0].tolist()
    )
    seq_len = (tokenized_input['attention_mask'] == 1).sum().item()
    return tokens[:seq_len]


def load_splits():
    """Return (train, held-out test); supports legacy model runs with split_eval.csv."""
    train_path = os.path.join(config.MODEL_RUN_DIR, 'split_train.csv')
    test_path = os.path.join(config.MODEL_RUN_DIR, 'split_test.csv')
    eval_path = os.path.join(config.MODEL_RUN_DIR, 'split_eval.csv')
    if os.path.exists(train_path) and os.path.exists(test_path):
        logger.info("Loading saved train split and held-out test split from model run directory.")
        return pd.read_csv(train_path), pd.read_csv(test_path)
    if os.path.exists(train_path) and os.path.exists(eval_path):
        logger.info("Loading legacy train/eval splits; using eval as the held-out split.")
        return pd.read_csv(train_path), pd.read_csv(eval_path)

    csv_path = os.path.join(config.DATASET_DIR, config.CSV_NAME)
    logger.info(f"Creating train/test split from {csv_path}")
    data = pd.read_csv(csv_path)
    return train_test_split(data, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)


class EuphemismDataset(Dataset):
    def __init__(self, data):
        tokenized_data = [tokenize(text) for text in data[config.DATA_COLUMN].tolist()]
        encoded_labels = LabelEncoder().fit_transform(data['label'])
        self.input_ids = torch.cat([item['input_ids'] for item in tokenized_data], dim=0)
        self.attention_mask = torch.cat([item['attention_mask'] for item in tokenized_data], dim=0)
        self.labels = torch.tensor(encoded_labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }
