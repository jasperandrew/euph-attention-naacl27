import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from . import config

TOKENIZER = AutoTokenizer.from_pretrained(config.MODEL_DIR)
if config.BOUNDED:
    TOKENIZER.add_special_tokens({'additional_special_tokens': [config.BOUND_TOKEN]})


def tokenize(text):
    return TOKENIZER(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')


def get_tokens(tokenized_input):
    tokens = TOKENIZER.convert_ids_to_tokens(tokenized_input['input_ids'][0].tolist())
    seq_len = (tokenized_input['attention_mask'] == 1).sum().item()
    return tokens[:seq_len]


def load_splits():
    """Load pre-saved train/eval splits from the model directory, or re-split from CSV."""
    train_path = os.path.join(config.MODEL_DIR, 'split_train.csv')
    eval_path = os.path.join(config.MODEL_DIR, 'split_eval.csv')
    if os.path.exists(train_path) and os.path.exists(eval_path):
        print("Loading saved train/eval splits from model directory.")
        return pd.read_csv(train_path), pd.read_csv(eval_path)

    csv_path = os.path.join(config.DATASET_DIR, config.CSV_NAME)
    print(f"Splitting dataset from {csv_path}")
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
