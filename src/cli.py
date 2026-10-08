import os
import json
import math
from pathlib import Path
import sys

from . import config


def add_model_arg(parser):
    parser.add_argument('--model-dir', required=True,
                        help='Exact fine-tuned model directory, or its run directory containing model/')


def add_json_arg(parser):
    parser.add_argument('--json-out', metavar='PATH',
                        help='Write structured results to this JSON file instead of the default output path')


def _json_safe(value):
    module = type(value).__module__.split('.')[0]
    name = type(value).__name__
    if module == 'pandas' and name == 'DataFrame':
        return _json_safe(value.to_dict(orient='records'))
    if module == 'pandas' and name == 'Series':
        return _json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if module == 'numpy' and name == 'ndarray':
        return _json_safe(value.tolist())
    if module == 'numpy' and hasattr(value, 'item'):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, payload):
    if not path:
        return
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def init_model(args):
    """Apply the explicitly selected model before importing data_utils/tokenizer."""
    config.set_model_dir(args.model_dir)
    if hasattr(args, 'json_out') and args.json_out is None:
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        args.json_out = os.path.join(
            config.BASE_DIR, 'output', 'stats',
            config.MODEL_RUN_NAME, f'{script_name}.json',
        )
