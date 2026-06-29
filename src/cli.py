"""Shared --lang plumbing for the entry scripts."""
import os

from . import config


def add_lang_arg(parser):
    parser.add_argument('--lang', choices=config.LANGS, default=os.environ.get('EUPH_LANG', 'en'),
                        help="Which finetuned model/dataset to use (model/<lang>, default: en)")


def init_lang(args):
    """Apply --lang; call before importing src.data_utils (it loads the tokenizer)."""
    config.set_lang(args.lang)
