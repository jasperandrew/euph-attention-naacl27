"""
Report classifier accuracy and F1 on the selected run's test split.

Usage:
  python evaluate.py --model-dir runs/en/<run-name>
  python evaluate.py --model-dir runs/en/<run-name> --batch-size 32
"""
import argparse
import logging
import os

from src.cli import add_json_arg, add_model_arg, init_model, write_json
from src.stats.evaluation import TARGET_NAMES, evaluate_model
from src.stats.models import load_classifier
from torch.utils.data import DataLoader
from src import config
from src.data_utils import load_splits, EuphemismDataset

logger = logging.getLogger(__name__)


def load_model(model_dir):
    return load_classifier(model_dir)


def main(args):
    logger.info("Loading finetuned model...")
    model = load_model(config.MODEL_DIR)
    logger.info(f"  Device: {next(model.parameters()).device}")

    logger.info("Loading held-out test split...")
    _, test_df = load_splits()
    if args.max_samples is not None:
        test_df = test_df.head(args.max_samples)
    if 'id' not in test_df.columns:
        raise ValueError("The selected test split must contain an 'id' column.")
    row_ids = test_df['id'].astype(str).tolist()
    logger.info(f"  {len(test_df)} samples")

    logger.info("Tokenizing...")
    test_ds = EuphemismDataset(test_df)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    logger.info("Running inference...")
    result = evaluate_model(model, loader, row_ids)
    logger.info(f"\nAccuracy: {result['accuracy']:.4f}")
    logger.debug("\nClassification report:")
    logger.debug(result['classification_report_text'])
    logger.debug("Confusion matrix (rows = true, cols = predicted):")
    logger.debug(f"  {'':>17}{'Pred Non-Euph':>17}{'Pred Euph':>13}")
    for name, row in zip(TARGET_NAMES, result['confusion_matrix']):
        logger.debug(f"  {name:>17}{row[0]:>17}{row[1]:>13}")

    split_name = 'test' if os.path.isfile(
        os.path.join(config.MODEL_RUN_DIR, 'split_test.csv')) else 'legacy_eval'
    result.pop('classification_report_text')
    write_json(args.json_out, {
        'script': 'evaluate.py',
        'model_dir': config.MODEL_DIR,
        'split_dir': config.MODEL_RUN_DIR,
        'split': split_name,
        **result,
    })
    if args.json_out:
        logger.info(f"JSON results written to {args.json_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classifier accuracy/F1 on the selected test split")
    add_model_arg(parser)
    add_json_arg(parser)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit the evaluation subset (for quick checks).')
    parser.add_argument('--verbose', action='store_true',
                        help='Show the detailed classification report in the console.')
    args = parser.parse_args()
    init_model(args)

    main(args)
