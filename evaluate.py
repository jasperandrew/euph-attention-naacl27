"""
Report classifier accuracy and F1 on the selected run's test split.

Usage:
  python evaluate.py --model-dir runs/en/<run-name>
  python evaluate.py --model-dir runs/en/<run-name> --batch-size 32
"""
import argparse
import os

from src.cli import add_json_arg, add_model_arg, init_model, write_json


def load_model(model_dir):
    # Match the training notebook's explicit attention backend.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, attn_implementation='eager'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model


def predict(model, loader):
    all_preds, all_labels, all_probabilities = [], [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            # TrainingArguments(bf16=True) makes Trainer use BF16 autocast for
            # evaluation on CUDA. Keep CPU evaluation in full precision.
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
            else:
                logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            all_probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            all_labels.extend(batch['labels'].tolist())
    return all_preds, all_labels, all_probabilities


def main(args):
    print("Loading finetuned model...")
    model = load_model(config.MODEL_DIR)
    print(f"  Device: {next(model.parameters()).device}")

    print("Loading held-out test split...")
    _, test_df = load_splits()
    if 'id' not in test_df.columns:
        raise ValueError("The selected test split must contain an 'id' column.")
    row_ids = test_df['id'].astype(str).tolist()
    print(f"  {len(test_df)} samples")

    print("Tokenizing...")
    test_ds = EuphemismDataset(test_df)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print("Running inference...")
    preds, labels, probabilities = predict(model, loader)
    if not (len(row_ids) == len(labels) == len(preds) == len(probabilities)):
        raise RuntimeError("Evaluation IDs, labels, and predictions are not aligned.")

    target_names = ['Non-Euphemistic', 'Euphemistic']

    print(f"\nAccuracy: {accuracy_score(labels, preds):.4f}")
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=target_names, digits=4))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(f"  {'':>17}{'Pred Non-Euph':>17}{'Pred Euph':>13}")
    cm = confusion_matrix(labels, preds)
    for name, row in zip(target_names, cm):
        print(f"  {name:>17}{row[0]:>17}{row[1]:>13}")

    report = classification_report(labels, preds, target_names=target_names,
                                   output_dict=True, zero_division=0)
    split_name = 'test' if os.path.isfile(
        os.path.join(config.MODEL_RUN_DIR, 'split_test.csv')) else 'legacy_eval'
    write_json(args.json_out, {
        'script': 'evaluate.py',
        'model_dir': config.MODEL_DIR,
        'split_dir': config.MODEL_RUN_DIR,
        'split': split_name,
        'n_samples': len(labels),
        'accuracy': accuracy_score(labels, preds),
        'classification_report': report,
        'confusion_matrix': cm,
        'confusion_matrix_labels': target_names,
        'class_probability_order': target_names,
        'predictions': [
            {
                'id': row_id,
                'true_label': int(y),
                'predicted_label': int(p),
                'class_probabilities': {
                    target_names[0]: float(prob[0]),
                    target_names[1]: float(prob[1]),
                },
            }
            for row_id, y, p, prob in zip(row_ids, labels, preds, probabilities)
        ],
    })
    if args.json_out:
        print(f"JSON results written to {args.json_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classifier accuracy/F1 on the selected test split")
    add_model_arg(parser)
    add_json_arg(parser)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    init_model(args)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from src import config
    from src.data_utils import load_splits, EuphemismDataset

    main(args)
