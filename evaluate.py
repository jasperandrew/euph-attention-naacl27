"""
Report classifier accuracy and F1 on the eval split.

Usage:
  python evaluate.py
  python evaluate.py --batch-size 32
"""
import argparse


def load_model(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model


def predict(model, loader):
    all_preds, all_labels = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            all_labels.extend(batch['labels'].tolist())
    return all_preds, all_labels


def main(args):
    print("Loading finetuned model...")
    model = load_model(config.MODEL_DIR)
    print(f"  Device: {next(model.parameters()).device}")

    print("Loading eval split...")
    _, eval_df = load_splits()
    print(f"  {len(eval_df)} samples")

    print("Tokenizing...")
    eval_ds = EuphemismDataset(eval_df)
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    print("Running inference...")
    preds, labels = predict(model, loader)

    target_names = ['Non-Euphemistic', 'Euphemistic']

    print(f"\nAccuracy: {accuracy_score(labels, preds):.4f}")
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=target_names, digits=4))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(f"  {'':>17}{'Pred Non-Euph':>17}{'Pred Euph':>13}")
    cm = confusion_matrix(labels, preds)
    for name, row in zip(target_names, cm):
        print(f"  {name:>17}{row[0]:>17}{row[1]:>13}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classifier accuracy/F1 on the eval split")
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from src import config
    from src.data_utils import load_splits, EuphemismDataset

    main(args)
