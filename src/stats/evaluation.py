"""Model evaluation routines independent of command-line configuration."""

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


TARGET_NAMES = ['Non-Euphemistic', 'Euphemistic']


def predict(model, loader):
    """Collect class predictions, labels, and probabilities from a data loader."""
    all_preds, all_labels, all_probabilities = [], [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            # Match the training notebook's BF16 evaluation on CUDA.
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(
                        input_ids=input_ids, attention_mask=attn_mask,
                        output_attentions=False,
                    ).logits
            else:
                logits = model(
                    input_ids=input_ids, attention_mask=attn_mask,
                    output_attentions=False,
                ).logits
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            all_probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            all_labels.extend(batch['labels'].tolist())
    return all_preds, all_labels, all_probabilities


def evaluate_model(model, loader, row_ids):
    """Evaluate a model and return JSON-ready metrics and per-example predictions."""
    preds, labels, probabilities = predict(model, loader)
    if not (len(row_ids) == len(labels) == len(preds) == len(probabilities)):
        raise RuntimeError('Evaluation IDs, labels, and predictions are not aligned.')

    report = classification_report(
        labels, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0,
    )
    return {
        'n_samples': len(labels),
        'accuracy': float(accuracy_score(labels, preds)),
        'classification_report': report,
        'classification_report_text': classification_report(
            labels, preds, target_names=TARGET_NAMES, digits=4,
        ),
        'confusion_matrix': confusion_matrix(labels, preds).tolist(),
        'confusion_matrix_labels': TARGET_NAMES,
        'class_probability_order': TARGET_NAMES,
        'predictions': [
            {
                'id': row_id,
                'true_label': int(y),
                'predicted_label': int(pred),
                'class_probabilities': {
                    TARGET_NAMES[0]: float(prob[0]),
                    TARGET_NAMES[1]: float(prob[1]),
                },
            }
            for row_id, y, pred, prob in zip(row_ids, labels, preds, probabilities)
        ],
    }
