"""Compare raw classifier predictions with an always-euphemistic predictor.

For each full-model (freeze-0) run, a raw prediction of 0 (non-euphemistic)
disagrees with the constant euphemistic prediction; a raw prediction of 1
agrees. This is not the PET-masking flip rate.
"""

import csv
from pathlib import Path


RUNS_DIR = Path(__file__).resolve().parent / "runs"
SPLITS = {
    "Stratified": "split-42_train-",
    "PET-disjoint": "split-lrec_train-",
}


def count_predictions(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labels = [row["predicted_label"] for row in rows]
    if not labels or any(label not in {"0", "1"} for label in labels):
        raise ValueError(f"Missing or invalid predicted_label values in {path}")

    # Always-euph predicts 1: raw 0s flip, raw 1s agree.
    return len(labels), labels.count("0"), labels.count("1")


def report(split, prefix):
    paths = sorted(
        path
        for path in RUNS_DIR.glob(f"{prefix}*_freeze-0/test_predictions.csv")
        if path.is_file()
    )
    if not paths:
        raise FileNotFoundError(f"No full-model prediction files found for {split}")

    total = flips = agreements = 0
    print(f"{split} (always-euphemistic comparison)")
    for path in paths:
        n, run_flips, run_agreements = count_predictions(path)
        total += n
        flips += run_flips
        agreements += run_agreements
        print(
            f"  {path.parent.name}: disagreement {run_flips}/{n} "
            f"({run_flips / n:.1%}); agreement {run_agreements}/{n} "
            f"({run_agreements / n:.1%})"
        )

    print(
        f"  pooled: disagreement {flips}/{total} ({flips / total:.1%}); "
        f"agreement {agreements}/{total} ({agreements / total:.1%})"
    )


if __name__ == "__main__":
    for split, prefix in SPLITS.items():
        report(split, prefix)
