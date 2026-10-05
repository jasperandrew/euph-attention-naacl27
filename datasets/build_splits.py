#!/usr/bin/env python3
"""Build reproducible stratified 80/10/10 splits for processed euphemism data."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PARTITIONS = ("train", "val", "test")


def _id_key(value):
    """Compare IDs from differently formatted CSVs (e.g. 12 and 12.0) safely."""
    if pd.isna(value):
        raise ValueError("IDs cannot be missing")
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _validate_ids(frame, description):
    keys = frame["id"].map(_id_key)
    duplicates = keys[keys.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ValueError(f"{description} contains duplicate IDs: {duplicates[:10]}")
    return keys


def _validate_split(parts, all_keys, labels, description, check_balance=True):
    sets = {name: set(frame["id"].map(_id_key)) for name, frame in parts.items()}
    for i, left in enumerate(PARTITIONS):
        for right in PARTITIONS[i + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"{description}: {left}/{right} overlap: {sorted(overlap)[:10]}")
    combined = set.union(*sets.values())
    if combined != all_keys:
        missing, extra = all_keys - combined, combined - all_keys
        raise ValueError(f"{description}: incomplete coverage; missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]}")

    full_rates = labels.value_counts(normalize=True)
    max_delta = 0.0
    for name, frame in parts.items():
        rates = frame["label"].value_counts(normalize=True)
        delta = max((abs(float(rates.get(label, 0)) - float(rate)) for label, rate in full_rates.items()), default=0.0)
        max_delta = max(max_delta, delta)
    if check_balance and max_delta > 0.02:
        raise ValueError(f"{description}: label proportions differ from full data by {max_delta:.1%} (limit 2%)")


def _report(parts, title):
    print(f"\n{title}")
    pet_sets = {}
    for name in PARTITIONS:
        frame = parts[name]
        counts = frame["label"].value_counts().sort_index().to_dict()
        pets = set(frame["pet"].dropna().astype(str))
        pet_sets[name] = pets
        print(f"  {name}: {len(frame)} rows; labels={counts}; unique PETs={len(pets)}")
    for i, left in enumerate(PARTITIONS):
        for right in PARTITIONS[i + 1 :]:
            overlap = pet_sets[left] & pet_sets[right]
            print(f"  PET overlap {left}/{right}: {len(overlap)} unique PETs")


def _write(parts, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in PARTITIONS:
        parts[name].to_csv(directory / f"{name}.csv", index=False)


def _generated_splits(data, n, seed):
    result = []
    for current_seed in range(seed, seed + n):
        train, remainder = train_test_split(
            data, train_size=0.8, random_state=current_seed, stratify=data["label"]
        )
        val, test = train_test_split(
            remainder, test_size=0.5, random_state=current_seed, stratify=remainder["label"]
        )
        parts = {"train": train, "val": val, "test": test}
        _validate_split(parts, set(data["id"].map(_id_key)), data["label"], f"seed {current_seed}")
        result.append((current_seed, parts))
    return result


def _lrec_split(data, lrec_dir):
    paths = {
        "train": lrec_dir / "train_20.csv",
        "val": lrec_dir / "val_20.csv",
        "test": lrec_dir / "test_0_english.csv",
    }
    membership = {}
    for partition, path in paths.items():
        frame = pd.read_csv(path)
        if "ID" not in frame.columns:
            raise ValueError(f"{path} must contain an ID column")
        keys = frame["ID"].map(_id_key)
        duplicates = keys[keys.duplicated(keep=False)].unique().tolist()
        if duplicates:
            raise ValueError(f"{path} contains duplicate IDs: {duplicates[:10]}")
        membership[partition] = set(keys)

    for i, left in enumerate(PARTITIONS):
        for right in PARTITIONS[i + 1 :]:
            overlap = membership[left] & membership[right]
            if overlap:
                raise ValueError(f"LREC {left}/{right} overlap: {sorted(overlap)[:10]}")

    processed_keys = set(data["id"].map(_id_key))
    lrec_keys = set.union(*membership.values())
    absent = processed_keys - lrec_keys
    if absent:
        raise ValueError(f"Processed IDs absent from LREC partitions ({len(absent)}): {sorted(absent)[:20]}")
    excluded = lrec_keys - processed_keys
    print(f"\nLREC IDs excluded by preprocessing: {len(excluded)}")
    if excluded:
        print(f"  IDs: {sorted(excluded, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))}")

    keys = data["id"].map(_id_key)
    parts = {name: data.loc[keys.isin(membership[name])].copy() for name in PARTITIONS}
    # The published LREC memberships are fixed and need not be stratified to
    # match this processed dataset's label proportions.
    _validate_split(parts, processed_keys, data["label"], "LREC split", check_balance=False)
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_csv", type=Path, help="Processed dataset CSV (must have id, label, and pet columns)")
    parser.add_argument("n", type=int, help="Number of generated splits")
    parser.add_argument("seed", type=int, help="Starting random seed")
    parser.add_argument("output_dir", type=Path, help="Directory in which split directories are created")
    parser.add_argument("--lrec-dir", type=Path, help="Directory containing train_20.csv, val_20.csv, and test_0_english.csv")
    args = parser.parse_args()

    if args.n <= 0:
        parser.error("n must be positive")
    data = pd.read_csv(args.processed_csv)
    required = {"id", "label", "pet"}
    missing = required - set(data.columns)
    if missing:
        parser.error(f"processed CSV is missing required columns: {sorted(missing)}")
    keys = _validate_ids(data, "processed CSV")
    if data["label"].isna().any():
        parser.error("processed CSV labels cannot be missing")

    for current_seed, parts in _generated_splits(data, args.n, args.seed):
        directory = args.output_dir / f"seed_{current_seed}"
        _write(parts, directory)
        _report(parts, f"Seed {current_seed} ({directory})")

    if args.lrec_dir:
        parts = _lrec_split(data, args.lrec_dir)
        directory = args.output_dir / "lrec_trial"
        _write(parts, directory)
        _report(parts, f"LREC reproduction ({directory})")


if __name__ == "__main__":
    main()
