import re
import sys
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

# EN FIXED ROWS (corrected in source CSV)
# 2038: "downsizedc.org" -> "downsize-dc.org"
# 2989: "between jobsi've" -> "between jobs i've"

# EN REMOVED ROWS
# 2134: "... from X Archipelago all the way to ..." =/= "go all the way" PET
# 2314,2319,2435: "over the hills" =/= "over the hill" PET
# 2321: "over the hilly..." =/= "over the hill" PET
# 2589: "sleep without" =/= "sleep with" PET
# 2605,2613,2615,2622: "stoutly" =/= "stout" PET

# split_mode: how to repair a word fragment glued to a boundary marker.
#   "fold"  — fragment is a piece of the PET word (EN: "downsize[B]d" -> PET
#             "downsized"); fold it inside the boundaries.
#   "space" — fragment is a separate word missing its space (SP: "la[B]Querida"
#             -> "la [B]Querida"); insert a space, keep it outside the PET.
#   None    — no repair. In ZH every neighboring Han character matches \w, so
#             either repair would absorb surrounding text into the PET.
LANGS = {
    "EN": {
        "split_mode": "fold",
        "bad_ids": [
            2134,
            2314, 2319, 2435,
            2321,
            2589,
            2605, 2613, 2615, 2622,
        ],
    },
    "SP": {"split_mode": "space", "bad_ids": []},
    "ZH": {"split_mode": None, "bad_ids": []},
}

BOUND = "[PET_BOUNDARY]"
BOUND_RE = re.compile(re.escape(BOUND))
SPLIT_RE = re.compile(r"(\w*)\[PET_BOUNDARY\](.*?)\[PET_BOUNDARY\](\w*)")

# pet_instance may be morphologically inflected relative to the base pet
# ("pass on" -> "passed on", "lose [pro] lunch" -> "lose their lunch"), so
# spans are compared by similarity, never exact match. Rows scoring below
# SIM_WARN are reported as BAD_IDS candidates, not rejected.
SIM_WARN = 0.6

OUT_COLS = ["id", "text_raw", "text_bounded", "pet", "pet_instance", "label", "category", "euph_status"]


def normalize(s):
    s = unicodedata.normalize("NFKD", str(s).lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


def similarity(pet, span):
    return SequenceMatcher(None, normalize(pet), normalize(span)).ratio()


def select_pair(text, pet, row_id, report):
    """Reduce text to exactly one boundary pair. The markers cannot be
    trusted to come in well-formed consecutive pairs (stray or unpaired
    markers occur), so every span between two consecutive markers is a
    candidate PET — e.g. with 4 markers the true PET may sit between the
    middle two. Keep the candidate most similar to the base pet and strip
    all other markers."""
    positions = [m.start() for m in BOUND_RE.finditer(text)]
    if len(positions) < 2:
        report["no_pair"].append(row_id)
        return None
    if len(positions) == 2:
        return text

    spans = [text[a + len(BOUND):b] for a, b in zip(positions, positions[1:])]
    scores = [similarity(pet, s) for s in spans]
    best = max(range(len(spans)), key=scores.__getitem__)

    key = "odd_count" if len(positions) % 2 else "multi_pair"
    report[key].append((
        row_id, len(positions), spans[best], scores[best],
        sorted({s for i, s in enumerate(spans) if i != best}),
    ))
    a, b = positions[best], positions[best + 1]
    return (text[:a].replace(BOUND, "")
            + text[a:b + len(BOUND)]
            + text[b + len(BOUND):].replace(BOUND, ""))


def _respace(match):
    prefix, inner, suffix = match.group(1), match.group(2), match.group(3)
    return (f"{prefix} " if prefix else "") + BOUND + inner + BOUND + (f" {suffix}" if suffix else "")


def process_row(row, split_mode, report):
    text = select_pair(str(row["text"]), row["pet"], row["id"], report)
    if text is None:
        return pd.Series({"pet_instance": "", "text_bounded": str(row["text"])})

    prefix, inner, suffix = SPLIT_RE.search(text).groups()
    instance = inner
    if split_mode and (prefix or suffix):
        if prefix:
            report["split"].append((row["id"], f"[{prefix}/]{inner}"))
        if suffix:
            report["split"].append((row["id"], f"{inner}[/{suffix}]"))
        if split_mode == "fold":
            instance = prefix + inner + suffix
            text = SPLIT_RE.sub(r"[PET_BOUNDARY]\1\2\3[PET_BOUNDARY]", text)
        else:  # "space"
            text = SPLIT_RE.sub(_respace, text)

    sim = similarity(row["pet"], instance)
    if sim < SIM_WARN:
        report["low_sim"].append((row["id"], row["pet"], instance, sim))
    return pd.Series({"pet_instance": instance, "text_bounded": text})


def _trunc(s, n=60):
    return s if len(s) <= n else s[: n - 1] + "…"


def print_report(lang, report, n_rows, split_mode):
    print(f"\n=== {lang} audit report ({n_rows} rows) ===")
    if report["no_pair"]:
        print(f"  FEWER than 2 markers — unprocessable ({len(report['no_pair'])}): ids {report['no_pair']}")
    for key, label in [("odd_count", "ODD marker count — verify carefully"), ("multi_pair", "MULTIPLE marker pairs")]:
        if report[key]:
            print(f"  {label} — kept best-similarity span ({len(report[key])}):")
            for row_id, n, kept, score, rejected in report[key]:
                rej = ", ".join(repr(_trunc(s)) for s in rejected)
                print(f"    id={row_id}: {n} markers, kept {kept!r} (sim={score:.2f}), stripped [{rej}]")
    if report["split"]:
        action = {"fold": "fragment folded into PET", "space": "space inserted, fragment kept outside PET"}[split_mode]
        print(f"  SPLIT boundaries — {action}, verify ({len(report['split'])}):")
        for row_id, desc in report["split"]:
            print(f"    id={row_id}: {desc}")
    if report["low_sim"]:
        print(f"  LOW similarity pet vs pet_instance (< {SIM_WARN}) — BAD_IDS candidates ({len(report['low_sim'])}):")
        for row_id, pet, instance, sim in report["low_sim"]:
            print(f"    id={row_id}: sim={sim:.2f} pet={pet!r} instance={instance!r}")
    if not any(report.values()):
        print("  clean — no anomalies")


def process_language(lang):
    cfg = LANGS[lang]
    df = pd.read_csv(f"{lang}-euphs-2025.csv")

    if cfg["bad_ids"]:
        df = df[~df["id"].isin(cfg["bad_ids"])]

    report = {"no_pair": [], "odd_count": [], "multi_pair": [], "split": [], "low_sim": []}
    df[["pet_instance", "text_bounded"]] = df.apply(
        lambda row: process_row(row, cfg["split_mode"], report), axis=1
    )
    df["text_raw"] = df["text_bounded"].str.replace(BOUND, "", regex=False)

    processed = df["pet_instance"] != ""
    bad_structure = df.loc[processed, "text_bounded"].str.count(re.escape(BOUND)) != 2
    assert not bad_structure.any(), f"{lang}: rows without exactly one boundary pair after processing"

    print_report(lang, report, len(df), cfg["split_mode"])
    out = f"{lang}-euphs-2025-processed.csv"
    df[OUT_COLS].to_csv(out, index=False)
    print(f"Done. {len(df)} rows written to {out}")


if __name__ == "__main__":
    langs = [arg.upper() for arg in sys.argv[1:]] or list(LANGS)
    for lang in langs:
        process_language(lang)
