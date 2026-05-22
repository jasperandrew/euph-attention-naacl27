import re
import pandas as pd

# FIXED ROWS
# 2038: "downsizedc.org" -> "downsize-dc.org"
# 2989: "between jobsi've" -> "between jobs i've"

# REMOVED ROWS
# 2134: "... from X Archipelago all the way to ..." =/= "go all the way" PET
# 2314,2319,2435: "over the hills" =/= "over the hill" PET
# 2321: "over the hilly..." =/= "over the hill" PET
# 2589: "sleep without" =/= "sleep with" PET
# 2605,2613,2615,2622: "stoutly" =/= "stout" PET

BAD_IDS = [
    2134,
    2314,2319,2435,
    2321,
    2589,
    2605,2613,2615,2622,
]


df = pd.read_csv("EN-euphs-2025.csv")

if BAD_IDS:
    df = df[~df["id"].isin(BAD_IDS)]

df = df.rename(columns={"text": "text_bounded"})

df["text_raw"] = df["text_bounded"].str.replace("[PET_BOUNDARY]", "", regex=False)

def process_pet(row):
    text = str(row["text_bounded"])
    match = re.search(r'(\w*)\[PET_BOUNDARY\](.*?)\[PET_BOUNDARY\](\w*)', text)
    if not match:
        return pd.Series({"pet_instance": "", "text_bounded": text})
    prefix, inner, suffix = match.group(1), match.group(2), match.group(3)
    if prefix:
        print(f"  split boundary (front) @ {row['id']}: [{prefix}/]{inner}")
    if suffix:
        print(f"  split boundary (back)  @ {row['id']}: {inner}[/{suffix}]")
    fixed_text = re.sub(
        r'(\w*)\[PET_BOUNDARY\](.*?)\[PET_BOUNDARY\](\w*)',
        r'[PET_BOUNDARY]\1\2\3[PET_BOUNDARY]',
        text
    )
    return pd.Series({"pet_instance": prefix + inner + suffix, "text_bounded": fixed_text})

df[["pet_instance", "text_bounded"]] = df.apply(process_pet, axis=1)

cols = ["id", "text_raw", "text_bounded", "pet", "pet_instance", "label", "category", "euph_status"]
df = df[cols]

df.to_csv("EN-euphs-2025-processed.csv", index=False)
print(f"Done. {len(df)} rows written to EN-euphs-2025-processed.csv")
