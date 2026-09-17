"""
Build leakage-controlled stratified train/val/test splits for each antibiotic.
70/10/20, 5 repeats with fixed seeds (0-4). Groups = Strain Name, so all
genomes sharing a strain name stay in the same partition.

Input:  /blue/boucher/nelseifi/phenotypes_matrix_aligned.csv
        (phenotype rows reordered to match KMX matrix row order)
Output: /blue/boucher/nelseifi/splits/<antibiotic>_seed<N>.json
        with train/val/test matrix-row indices and class counts

Run: python3 make_splits.py   (CPU only, runs on a login node in seconds)
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ALIGNED = "/blue/boucher/nelseifi/phenotypes_matrix_aligned.csv"
OUTDIR  = "/blue/boucher/nelseifi/splits"
SEEDS   = [0, 1, 2, 3, 4]
ANTIBIOTICS = ["methicillin", "oxacillin", "cefoxitin",
               "erythromycin", "ciprofloxacin", "clindamycin"]

os.makedirs(OUTDIR, exist_ok=True)
df = pd.read_csv(ALIGNED, dtype=str)
df["matrix_row"] = df["matrix_row"].astype(int)

for abx in ANTIBIOTICS:
    sub = df[df[abx].isin(["Resistant", "Susceptible"])].copy()
    y = (sub[abx] == "Resistant").astype(int).values
    groups = sub["Strain Name"].fillna("NA_" + sub["Genome ID"]).values
    rows = sub["matrix_row"].values
    idx = np.arange(len(sub))

    print(f"\n{abx}: {len(sub)} tested, {y.sum()} R ({y.mean():.1%})")

    for seed in SEEDS:
        # 20% test
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        trval_i, test_i = next(sgkf.split(idx, y, groups))

        # 10% of total = 1/8 of the remaining 80% -> val
        sgkf2 = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed)
        tr_rel, val_rel = next(sgkf2.split(trval_i, y[trval_i], groups[trval_i]))
        train_i, val_i = trval_i[tr_rel], trval_i[val_rel]

        # leakage assertion: no strain appears in more than one partition
        gs = [set(groups[p]) for p in (train_i, val_i, test_i)]
        assert not (gs[0] & gs[1]) and not (gs[0] & gs[2]) and not (gs[1] & gs[2]), \
            f"LEAKAGE in {abx} seed {seed}"

        split = {
            "antibiotic": abx, "seed": seed, "proportions": "70/10/20",
            "train": rows[train_i].tolist(),
            "val":   rows[val_i].tolist(),
            "test":  rows[test_i].tolist(),
            "counts": {
                "train": [int(len(train_i)), int(y[train_i].sum())],
                "val":   [int(len(val_i)),   int(y[val_i].sum())],
                "test":  [int(len(test_i)),  int(y[test_i].sum())],
            },
        }
        with open(f"{OUTDIR}/{abx}_seed{seed}.json", "w") as f:
            json.dump(split, f)

        c = split["counts"]
        print(f"  seed {seed}: train {c['train'][0]} ({c['train'][1]/c['train'][0]:.1%} R) "
              f"| val {c['val'][0]} ({c['val'][1]/c['val'][0]:.1%} R) "
              f"| test {c['test'][0]} ({c['test'][1]/c['test'][0]:.1%} R)")
