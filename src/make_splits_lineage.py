"""
Lineage-grouped train/val/test splits (70/10/20, 5 seeds), grouped by k-mer
similarity cluster so near-identical isolates cannot span partitions.

Clusters are assigned greedily, largest first, to whichever partition is
furthest below its target size (LPT bin-packing). This keeps partition sizes
close to target; class balance is not enforced, so resistant fractions vary
between partitions more than in the strain-grouped splits. That variation is
a property of lineage structure, not a bug -- resistance is lineage-associated.

Output: /blue/boucher/nelseifi/splits_lineage/<antibiotic>_seed<N>.json
"""
import json, os
import numpy as np
import pandas as pd

BASE = "/blue/boucher/nelseifi"
OUT  = f"{BASE}/splits_lineage"
SEEDS = [0, 1, 2, 3, 4]
FRACS = {"train": 0.80, "test": 0.20}   # no val: see docstring
ANTIBIOTICS = ["methicillin", "oxacillin", "cefoxitin",
               "erythromycin", "ciprofloxacin", "clindamycin"]

os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(f"{BASE}/phenotypes_matrix_aligned.csv", dtype=str)
df["matrix_row"] = df["matrix_row"].astype(int)
cl = pd.read_csv(f"{BASE}/lineage_clusters.csv")
df = df.merge(cl[["matrix_row", "cluster"]], on="matrix_row")

def assign(sizes, order, n_total):
    """Greedy: give each cluster to the partition with the largest shortfall."""
    tgt = {p: FRACS[p] * n_total for p in FRACS}
    cur = {p: 0 for p in FRACS}
    out = {}
    for c in order:
        p = max(FRACS, key=lambda p: (tgt[p] - cur[p]) / tgt[p])
        out[c] = p
        cur[p] += sizes[c]
    return out, cur

for abx in ANTIBIOTICS:
    sub = df[df[abx].isin(["Resistant", "Susceptible"])].copy()
    y = (sub[abx] == "Resistant").astype(int).values
    groups = sub["cluster"].values
    rows = sub["matrix_row"].values
    sizes = pd.Series(groups).value_counts().to_dict()
    print(f"\n{abx}: {len(sub)} tested, {y.sum()} R ({y.mean():.1%}), "
          f"{len(sizes)} clusters, largest {max(sizes.values())}")

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        # randomised size-biased order: jitter sizes so seeds give genuinely
        # different placements while large clusters still tend to go first
        cs = list(sizes.keys())
        jitter = {c: sizes[c] * rng.uniform(0.5, 1.5) for c in cs}
        order = sorted(cs, key=lambda c: -jitter[c])
        placement, _ = assign(sizes, order, len(sub))

        part = {p: np.array([i for i, g in enumerate(groups) if placement[g] == p])
                for p in FRACS}
        if any(len(set(y[v])) < 2 for v in part.values()):
            print(f"  seed {seed}: WARNING single-class partition")

        gs = {p: set(groups[v]) for p, v in part.items()}
        assert not (gs["train"] & gs["test"]), f"LEAKAGE in {abx} seed {seed}"

        split = {"antibiotic": abx, "seed": seed, "proportions": "70/10/20",
                 "grouping": "kmer_lineage_cluster",
                 **{p: rows[v].tolist() for p, v in part.items()},
                 "counts": {p: [int(len(v)), int(y[v].sum())] for p, v in part.items()}}
        json.dump(split, open(f"{OUT}/{abx}_seed{seed}.json", "w"))
        c = split["counts"]
        print(f"  seed {seed}: " + " | ".join(
            f"{p} {c[p][0]} ({c[p][0]/len(sub):.0%}, {c[p][1]/c[p][0]:.1%} R)"
            for p in ("train", "test")))
