"""
Apply an exported model to genomes outside the BV-BRC training set.

Features are presence/absence of the model's selected k-mers, matched directly
against each assembly, so no KMX run or matrix rebuild is needed. KMX stores
canonical k-mers, so both orientations are accepted.

Input manifest: tab-separated, no header, two columns --
    /path/to/assembly.fasta <TAB> Resistant|Susceptible
Rows with any other phenotype value are skipped and counted.

IMPORTANT: deduplicate against the training set BEFORE running this. A genome
that is also in the 4,610 is not external, and will inflate every metric.

Usage:  python3 score_external.py <antibiotic> <manifest.tsv> [out.json]
Output: printed metrics, plus per-genome predictions as JSON if out.json given
"""
import json
import sys

import numpy as np
import xgboost as xgb
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)

BASE = "/blue/boucher/nelseifi"
K = 31
COMP = str.maketrans("ACGT", "TGCA")

abx = sys.argv[1]
manifest = sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) > 3 else None
mdir = f"{BASE}/models/{abx}"

meta = json.load(open(f"{mdir}/meta.json"))
kmers = [l.strip() for l in open(f"{mdir}/kmers.txt") if l.strip()]
print(f"{abx}: {len(kmers):,} model k-mers "
      f"(trained on {meta['n_genomes']} genomes, {meta['pct_resistant']:.1%} R)")

# both orientations -> same feature column
lookup = {}
for j, km in enumerate(kmers):
    lookup[km] = j
    lookup[km[::-1].translate(COMP)] = j


def contigs(path):
    buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if buf:
                    yield "".join(buf).upper()
                buf = []
            elif line:
                buf.append(line)
    if buf:
        yield "".join(buf).upper()


def featurize(path):
    v = np.zeros(len(kmers), dtype=np.float32)
    hits = 0
    for seq in contigs(path):
        for i in range(len(seq) - K + 1):
            j = lookup.get(seq[i:i + K])
            if j is not None and v[j] == 0:
                v[j] = 1
                hits += 1
    return v, hits


paths, ys, skipped = [], [], 0
for line in open(manifest):
    line = line.rstrip("\n")
    if not line.strip():
        continue
    p, _, pheno = line.partition("\t")
    pheno = pheno.strip()
    if pheno == "Resistant":
        ys.append(1)
    elif pheno == "Susceptible":
        ys.append(0)
    else:
        skipped += 1
        continue
    paths.append(p.strip())
print(f"manifest: {len(paths)} genomes usable, {skipped} skipped, "
      f"{sum(ys)} resistant ({np.mean(ys):.1%})\n")

X = np.zeros((len(paths), len(kmers)), dtype=np.float32)
for n, p in enumerate(paths):
    X[n], hits = featurize(p)
    if n < 5 or n % 100 == 0:
        print(f"  [{n+1}/{len(paths)}] {p.split('/')[-1][:45]:45s} "
              f"{hits:,} of {len(kmers):,} model k-mers present", flush=True)

booster = xgb.Booster()
booster.load_model(f"{mdir}/model.json")
prob = booster.predict(xgb.DMatrix(X))
pred = (prob >= 0.5).astype(int)
y = np.array(ys)

print(f"\n=== {abx} external validation (n={len(y)})")
print(f"  accuracy   {np.mean(pred == y):.3f}")
if len(set(ys)) > 1:
    print(f"  precision  {precision_score(y, pred, zero_division=0):.3f}")
    print(f"  recall     {recall_score(y, pred, zero_division=0):.3f}")
    print(f"  f1         {f1_score(y, pred, zero_division=0):.3f}")
    print(f"  auroc      {roc_auc_score(y, prob):.3f}")
    print(f"  auprc      {average_precision_score(y, prob):.3f}")
    print(f"  confusion  {confusion_matrix(y, pred).tolist()}  [[TN,FP],[FN,TP]]")
else:
    print("  single-class external set: only accuracy is meaningful")

# coverage sanity check -- a genome matching almost no model k-mers is suspect
cov = X.mean(axis=1)
print(f"\n  model-k-mer coverage per genome: median {np.median(cov):.3f}, "
      f"min {cov.min():.3f}, max {cov.max():.3f}")

if out_path:
    json.dump({"antibiotic": abx, "n": len(y),
               "genomes": [{"path": p, "label": int(t), "prob": float(q)}
                           for p, t, q in zip(paths, ys, prob)]},
              open(out_path, "w"), indent=2)
    print(f"\n  wrote {out_path}")
