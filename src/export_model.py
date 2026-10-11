"""
Train a final model per antibiotic on all available BV-BRC genomes and export
it in a form that can be applied to genomes outside this dataset.

The pipeline's features are column indices into a matrix built from these 4,610
genomes, so they mean nothing for a new assembly. This script exports the
selected features as k-mer SEQUENCES instead, alongside the fitted classifier.
Scoring an external genome then reduces to presence/absence matching of those
k-mers against the assembly -- no KMX run and no matrix rebuild.

Feature counts are per antibiotic, taken from the top-K sensitivity sweep:
single-determinant mechanisms saturate early, MLSb mechanisms do not.

Chi-square and the classifier are fit on ALL labelled genomes. There is no
held-out split here by design -- the external dataset is the test set.

Usage:  python3 export_model.py [antibiotic ...]    (default: all six)
Output: /blue/boucher/nelseifi/models/<abx>/
          model.json      XGBoost booster
          kmers.txt       one k-mer sequence per line, in feature order
          meta.json       top_k, n_train, class balance, feature count
"""
import json
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
import xgboost as xgb

BASE = "/blue/boucher/nelseifi"
MAT = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
OUT = f"{BASE}/models"
N_FEATURES = 11_351_471

# per-antibiotic feature budget from the top-K sweep
TOP_K = {
    "methicillin":   5_000,
    "cefoxitin":     5_000,
    "ciprofloxacin": 5_000,
    "oxacillin":    50_000,
    "erythromycin": 50_000,
    "clindamycin": 200_000,
}
ANTIBIOTICS = sys.argv[1:] or list(TOP_K)

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")


def class_counts(rows):
    acc = np.zeros(N_FEATURES, dtype=np.float64)
    for i in range(0, len(rows), 400):
        c = np.concatenate([np.asarray(cols[indptr[r]:indptr[r + 1]])
                            for r in rows[i:i + 400]])
        acc += np.bincount(c, minlength=N_FEATURES)
    return acc


def submatrix(rows, keep):
    lut = np.full(N_FEATURES, -1, dtype=np.int32)
    lut[keep] = np.arange(len(keep), dtype=np.int32)
    data, ind, ptr = [], [], [0]
    for r in rows:
        c = np.asarray(cols[indptr[r]:indptr[r + 1]])
        mk = lut[c] >= 0
        ind.append(lut[c[mk]])
        data.append(np.ones(int(mk.sum()), dtype=np.float32))
        ptr.append(ptr[-1] + int(mk.sum()))
    return csr_matrix((np.concatenate(data), np.concatenate(ind),
                       np.array(ptr)), shape=(len(rows), len(keep)))


# ------------------------------------------------- select features per drug
selected = {}
for abx in ANTIBIOTICS:
    t0 = time.time()
    labels = np.load(f"{BASE}/labels_{abx}.npy")
    rows = np.where(labels >= 0)[0]
    y = labels[rows]

    oR, oS = class_counts(rows[y == 1]), class_counts(rows[y == 0])
    tot, pR = oR + oS, float((y == 1).mean())
    eR, eS = pR * tot, (1 - pR) * tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = (oR - eR) ** 2 / eR + (oS - eS) ** 2 / eS
    chi2[~np.isfinite(chi2)] = 0.0

    k = min(TOP_K[abx], N_FEATURES)
    keep = np.argpartition(chi2, -k)[-k:]
    keep = keep[np.argsort(chi2[keep])[::-1]]        # rank order, best first
    selected[abx] = (rows, y, keep, pR)
    print(f"{abx:14s} {len(rows):5d} genomes ({pR:.1%} R)  "
          f"top_k={k:,}  {time.time()-t0:.0f}s", flush=True)

# ------------------------------------- k-mer sequences, one pass over index
want = {int(i) for _, _, keep, _ in selected.values() for i in keep}
seqs = {}
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        i = int(i_str)
        if i in want:
            seqs[i] = s
            if len(seqs) == len(want):
                break
print(f"\nresolved {len(seqs):,}/{len(want):,} k-mer sequences", flush=True)

# ------------------------------------------------------- fit and write out
for abx, (rows, y, keep, pR) in selected.items():
    t0 = time.time()
    X = submatrix(rows, np.sort(keep))               # columns ascending
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", n_jobs=4,
                            random_state=0, tree_method="hist")
    clf.fit(X, y)

    d = f"{OUT}/{abx}"
    os.makedirs(d, exist_ok=True)
    clf.get_booster().save_model(f"{d}/model.json")
    with open(f"{d}/kmers.txt", "w") as fh:
        for i in np.sort(keep):                      # SAME order as X columns
            fh.write(seqs[int(i)] + "\n")
    json.dump({"antibiotic": abx, "top_k": int(len(keep)),
               "n_genomes": int(len(rows)), "pct_resistant": pR,
               "k": 31, "features_are": "presence/absence, column order "
                                        "matches kmers.txt line order"},
              open(f"{d}/meta.json", "w"), indent=2)
    print(f"{abx:14s} wrote {d}  ({time.time()-t0:.0f}s)", flush=True)
