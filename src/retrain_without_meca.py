"""
Retrain clindamycin with mecA excluded, to test whether removing a confounded
feature improves external generalisation.

The clindamycin model's top features are mecA loci. In the training population
mecA is strongly associated with clindamycin resistance (OR 16.8) purely through
co-carriage, but in the external population that association is gone (OR 1.5) --
which is why the model fails there while every other model generalises.

If the confounder is the problem, forcing the model off it should trade internal
accuracy for external accuracy. Two exclusion widths are tested, because
removing one gene may leave the rest of the cassette serving the same proxy role:

  narrow  the k-mers of the assembled mecA block itself
  broad   every k-mer whose presence correlates with mecA carriage above PHI,
          i.e. the whole linked region, however far it extends

Erythromycin is refit the same way as a control: its determinant is erm(C), not
mecA, so excluding mecA should barely move it. If erythromycin drops as much as
clindamycin, the exclusion is removing something other than the confounder.

Usage:  python3 retrain_without_meca.py [phi=0.7] [carrier_frac=0.5]
Output: /blue/boucher/nelseifi/models/<abx>_nomecA_{narrow,broad}/
        (same model.json / kmers.txt / meta.json layout as export_model.py)
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix
import xgboost as xgb

BASE = "/blue/boucher/nelseifi"
MAT = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
N_FEATURES = 11_351_471
K = 31
PHI = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
COMP = str.maketrans("ACGT", "TGCA")

TOP_K = {"clindamycin": 200_000, "erythromycin": 50_000}

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
n_train = len(indptr) - 1

# ------------------------------------------------- the mecA block, as k-mers
meca = set()
with open(f"{BASE}/interpret2/contigs_all.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["antibiotic"] == "methicillin" and r["contig"] == "1":
            s = r["sequence"]
            for i in range(len(s) - K + 1):
                km = s[i:i + K]
                meca.add(km)
                meca.add(km[::-1].translate(COMP))
print(f"mecA block: {len(meca)//2} k-mers", flush=True)

meca_cols = []
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        if s in meca:
            meca_cols.append(int(i_str))
meca_cols = np.array(sorted(meca_cols), dtype=np.int64)
print(f"located {len(meca_cols)} matrix columns", flush=True)

# ---------------------------------------------- mecA carriage per genome
lut = np.full(N_FEATURES, -1, dtype=np.int32)
lut[meca_cols] = np.arange(len(meca_cols), dtype=np.int32)
blk = np.zeros((n_train, len(meca_cols)), dtype=bool)
for r in range(n_train):
    j = lut[np.asarray(cols[indptr[r]:indptr[r + 1]])]
    blk[r, j[j >= 0]] = True
carrier = blk.mean(axis=1) >= FRAC
M, N = int(carrier.sum()), int((~carrier).sum())
print(f"mecA carriers: {M}/{n_train}\n", flush=True)


def counts(rows):
    acc = np.zeros(N_FEATURES, dtype=np.float64)
    for i in range(0, len(rows), 400):
        c = np.concatenate([np.asarray(cols[indptr[r]:indptr[r + 1]])
                            for r in rows[i:i + 400]])
        acc += np.bincount(c, minlength=N_FEATURES)
    return acc


# ------------------------- phi correlation of every k-mer with mecA carriage
n11 = counts(np.where(carrier)[0])
n10 = counts(np.where(~carrier)[0])
n01, n00 = M - n11, N - n10
den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
with np.errstate(divide="ignore", invalid="ignore"):
    phi = np.where(den > 0, (n11 * n00 - n10 * n01) / den, 0.0)
phi[~np.isfinite(phi)] = 0.0

excl = {"narrow": np.zeros(N_FEATURES, dtype=bool),
        "broad": np.abs(phi) > PHI}
excl["narrow"][meca_cols] = True
excl["broad"][meca_cols] = True          # broad always contains narrow
for k, v in excl.items():
    print(f"exclusion '{k}': {int(v.sum()):,} k-mers", flush=True)


def submatrix(rows, keep):
    l2 = np.full(N_FEATURES, -1, dtype=np.int32)
    l2[keep] = np.arange(len(keep), dtype=np.int32)
    data, ind, ptr = [], [], [0]
    for r in rows:
        c = np.asarray(cols[indptr[r]:indptr[r + 1]])
        mk = l2[c] >= 0
        ind.append(l2[c[mk]])
        data.append(np.ones(int(mk.sum()), dtype=np.float32))
        ptr.append(ptr[-1] + int(mk.sum()))
    return csr_matrix((np.concatenate(data), np.concatenate(ind),
                       np.array(ptr)), shape=(len(rows), len(keep)))


# ----------------------------------------------- refit with each exclusion
jobs = []
for abx, k_top in TOP_K.items():
    labels = np.load(f"{BASE}/labels_{abx}.npy")
    rows = np.where(labels >= 0)[0]
    y = labels[rows]
    oR, oS = counts(rows[y == 1]), counts(rows[y == 0])
    tot, pR = oR + oS, float((y == 1).mean())
    eR, eS = pR * tot, (1 - pR) * tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = (oR - eR) ** 2 / eR + (oS - eS) ** 2 / eS
    chi2[~np.isfinite(chi2)] = 0.0

    for tag, mask in excl.items():
        c2 = chi2.copy()
        c2[mask] = 0.0                                   # drop the confounder
        keep = np.argpartition(c2, -k_top)[-k_top:]
        keep = np.sort(keep[c2[keep] > 0])
        jobs.append((f"{abx}_nomecA_{tag}", abx, rows, y, keep, pR))
        print(f"{abx:14s} {tag:7s} kept {len(keep):,} features "
              f"(top chi2 {c2[keep].max():.1f})", flush=True)

want = {int(i) for _, _, _, _, keep, _ in jobs for i in keep}
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
print(f"\nresolved {len(seqs):,} k-mer sequences\n", flush=True)

for name, abx, rows, y, keep, pR in jobs:
    X = submatrix(rows, keep)
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", n_jobs=4,
                            random_state=0, tree_method="hist")
    clf.fit(X, y)
    d = f"{BASE}/models/{name}"
    os.makedirs(d, exist_ok=True)
    clf.get_booster().save_model(f"{d}/model.json")
    with open(f"{d}/kmers.txt", "w") as fh:
        for i in keep:
            fh.write(seqs[int(i)] + "\n")
    json.dump({"antibiotic": abx, "variant": name, "top_k": int(len(keep)),
               "n_genomes": int(len(rows)), "pct_resistant": pR, "k": 31,
               "excluded": "mecA " + name.rsplit("_", 1)[1],
               "phi_threshold": PHI},
              open(f"{d}/meta.json", "w"), indent=2)
    print(f"wrote {d}", flush=True)
