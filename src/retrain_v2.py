"""
Exclusion done properly: remove what the beta-lactam models use, not just mecA.

retrain_without_meca.py excluded the mecA block and reported +0.0168 external
AUROC for clindamycin. That result stands, but its interpretation does not.
Ranked by chi-square, clindamycin's strongest feature is not mecA: it is a
258 bp element with no mecA k-mers, matching contig 2 of the methicillin and
oxacillin models, carried by 68.7% of resistant and 12.8% of susceptible
isolates (chi2 364.3, rank 1). mecA's best fragment is rank 39. The earlier
experiment removed the second confounder and left the first in place.

The fix is to stop naming loci and use a rule that does not depend on knowing
which one matters: exclude everything the beta-lactam models rely on. Those
models predict a different drug class, so any clindamycin signal shared with
them is linkage rather than mechanism. Two widths again, but anchored properly:

  loci     every k-mer among the top features of methicillin, oxacillin and
           cefoxitin -- the beta-lactam feature set entire
  phi      that, plus every k-mer correlating above PHI with carriage of
           either mecA or the 258 bp element, which catches sequence that
           travels with the cassette without being selected for it

Erythromycin is refit identically as the control. Its determinant is erm(C),
so it should lose little; if it falls as far as clindamycin, the exclusion is
removing something other than linkage.

Usage:  python3 retrain_v2.py [phi=0.5] [bl_top=50000] [carrier_frac=0.5]
Output: /blue/boucher/nelseifi/models/<abx>_v2_{loci,phi}/
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
COMP = str.maketrans("ACGT", "TGCA")

PHI = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
BL_TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

BETA_LACTAMS = ["methicillin", "oxacillin", "cefoxitin"]
TOP_K = {"clindamycin": 200_000, "erythromycin": 50_000}

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
n_train = len(indptr) - 1


def rc(s):
    return s[::-1].translate(COMP)


def counts(rs):
    acc = np.zeros(N_FEATURES, dtype=np.float64)
    for i in range(0, len(rs), 400):
        c = np.concatenate([np.asarray(cols[indptr[r]:indptr[r + 1]])
                            for r in rs[i:i + 400]])
        acc += np.bincount(c, minlength=N_FEATURES)
        del c
    return acc


# ------------------------- what the beta-lactam models use, from saved rankings
bl = np.zeros(N_FEATURES, dtype=bool)
for abx in BETA_LACTAMS:
    d = np.load(f"{BASE}/chi2/{abx}_seed0_chi2top.npz")
    idx, score = d["idx"], d["score"]
    take = idx[np.argsort(-score)[:BL_TOP]]
    bl[take.astype(np.int64)] = True
    print(f"{abx:14s} top {BL_TOP:,} features added", flush=True)
print(f"beta-lactam feature set: {int(bl.sum()):,} k-mers\n", flush=True)

# ------------------------------- the two loci whose carriage anchors the phi cut
LOCI = {"mecA": ("methicillin", "1"), "element258": ("methicillin", "2")}
loci_cols = {}
want = {}
with open(f"{BASE}/interpret2/contigs_all.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        for name, (a, c) in LOCI.items():
            if r["antibiotic"] == a and r["contig"] == c:
                s = r["sequence"]
                for i in range(len(s) - K + 1):
                    km = s[i:i + K]
                    want[km] = name
                    want[rc(km)] = name
                print(f"{name}: {len(s)} bp")
hits = {n: [] for n in LOCI}
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        n = want.get(s)
        if n:
            hits[n].append(int(i_str))
for n in LOCI:
    loci_cols[n] = np.array(sorted(hits[n]), dtype=np.int64)
    print(f"  {n}: {len(loci_cols[n]):,} matrix columns", flush=True)

allc = np.array(sorted({c for v in loci_cols.values() for c in v}),
                dtype=np.int64)
lut = np.full(N_FEATURES, -1, dtype=np.int32)
lut[allc] = np.arange(len(allc), dtype=np.int32)
pres = np.zeros((n_train, len(allc)), dtype=bool)
for r in range(n_train):
    j = lut[np.asarray(cols[indptr[r]:indptr[r + 1]])]
    pres[r, j[j >= 0]] = True
pos = {c: j for j, c in enumerate(allc)}

carriers = {}
for n, cc in loci_cols.items():
    sub = pres[:, np.array([pos[c] for c in cc])]
    carriers[n] = sub.mean(axis=1) >= FRAC
    print(f"  {n} carriage: {int(carriers[n].sum()):,}/{n_train}", flush=True)

# ----------------------------------------- phi of every k-mer against each locus
phi_mask = np.zeros(N_FEATURES, dtype=bool)
for n, carr in carriers.items():
    M, Nn = int(carr.sum()), int((~carr).sum())
    n11 = counts(np.where(carr)[0])
    n10 = counts(np.where(~carr)[0])
    n01, n00 = M - n11, Nn - n10
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(den > 0, (n11 * n00 - n10 * n01) / den, 0.0)
    phi[~np.isfinite(phi)] = 0.0
    add = np.abs(phi) > PHI
    phi_mask |= add
    print(f"  |phi|>{PHI} vs {n}: {int(add.sum()):,} k-mers", flush=True)

excl = {"loci": bl.copy(), "phi": bl | phi_mask}
for n, cc in loci_cols.items():
    for m in excl.values():
        m[cc] = True
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


# -------------------------------------------------------------- refit and export
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
    print(f"\n{abx}: unexcluded top chi2 {chi2.max():.1f}", flush=True)

    for tag, mask in excl.items():
        c2 = chi2.copy()
        c2[mask] = 0.0
        keep = np.argpartition(c2, -k_top)[-k_top:]
        keep = np.sort(keep[c2[keep] > 0])
        jobs.append((f"{abx}_v2_{tag}", abx, rows, y, keep, pR))
        print(f"  {tag:5s} kept {len(keep):,} (top chi2 {c2[keep].max():.1f})",
              flush=True)

need = {int(i) for _, _, _, _, keep, _ in jobs for i in keep}
seqs = {}
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        i = int(i_str)
        if i in need:
            seqs[i] = s
            if len(seqs) == len(need):
                break
print(f"\nresolved {len(seqs):,} sequences\n", flush=True)

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
               "excluded": name.rsplit("_", 1)[1], "phi_threshold": PHI,
               "bl_top": BL_TOP},
              open(f"{d}/meta.json", "w"), indent=2)
    print(f"wrote {d}", flush=True)
