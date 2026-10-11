"""
Measure whether the mecA-clindamycin association differs between the training
and external datasets.

The clindamycin model's top-ranked features are mecA loci rather than an MLSb
determinant, so it predicts clindamycin resistance through linkage to SCCmec.
Linkage is a property of a population, not of the organism, so a population
with a different MRSA composition should break it -- which is the proposed
explanation for clindamycin being the one model that fails externally.

This quantifies it. Two loci are tracked, located by matching the assembled
contigs from interpret2 against each model's k-mer list:
  mecA block     methicillin contig 1  (2,829 bp)
  erm(C) plasmid erythromycin contig 1 (2,503 bp, circular)

Carriage is scored the same way in both datasets (at least half the locus's
k-mers present), then cross-tabulated against phenotype.

Usage:  python3 linkage_external.py [carrier_frac=0.5]
Output: printed tables
"""
import csv
import sys
from collections import defaultdict

import numpy as np

BASE = "/blue/boucher/nelseifi"
MAT = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
N_FEATURES = 11_351_471
K = 31
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
COMP = str.maketrans("ACGT", "TGCA")

LOCI = {"mecA block": ("methicillin", "1"),
        "erm(C) plasmid": ("erythromycin", "1")}

# ------------------------------------------------- locus sequences -> k-mers
loci_kmers = {}
with open(f"{BASE}/interpret2/contigs_all.tsv") as f:
    rd = csv.DictReader(f, delimiter="\t")
    for r in rd:
        for name, (abx, cn) in LOCI.items():
            if r["antibiotic"] == abx and r["contig"] == cn:
                seq = r["sequence"]
                s = set()
                for i in range(len(seq) - K + 1):
                    km = seq[i:i + K]
                    s.add(km)
                    s.add(km[::-1].translate(COMP))
                loci_kmers[name] = s
                print(f"{name}: {len(seq)} bp -> {len(s)//2} k-mers")

# --------------------------------- locate each locus inside the external npz
d = np.load(f"{BASE}/external/external_features.npz", allow_pickle=True)
ext = d["ext"]
accs = [str(a) for a in d["accessions"]]
ext_cols = {}
for name, kms in loci_kmers.items():
    found = {}
    for abx in [k[len("model_"):] for k in d.files if k.startswith("model_")]:
        mc = d[f"model_{abx}"]
        for j, line in enumerate(open(f"{BASE}/models/{abx}/kmers.txt")):
            s = line.strip()
            if s and s in kms:
                found[int(mc[j])] = True
    ext_cols[name] = np.array(sorted(found), dtype=np.int64)
    print(f"  {name}: {len(ext_cols[name])} union columns located")

# ------------------------------------------- the same loci in the training matrix
indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
n_train = len(indptr) - 1

want = {s: n for n, kms in loci_kmers.items() for s in kms}
train_cols = defaultdict(list)
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        n = want.get(s)
        if n:
            train_cols[n].append(int(i_str))
for n in loci_kmers:
    print(f"  {n}: {len(train_cols[n])} training matrix columns located")

allc = sorted({c for v in train_cols.values() for c in v})
lut = np.full(N_FEATURES, -1, dtype=np.int32)
lut[np.array(allc, dtype=np.int64)] = np.arange(len(allc), dtype=np.int32)
tp = np.zeros((n_train, len(allc)), dtype=bool)
for r in range(n_train):
    j = lut[np.asarray(cols[indptr[r]:indptr[r + 1]])]
    tp[r, j[j >= 0]] = True
tpos = {c: j for j, c in enumerate(allc)}

# --------------------------------------------------------------- phenotypes
ext_ph = defaultdict(dict)
with open(f"{BASE}/external_phenotypes.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        ext_ph[r["antibiotic"]][r["assembly"].split(".")[0]] = r["phenotype"]
idx_of = {a.split(".")[0]: i for i, a in enumerate(accs)}


def table(name, carrier, labels, rows, tag):
    R = labels[rows] == 1
    S = labels[rows] == 0
    c = carrier[rows]
    if R.sum() == 0 or S.sum() == 0:
        print(f"    {tag:10s} single-class, skipped")
        return
    print(f"    {tag:10s} carried by {100*c[R].mean():5.1f}% of resistant "
          f"({int(c[R].sum())}/{int(R.sum())}), "
          f"{100*c[S].mean():5.1f}% of susceptible "
          f"({int(c[S].sum())}/{int(S.sum())})")


for name in loci_kmers:
    print(f"\n=== {name}")
    # training side
    tc = np.array([tpos[c] for c in train_cols[name]])
    train_carrier = tp[:, tc].mean(axis=1) >= FRAC
    # external side
    ext_carrier = ext[:, ext_cols[name]].mean(axis=1) >= FRAC
    print(f"  carriage: training {int(train_carrier.sum())}/{n_train}, "
          f"external {int(ext_carrier.sum())}/{len(accs)}")

    for abx in ("clindamycin", "erythromycin", "cefoxitin", "oxacillin"):
        print(f"  {abx}:")
        lab = np.load(f"{BASE}/labels_{abx}.npy")
        table(name, train_carrier, lab, np.where(lab >= 0)[0], "training")

        rows, ys = [], []
        for acc, p in ext_ph.get(abx, {}).items():
            i = idx_of.get(acc)
            if i is not None and p in ("Resistant", "Susceptible"):
                rows.append(i)
                ys.append(1 if p == "Resistant" else 0)
        if rows:
            c, yy = ext_carrier[np.array(rows)], np.array(ys)
            if yy.sum() and (yy == 0).sum():
                print(f"    {'external':10s} carried by "
                      f"{100*c[yy==1].mean():5.1f}% of resistant "
                      f"({int(c[yy==1].sum())}/{int((yy==1).sum())}), "
                      f"{100*c[yy==0].mean():5.1f}% of susceptible "
                      f"({int(c[yy==0].sum())}/{int((yy==0).sum())})")
            else:
                print(f"    {'external':10s} single-class, skipped")
