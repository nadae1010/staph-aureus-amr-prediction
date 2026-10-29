"""
Does the standard 2x2 chi-square rank the causal determinant above the linked
marker? Stage 1 of the 2x2 experiment -- a diagnostic, before any retraining.

The pipeline ranks features with a chi-square that conditions on feature-positive
samples: for each k-mer it looks only at the genomes carrying it and asks whether
the resistant/susceptible split among carriers differs from the population split.
Genomes lacking the k-mer never enter the statistic.

That rewards coverage. For clindamycin, mecA and erm(C) make their carriers
equally likely to be resistant -- 683/894 and 245/322, both 76% -- but mecA has
2.8x more carriers and scores 2.8x higher (334 vs 118). mecA has no mechanistic
role in clindamycin resistance; erm(C) is the determinant.

The standard 2x2 chi-square builds the full contingency table, counting
non-carriers as well:

                 Resistant   Susceptible
    carries          a            b
    does not         c            d

    chi2 = n(ad - bc)^2 / ((a+b)(c+d)(a+c)(b+d))

Because b is now weighed against the whole susceptible population rather than
diluted by carrier count, a feature present in 21.6% of susceptible isolates
(mecA) is penalised where one present in 7.9% (erm(C)) is not.

This computes both statistics over all labelled genomes for each antibiotic and
reports where the known loci land under each, plus how much the selected feature
sets would differ. If erm(C) does not outrank mecA for clindamycin under 2x2,
the premise of the experiment is wrong and nothing further should be run.

Usage:  python3 chi2_compare.py [top_k=200000] [abx ...]
Output: printed report, plus /blue/boucher/nelseifi/chi2/compare_<abx>.tsv
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
COMP = str.maketrans("ACGT", "TGCA")

TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
ANTIBIOTICS = sys.argv[2:] or ["clindamycin", "erythromycin", "cefoxitin",
                               "oxacillin", "ciprofloxacin"]

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")


def rc(s):
    return s[::-1].translate(COMP)


# ------------------------------------------------- known loci, for locating them
LOCI = {
    "mecA (C-term)": ("methicillin", "1"),
    "mecA (N-term)": ("methicillin", "2"),
    "erm(C) plasmid": ("erythromycin", "1"),
    "gyrA/grlA region": ("ciprofloxacin", "1"),
}
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

locus_cols = defaultdict(list)
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        n = want.get(s)
        if n:
            locus_cols[n].append(int(i_str))
for n in LOCI:
    print(f"{n}: {len(locus_cols.get(n, [])):,} matrix columns", flush=True)
print(flush=True)


def counts(rs):
    acc = np.zeros(N_FEATURES, dtype=np.float64)
    for i in range(0, len(rs), 400):
        c = np.concatenate([np.asarray(cols[indptr[r]:indptr[r + 1]])
                            for r in rs[i:i + 400]])
        acc += np.bincount(c, minlength=N_FEATURES)
        del c
    return acc


for abx in ANTIBIOTICS:
    labels = np.load(f"{BASE}/labels_{abx}.npy")
    rows = np.where(labels >= 0)[0]
    y = labels[rows]
    nR, nS = float((y == 1).sum()), float((y == 0).sum())
    n = nR + nS
    print(f"{'='*74}\n{abx}  n={int(n)}  R={int(nR)}  S={int(nS)}", flush=True)

    a = counts(rows[y == 1])          # carriers that are resistant
    b = counts(rows[y == 0])          # carriers that are susceptible
    c = nR - a                        # non-carriers, resistant
    d = nS - b                        # non-carriers, susceptible

    # arm A: conditioned on carriers only
    tot = a + b
    pR = nR / n
    eR, eS = pR * tot, (1 - pR) * tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chiA = (a - eR) ** 2 / eR + (b - eS) ** 2 / eS
    chiA[~np.isfinite(chiA)] = 0.0

    # arm B: the full 2x2 table
    den = (a + b) * (c + d) * nR * nS
    with np.errstate(divide="ignore", invalid="ignore"):
        chiB = np.where(den > 0, n * (a * d - b * c) ** 2 / den, 0.0)
    chiB[~np.isfinite(chiB)] = 0.0

    rankA = np.argsort(np.argsort(-chiA))
    rankB = np.argsort(np.argsort(-chiB))

    print(f"\n  {'locus':18s} {'chi2 A':>9} {'rank A':>9} "
          f"{'chi2 B':>9} {'rank B':>9}   {'R cov':>6} {'S cov':>6}")
    for name in LOCI:
        idx = np.array(locus_cols.get(name, []), dtype=np.int64)
        if len(idx) == 0:
            continue
        bestA = idx[np.argmax(chiA[idx])]
        bestB = idx[np.argmax(chiB[idx])]
        print(f"  {name:18s} {chiA[bestA]:>9.1f} {int(rankA[bestA]):>9,} "
              f"{chiB[bestB]:>9.1f} {int(rankB[bestB]):>9,}   "
              f"{a[bestB]/nR:>5.1%} {b[bestB]/nS:>5.1%}")

    keepA = set(np.argpartition(chiA, -TOP_K)[-TOP_K:].tolist())
    keepB = set(np.argpartition(chiB, -TOP_K)[-TOP_K:].tolist())
    shared = len(keepA & keepB)
    print(f"\n  top {TOP_K:,} feature sets share {shared:,} "
          f"({100*shared/TOP_K:.1f}%); {TOP_K-shared:,} differ", flush=True)

    # what each criterion puts first
    for tag, chi, rank in (("A (carrier-conditioned)", chiA, rankA),
                           ("B (standard 2x2)", chiB, rankB)):
        top = np.argsort(-chi)[:5]
        print(f"\n  top 5 under {tag}:")
        for i, col in enumerate(top):
            lab = next((nm for nm, cc in locus_cols.items() if col in set(cc)),
                       "unidentified")
            print(f"    {i+1}. chi2 {chi[col]:>9.1f}  "
                  f"R {a[col]/nR:>5.1%}  S {b[col]/nS:>5.1%}  {lab}")

    with open(f"{BASE}/chi2/compare_{abx}.tsv", "w") as fh:
        fh.write("locus\tchi2_A\trank_A\tchi2_B\trank_B\tR_cov\tS_cov\n")
        for name in LOCI:
            idx = np.array(locus_cols.get(name, []), dtype=np.int64)
            if len(idx) == 0:
                continue
            bA, bB = idx[np.argmax(chiA[idx])], idx[np.argmax(chiB[idx])]
            fh.write(f"{name}\t{chiA[bA]:.1f}\t{int(rankA[bA])}\t"
                     f"{chiB[bB]:.1f}\t{int(rankB[bB])}\t"
                     f"{a[bB]/nR:.4f}\t{b[bB]/nS:.4f}\n")
    print(f"\n  wrote {BASE}/chi2/compare_{abx}.tsv\n", flush=True)
