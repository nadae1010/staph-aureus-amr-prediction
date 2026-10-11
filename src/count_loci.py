"""
Count how many INDEPENDENT loci each model's top features actually represent.

The chi-square score curve for clindamycin is nearly flat -- rank 1 scores 270.3
and rank 1,000 scores 252.0 -- and the top scores arrive in exact ties (270.3
four times, 269.2 four times). An exact tie means two k-mers have identical
presence across every genome, which happens when they tile the same locus. So a
model holding 200,000 "features" may be standing on a handful of genomic blocks,
each contributing hundreds of redundant columns.

That matters for generalisation. A model resting on three linked blocks carries
three bets, however many columns it reports, and if one of those blocks is a
population-specific linkage marker rather than a determinant, the model breaks
in a population where the linkage differs -- which is what clindamycin does.

This collapses each model's top features into distinct presence patterns and
reports, per pattern: how many k-mers share it, the implied locus length, and
its association with phenotype. The comparison across antibiotics is the point.
Clindamycin fails externally (AUROC 0.758) while erythromycin does not (0.961);
if clindamycin rests on far fewer independent patterns, that is a cleaner
explanation than the mecA exclusion test produced.

Usage:  python3 count_loci.py [top_n=10000] [abx ...]
Output: printed report, plus /blue/boucher/nelseifi/interpret2/loci_<abx>.tsv
"""
import os
import sys

import numpy as np

BASE = "/blue/boucher/nelseifi"
MAT = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
N_FEATURES = 11_351_471
K = 31

TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
ANTIBIOTICS = sys.argv[2:] or ["clindamycin", "erythromycin", "cefoxitin",
                               "oxacillin", "ciprofloxacin"]

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
os.makedirs(f"{BASE}/interpret2", exist_ok=True)


def counts(rows):
    acc = np.zeros(N_FEATURES, dtype=np.float64)
    for i in range(0, len(rows), 400):
        c = np.concatenate([np.asarray(cols[indptr[r]:indptr[r + 1]])
                            for r in rows[i:i + 400]])
        acc += np.bincount(c, minlength=N_FEATURES)
        del c
    return acc


def presence(rows, keep):
    """bool (len(rows) x len(keep)) presence over the chosen columns."""
    lut = np.full(N_FEATURES, -1, dtype=np.int32)
    lut[keep] = np.arange(len(keep), dtype=np.int32)
    P = np.zeros((len(rows), len(keep)), dtype=bool)
    for n, r in enumerate(rows):
        j = lut[np.asarray(cols[indptr[r]:indptr[r + 1]])]
        P[n, j[j >= 0]] = True
    return P


for abx in ANTIBIOTICS:
    labels = np.load(f"{BASE}/labels_{abx}.npy")
    rows = np.where(labels >= 0)[0]
    y = labels[rows]
    nR, nS = int((y == 1).sum()), int((y == 0).sum())
    print(f"\n{'='*70}\n{abx}  n={len(rows)}  R={nR}  S={nS}", flush=True)

    # same carrier-conditioned chi-square the pipeline uses, on all labelled rows
    oR, oS = counts(rows[y == 1]), counts(rows[y == 0])
    tot, pR = oR + oS, float((y == 1).mean())
    eR, eS = pR * tot, (1 - pR) * tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = (oR - eR) ** 2 / eR + (oS - eS) ** 2 / eS
    chi2[~np.isfinite(chi2)] = 0.0

    keep = np.argpartition(chi2, -TOP_N)[-TOP_N:]
    keep = np.sort(keep[chi2[keep] > 0])
    P = presence(rows, keep)

    # a column's presence vector, packed to bytes, is its pattern identity
    packed = np.packbits(P, axis=0)                      # (ceil(n/8), n_keep)
    groups = {}
    for j in range(P.shape[1]):
        groups.setdefault(packed[:, j].tobytes(), []).append(j)
    order = sorted(groups.values(), key=len, reverse=True)

    print(f"top {len(keep):,} k-mers collapse to {len(order):,} distinct "
          f"presence patterns", flush=True)
    sizes = np.array([len(g) for g in order])
    print(f"  largest pattern: {sizes[0]:,} k-mers "
          f"(~{sizes[0] + K - 1:,} bp of locus)")
    print(f"  patterns with >=100 k-mers: {int((sizes >= 100).sum())}  "
          f"covering {int(sizes[sizes >= 100].sum()):,} of the top "
          f"{len(keep):,} ({100*sizes[sizes>=100].sum()/len(keep):.1f}%)")
    print(f"  singleton patterns: {int((sizes == 1).sum()):,}")

    out = f"{BASE}/interpret2/loci_{abx}.tsv"
    with open(out, "w") as fh:
        fh.write("pattern\tn_kmers\tapprox_bp\tchi2\tpct_R_carry\t"
                 "pct_S_carry\todds_ratio\n")
        print(f"\n  {'#':>3} {'kmers':>7} {'~bp':>7} {'chi2':>7} "
              f"{'R carry':>8} {'S carry':>8} {'OR':>9}")
        for i, g in enumerate(order):
            v = P[:, g[0]]
            a, b = int(v[y == 1].sum()), int(v[y == 0].sum())
            cR, cS = a / nR, b / nS
            orat = ((a + 0.5) * (nS - b + 0.5)) / ((b + 0.5) * (nR - a + 0.5))
            row = (f"{i+1}\t{len(g)}\t{len(g)+K-1}\t{chi2[keep[g[0]]]:.1f}\t"
                   f"{cR:.4f}\t{cS:.4f}\t{orat:.2f}\n")
            fh.write(row)
            if i < 20:
                print(f"  {i+1:>3} {len(g):>7,} {len(g)+K-1:>7,} "
                      f"{chi2[keep[g[0]]]:>7.1f} {cR:>7.1%} {cS:>7.1%} "
                      f"{orat:>9.1f}")
        print(f"\n  wrote {out}", flush=True)
