"""
Resolve the ciprofloxacin contradiction.

The mechanism analysis found ciprofloxacin's signal in two point mutations,
gyrA S84L and grlA S80F, each assembling to a 61 bp contig from exactly 31
k-mers -- the arithmetic signature of a SNP. The locus-pattern analysis then
found ciprofloxacin's largest presence patterns span 700-960 bp. Both describe
the same model's top features, and they cannot both be right.

Three explanations, which this distinguishes:

  (a) the SNP k-mers sit INSIDE the large patterns, meaning the mutation is in
      perfect linkage with a ~1 kb block that travels with it. Ciprofloxacin
      would then have the same confounding structure as clindamycin, hidden by
      the external population happening to share the linkage.
  (b) the large patterns are unrelated loci that outrank the SNPs on all
      labelled genomes, so the earlier per-seed contig assembly was reading a
      different part of the ranking than the model actually leans on.
  (c) the large patterns ARE the SNP neighbourhood, assembled differently --
      the earlier run required a k-mer to be top-5,000 in four of five seeds,
      which would keep only the SNP-spanning window and drop its flanks.

The test: recompute the all-rows ranking, collapse to presence patterns as in
count_loci.py, then ask of each large pattern whether it contains the known
gyrA/grlA k-mers, and assemble it back to sequence so it can be identified.

Usage:  python3 cipro_patterns.py [top_n=10000] [n_patterns=10]
Output: printed report, plus /blue/boucher/nelseifi/interpret2/cipro_patterns.fasta
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
ABX = "ciprofloxacin"
TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
NPAT = int(sys.argv[2]) if len(sys.argv) > 2 else 10
COMP = str.maketrans("ACGT", "TGCA")


def rc(s):
    return s[::-1].translate(COMP)


# ------------------------------------------- the previously identified contigs
known = {}
with open(f"{BASE}/interpret2/contigs_all.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["antibiotic"] != ABX:
            continue
        seq = r["sequence"]
        kms = set()
        for i in range(len(seq) - K + 1):
            km = seq[i:i + K]
            kms.add(km)
            kms.add(rc(km))
        known[f"contig{r['contig']}"] = (len(seq), kms)
for n, (L, kms) in known.items():
    print(f"known {n}: {L} bp, {len(kms)//2} k-mers")
print()

# -------------------------------------------------- all-rows chi-square ranking
indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
labels = np.load(f"{BASE}/labels_{ABX}.npy")
rows = np.where(labels >= 0)[0]
y = labels[rows]
nR, nS = int((y == 1).sum()), int((y == 0).sum())
print(f"{ABX}: n={len(rows)} R={nR} S={nS}", flush=True)


def counts(rs):
    acc = np.zeros(N_FEATURES, dtype=np.float64)
    for i in range(0, len(rs), 400):
        c = np.concatenate([np.asarray(cols[indptr[r]:indptr[r + 1]])
                            for r in rs[i:i + 400]])
        acc += np.bincount(c, minlength=N_FEATURES)
        del c
    return acc


oR, oS = counts(rows[y == 1]), counts(rows[y == 0])
tot, pR = oR + oS, float((y == 1).mean())
eR, eS = pR * tot, (1 - pR) * tot
with np.errstate(divide="ignore", invalid="ignore"):
    chi2 = (oR - eR) ** 2 / eR + (oS - eS) ** 2 / eS
chi2[~np.isfinite(chi2)] = 0.0

keep = np.argpartition(chi2, -TOP_N)[-TOP_N:]
keep = np.sort(keep[chi2[keep] > 0])

lut = np.full(N_FEATURES, -1, dtype=np.int32)
lut[keep] = np.arange(len(keep), dtype=np.int32)
P = np.zeros((len(rows), len(keep)), dtype=bool)
for n, r in enumerate(rows):
    j = lut[np.asarray(cols[indptr[r]:indptr[r + 1]])]
    P[n, j[j >= 0]] = True

packed = np.packbits(P, axis=0)
groups = defaultdict(list)
for j in range(P.shape[1]):
    groups[packed[:, j].tobytes()].append(j)
order = sorted(groups.values(), key=len, reverse=True)
print(f"top {len(keep):,} k-mers -> {len(order):,} patterns\n", flush=True)

# -------------------------------------------------------- sequences we need
want = {int(keep[j]) for g in order[:NPAT] for j in g}
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
print(f"resolved {len(seqs):,} sequences\n", flush=True)


# ------------------------------------------------------------- assembly
def assemble(kmers):
    """Greedy walk on (k-1)-mer overlap, trying both orientations."""
    pool = set(kmers)
    by_pre, by_suf = defaultdict(list), defaultdict(list)
    for km in pool:
        for s in (km, rc(km)):
            by_pre[s[:K - 1]].append((km, s))
            by_suf[s[-(K - 1):]].append((km, s))

    def take(idx, key, used):
        for km, s in idx.get(key, ()):
            if km not in used:
                return km, s
        return None

    used, contigs = set(), []
    for seed in sorted(pool):
        if seed in used:
            continue
        used.add(seed)
        contig = seed
        while True:
            h = take(by_pre, contig[-(K - 1):], used)
            if not h:
                break
            used.add(h[0])
            contig += h[1][-1]
        while True:
            h = take(by_suf, contig[:K - 1], used)
            if not h:
                break
            used.add(h[0])
            contig = h[1][0] + contig
        contigs.append(contig)
    return sorted(contigs, key=len, reverse=True)


out = open(f"{BASE}/interpret2/cipro_patterns.fasta", "w")
for i, g in enumerate(order[:NPAT]):
    v = P[:, g[0]]
    a, b = int(v[y == 1].sum()), int(v[y == 0].sum())
    orat = ((a + 0.5) * (nS - b + 0.5)) / ((b + 0.5) * (nR - a + 0.5))
    kms = {seqs[int(keep[j])] for j in g}

    print(f"=== pattern {i+1}: {len(g):,} k-mers   chi2 {chi2[keep[g[0]]]:.1f}   "
          f"R {a/nR:.1%}  S {b/nS:.1%}  OR {orat:.1f}")

    # containment in the previously identified contigs
    hits = []
    for name, (L, kk) in known.items():
        n_in = len(kms & kk)
        if n_in:
            hits.append(f"{name} ({L}bp): {n_in}/{len(kk)//2} of its k-mers")
    print("    known-contig overlap: " + ("; ".join(hits) if hits else "NONE"))

    ctgs = assemble(kms)
    print(f"    assembles to {len(ctgs)} contig(s); "
          f"lengths {[len(c) for c in ctgs[:6]]}"
          f"{' ...' if len(ctgs) > 6 else ''}")
    for n, c in enumerate(ctgs[:3]):
        out.write(f">pattern{i+1}_contig{n+1}_len{len(c)}_R{a/nR:.3f}_S{b/nS:.3f}\n")
        for p in range(0, len(c), 70):
            out.write(c[p:p + 70] + "\n")
    print(flush=True)

out.close()

# ------------------------------------ where do the known SNP k-mers rank now?
print("=" * 66)
allk = {}
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        for name, (L, kk) in known.items():
            if s in kk:
                allk.setdefault(name, []).append(int(i_str))

rank = np.argsort(np.argsort(-chi2))
for name, (L, kk) in known.items():
    idx = np.array(allk.get(name, []), dtype=np.int64)
    if len(idx) == 0:
        print(f"{name}: no matrix columns found")
        continue
    print(f"{name} ({L} bp): {len(idx)} columns, chi2 "
          f"{chi2[idx].min():.1f}-{chi2[idx].max():.1f}, "
          f"best rank {int(rank[idx].min()):,}, "
          f"{int((rank[idx] < TOP_N).sum())}/{len(idx)} inside top {TOP_N:,}")

print(f"\nwrote {BASE}/interpret2/cipro_patterns.fasta")
