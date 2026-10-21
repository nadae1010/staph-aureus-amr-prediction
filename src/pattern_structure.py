"""
What is each model actually betting on? Ranked correctly this time.

Two faults in count_loci.py are corrected here.

First, it ordered patterns by SIZE. Size is the number of k-mers sharing a
presence vector, which measures how much sequence travels together, not how
predictive it is. Ciprofloxacin shows the gap: its largest pattern holds 933
k-mers scoring chi2 600, while its best feature is a 31 k-mer SNP signature at
rank 0 scoring 1303. A table built on the size ordering describes the bulkiest
feature and ignores the strongest one. Patterns are ranked by chi-square here,
with the size ordering printed alongside so the two can be compared.

Second, it reported a locus length of n_kmers + k - 1, which assumes the k-mers
of a pattern tile one contiguous region. For ciprofloxacin that is false: the
933 k-mers assemble into 42 separate 61 bp contigs, unlinked point mutations
co-occurring because they are inherited together. Each pattern is assembled
here instead, and the contig structure separates:

  single locus  few contigs, at least one long -- an acquired gene
  haplotype     many contigs, most 2k-1 bp -- point mutations locked by ancestry

A model resting on a gene carries a biological bet. A model resting on a
haplotype carries a bet on population structure, which survives only in a
population sharing that structure. The closing summary reports, for the
strongest features of each antibiotic, how the two are divided.

Usage:  python3 pattern_structure.py [top_n=10000] [n_show=10] [abx ...]
Output: printed report, plus /blue/boucher/nelseifi/interpret2/patterns_<abx>.fasta
"""
import sys
from collections import defaultdict

import numpy as np

BASE = "/blue/boucher/nelseifi"
MAT = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
N_FEATURES = 11_351_471
K = 31
COMP = str.maketrans("ACGT", "TGCA")

TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
NSHOW = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ANTIBIOTICS = sys.argv[3:] or ["clindamycin", "erythromycin", "cefoxitin",
                               "oxacillin", "ciprofloxacin"]

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")


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


def classify(lengths):
    snpish = sum(1 for x in lengths if x <= 2 * K + 10)
    if len(lengths) <= 3 and max(lengths) >= 200:
        return "single locus"
    if len(lengths) >= 5 and snpish >= 0.6 * len(lengths):
        return f"haplotype ({snpish}/{len(lengths)} SNP-sized)"
    return "mixed"


# ------------------------------------------------------------- per antibiotic
plans = []
for abx in ANTIBIOTICS:
    labels = np.load(f"{BASE}/labels_{abx}.npy")
    rows = np.where(labels >= 0)[0]
    y = labels[rows]
    nR, nS = int((y == 1).sum()), int((y == 0).sum())
    print(f"\n{'='*78}\n{abx}  n={len(rows)}  R={nR}  S={nS}", flush=True)

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
    allpat = list(groups.values())

    recs = []
    for g in allpat:
        v = P[:, g[0]]
        a, b = int(v[y == 1].sum()), int(v[y == 0].sum())
        orat = ((a + .5) * (nS - b + .5)) / ((b + .5) * (nR - a + .5))
        recs.append(dict(n=len(g), chi2=float(chi2[keep[g[0]]]),
                         cR=a / nR, cS=b / nS, orat=orat,
                         idx=[int(keep[j]) for j in g],
                         best_rank=int((chi2 > chi2[keep[g[0]]]).sum())))

    by_chi2 = sorted(recs, key=lambda r: -r["chi2"])[:NSHOW]
    by_size = sorted(recs, key=lambda r: -r["n"])[:NSHOW]
    # patterns holding the 1,000 strongest individual features
    cut = np.sort(chi2[keep])[-1000]
    strong = [r for r in recs if r["chi2"] >= cut]
    print(f"top {len(keep):,} k-mers -> {len(recs):,} patterns; "
          f"the 1,000 strongest features sit in {len(strong)} of them",
          flush=True)
    plans.append((abx, recs, by_chi2, by_size, strong))

# -------------------------------------------- resolve every sequence, one scan
want = {c for _, _, bc, bs, st in plans
        for grp in (bc, bs, st) for r in grp for c in r["idx"]}
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
print(f"\nresolved {len(seqs):,} k-mer sequences", flush=True)

HDR = (f"  {'#':>2} {'chi2':>7} {'rank':>7} {'kmers':>6} {'R':>6} {'S':>6} "
       f"{'OR':>7} {'ctgs':>5} {'longest':>7}  structure")

for abx, recs, by_chi2, by_size, strong in plans:
    print(f"\n{'='*78}\n{abx}")

    for title, grp in (("ranked by chi-square (what the model leans on)", by_chi2),
                       ("ranked by size (what count_loci.py reported)", by_size)):
        print(f"\n  {title}")
        print(HDR)
        for i, r in enumerate(grp):
            ctgs = assemble({seqs[c] for c in r["idx"]})
            L = [len(c) for c in ctgs]
            r["ctgs"], r["L"] = ctgs, L
            print(f"  {i+1:>2} {r['chi2']:>7.1f} {r['best_rank']:>7,} "
                  f"{r['n']:>6,} {r['cR']:>5.1%} {r['cS']:>5.1%} "
                  f"{r['orat']:>7.1f} {len(ctgs):>5} {max(L):>7,}  "
                  f"{classify(L)}")

    # how the strongest features divide between gene-like and haplotype-like
    kinds = defaultdict(int)
    kmers_in = defaultdict(int)
    for r in strong:
        L = r.get("L") or [len(c) for c in assemble({seqs[c] for c in r["idx"]})]
        k = classify(L).split(" (")[0]
        kinds[k] += 1
        kmers_in[k] += r["n"]
    print(f"\n  the 1,000 strongest features, by pattern structure:")
    for k in ("single locus", "haplotype", "mixed"):
        if kinds[k]:
            print(f"    {k:14s} {kinds[k]:>4} patterns, {kmers_in[k]:>7,} k-mers")

    with open(f"{BASE}/interpret2/patterns_{abx}.fasta", "w") as out:
        for i, r in enumerate(by_chi2):
            for j, c in enumerate(r["ctgs"][:5]):
                out.write(f">{abx}_chi2rank{i+1}_c{j+1}_len{len(c)}"
                          f"_chi2{r['chi2']:.0f}_R{r['cR']:.3f}_S{r['cS']:.3f}\n")
                for p in range(0, len(c), 70):
                    out.write(c[p:p + 70] + "\n")
    print(f"  wrote {BASE}/interpret2/patterns_{abx}.fasta", flush=True)
