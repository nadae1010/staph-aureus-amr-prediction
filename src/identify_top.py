"""
Identify what each model's strongest features actually are.

Ranked by chi-square rather than by pattern size, clindamycin's best feature is
not mecA. Ranks 1-6 sit at 68.7% of resistant and 12.8% of susceptible (chi2
364.3, OR 14.9); mecA, at 82.2% and 20.5%, does not appear until rank 31. So
something outranks mecA with better specificity, and the earlier contig
assembly never surfaced it -- that run kept only k-mers appearing in the top
5,000 of at least four of five seed folds, a different slice of the ranking.

The per-pattern contigs are too short to identify (32-57 bp), because a pattern
holds only k-mers with identical presence and a real locus fragments into
several patterns as coverage varies across genomes. This assembles the whole
top-N feature set at once, which lets adjacent k-mers from different patterns
join into full-length contigs, then characterises each one:

  its best chi-square and rank, so the strongest contigs are identifiable
  carriage in resistant and susceptible isolates, with an odds ratio
  whether it matches mecA, by comparison against the methicillin contig
  whether it matches anything already in contigs_all.tsv

A contig that outranks mecA and is not mecA is the thing to BLAST.

Usage:  python3 identify_top.py [top_n=2000] [abx ...]   (default clindamycin)
Output: printed report, plus /blue/boucher/nelseifi/interpret2/top_<abx>.fasta
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

TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000
ANTIBIOTICS = sys.argv[2:] or ["clindamycin"]

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")


def rc(s):
    return s[::-1].translate(COMP)


# ----------------------------------------------- reference loci for labelling
ref = {}
with open(f"{BASE}/interpret2/contigs_all.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        seq = r["sequence"]
        kms = set()
        for i in range(len(seq) - K + 1):
            km = seq[i:i + K]
            kms.add(km)
            kms.add(rc(km))
        ref[f"{r['antibiotic']}_c{r['contig']}"] = kms
meca = ref.get("methicillin_c1", set())
print(f"reference loci: {len(ref)} contigs; mecA block {len(meca)//2} k-mers\n",
      flush=True)


def counts(rs):
    acc = np.zeros(N_FEATURES, dtype=np.float64)
    for i in range(0, len(rs), 400):
        c = np.concatenate([np.asarray(cols[indptr[r]:indptr[r + 1]])
                            for r in rs[i:i + 400]])
        acc += np.bincount(c, minlength=N_FEATURES)
        del c
    return acc


def assemble(kmers):
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
        members = [seed]
        contig = seed
        while True:
            h = take(by_pre, contig[-(K - 1):], used)
            if not h:
                break
            used.add(h[0])
            members.append(h[0])
            contig += h[1][-1]
        while True:
            h = take(by_suf, contig[:K - 1], used)
            if not h:
                break
            used.add(h[0])
            members.append(h[0])
            contig = h[1][0] + contig
        contigs.append((contig, members))
    return sorted(contigs, key=lambda t: -len(t[0]))


for abx in ANTIBIOTICS:
    labels = np.load(f"{BASE}/labels_{abx}.npy")
    rows = np.where(labels >= 0)[0]
    y = labels[rows]
    nR, nS = int((y == 1).sum()), int((y == 0).sum())
    print(f"{'='*78}\n{abx}  n={len(rows)}  R={nR}  S={nS}", flush=True)

    oR, oS = counts(rows[y == 1]), counts(rows[y == 0])
    tot, pR = oR + oS, float((y == 1).mean())
    eR, eS = pR * tot, (1 - pR) * tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = (oR - eR) ** 2 / eR + (oS - eS) ** 2 / eS
    chi2[~np.isfinite(chi2)] = 0.0

    keep = np.argpartition(chi2, -TOP_N)[-TOP_N:]
    keep = np.sort(keep[chi2[keep] > 0])
    rank = np.argsort(np.argsort(-chi2))

    lut = np.full(N_FEATURES, -1, dtype=np.int32)
    lut[keep] = np.arange(len(keep), dtype=np.int32)
    P = np.zeros((len(rows), len(keep)), dtype=bool)
    for n, r in enumerate(rows):
        j = lut[np.asarray(cols[indptr[r]:indptr[r + 1]])]
        P[n, j[j >= 0]] = True

    seqs, col_of = {}, {}
    want = {int(c) for c in keep}
    with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
        next(f)
        for line in f:
            i_str, s = line.rstrip("\n").split(",", 1)
            i = int(i_str)
            if i in want:
                seqs[i] = s
                col_of[s] = i
                if len(seqs) == len(want):
                    break
    print(f"resolved {len(seqs):,}/{TOP_N:,} sequences; assembling", flush=True)

    ctgs = assemble(set(seqs.values()))
    print(f"{len(ctgs)} contigs\n", flush=True)

    recs = []
    for seq, members in ctgs:
        mcols = [col_of[m] for m in members]
        best = max(mcols, key=lambda c: chi2[c])
        v = P[:, lut[best]]
        a, b = int(v[y == 1].sum()), int(v[y == 0].sum())
        orat = ((a + .5) * (nS - b + .5)) / ((b + .5) * (nR - a + .5))
        kms = set(members) | {rc(m) for m in members}
        n_meca = len(kms & meca) // 2 if meca else 0
        hits = sorted((len(kms & kk) // 2, name) for name, kk in ref.items()
                      if kms & kk)[-3:]
        recs.append(dict(seq=seq, n=len(members), chi2=float(chi2[best]),
                         rank=int(rank[best]), cR=a / nR, cS=b / nS, orat=orat,
                         meca=n_meca,
                         hits="; ".join(f"{nm}:{c}" for c, nm in reversed(hits))))

    recs.sort(key=lambda r: -r["chi2"])
    print(f"  {'#':>3} {'bp':>6} {'kmers':>6} {'chi2':>7} {'rank':>6} "
          f"{'R':>6} {'S':>6} {'OR':>8} {'mecA':>6}  matches")
    for i, r in enumerate(recs[:25]):
        print(f"  {i+1:>3} {len(r['seq']):>6,} {r['n']:>6,} {r['chi2']:>7.1f} "
              f"{r['rank']:>6,} {r['cR']:>5.1%} {r['cS']:>5.1%} "
              f"{r['orat']:>8.1f} {r['meca']:>6}  {r['hits'][:52]}")

    with open(f"{BASE}/interpret2/top_{abx}.fasta", "w") as out:
        for i, r in enumerate(recs[:25]):
            out.write(f">{abx}_rank{i+1}_len{len(r['seq'])}_chi2{r['chi2']:.0f}"
                      f"_R{r['cR']:.3f}_S{r['cS']:.3f}_OR{r['orat']:.1f}"
                      f"_mecA{r['meca']}\n")
            for p in range(0, len(r["seq"]), 70):
                out.write(r["seq"][p:p + 70] + "\n")
    print(f"\n  wrote {BASE}/interpret2/top_{abx}.fasta\n", flush=True)
