"""
Assemble the stable top chi-square k-mers of each antibiotic into contigs, so
that top features can be identified by locus rather than as isolated 31-mers.

Overlapping k-mers from the same gene share 30 bp, so they can be stitched into
longer sequences. A 31 bp BLAST query matches whole-genome assemblies and names
no gene; a 200-400 bp contig usually returns an annotated hit directly.

For each antibiotic:
  - stable k-mers = in the top N of at least MIN_SEEDS of the 5 seeds
  - greedy orientation-aware assembly on 30 bp suffix/prefix overlap
  - per-contig resistant/susceptible prevalence, taken from its highest-scoring
    member k-mer (members come from one locus, so their prevalences agree)

Sequences are read from the k-mer index once for the union of all antibiotics,
and genome presence is computed in a single pass over the matrix.

Usage:  python3 kmer_contigs.py [n_stable=200] [min_seeds=4]
Output: /blue/boucher/nelseifi/interpret2/
          contigs_all.tsv          every contig, all antibiotics
          <abx>_contigs.fasta      per-antibiotic FASTA for BLAST
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np

BASE = "/blue/boucher/nelseifi"
MAT = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
OUT = f"{BASE}/interpret2"
N_FEATURES = 11_351_471
K = 31

ANTIBIOTICS = ["methicillin", "oxacillin", "cefoxitin",
               "erythromycin", "ciprofloxacin", "clindamycin"]
N_STABLE = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MIN_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4

os.makedirs(OUT, exist_ok=True)
COMP = str.maketrans("ACGT", "TGCA")


def rc(s):
    return s[::-1].translate(COMP)


# ---------------------------------------------------------------- stable sets
stable = {}
for abx in ANTIBIOTICS:
    counts, scores = Counter(), defaultdict(list)
    for s in range(5):
        z = np.load(f"{BASE}/chi2/{abx}_seed{s}_chi2top.npz")
        for i, sc in zip(z["idx"][:N_STABLE], z["score"][:N_STABLE]):
            i = int(i)
            counts[i] += 1
            scores[i].append(float(sc))
    stable[abx] = {i: float(np.mean(scores[i]))
                   for i, c in counts.items() if c >= MIN_SEEDS}
    print(f"{abx:14s} {len(stable[abx]):4d} stable k-mers "
          f"(top {N_STABLE} in >={MIN_SEEDS}/5 seeds)", flush=True)

union = sorted(set().union(*[set(d) for d in stable.values()]))
print(f"\nunion across antibiotics: {len(union)} k-mers", flush=True)

# ------------------------------------------------------------- sequences once
want = set(union)
seqs = {}
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, seq = line.rstrip().split(",", 1)
        i = int(i_str)
        if i in want:
            seqs[i] = seq.upper()
            if len(seqs) == len(want):
                break
print(f"sequences found: {len(seqs)}/{len(want)}", flush=True)

# ---------------------------------------------------- genome presence, 1 pass
indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
n_gen = len(indptr) - 1

lut = np.full(N_FEATURES, -1, dtype=np.int32)
lut[np.array(union, dtype=np.int64)] = np.arange(len(union), dtype=np.int32)
present = np.zeros((n_gen, len(union)), dtype=bool)
for r in range(n_gen):
    j = lut[np.asarray(cols[indptr[r]:indptr[r + 1]])]
    present[r, j[j >= 0]] = True
pos = {k: j for j, k in enumerate(union)}
print(f"presence matrix {present.shape}, "
      f"mean k-mers per genome {present.sum(1).mean():.1f}\n", flush=True)


# ----------------------------------------------------------------- assembly
def assemble(ids, chi2):
    """Greedy orientation-aware unitigs on 30 bp overlap."""
    by_pref, by_suff = defaultdict(list), defaultdict(list)
    for i in ids:
        for o in (seqs[i], rc(seqs[i])):
            by_pref[o[:K - 1]].append((i, o))
            by_suff[o[-(K - 1):]].append((i, o))

    used, contigs = set(), []
    for start in sorted(ids, key=lambda i: -chi2[i]):
        if start in used:
            continue
        used.add(start)
        seq, members = seqs[start], [start]

        while True:                                   # extend right
            nxt = next(((i, o) for i, o in by_pref.get(seq[-(K - 1):], [])
                        if i not in used), None)
            if nxt is None:
                break
            used.add(nxt[0])
            members.append(nxt[0])
            seq += nxt[1][-1]

        while True:                                   # extend left
            prv = next(((i, o) for i, o in by_suff.get(seq[:K - 1], [])
                        if i not in used), None)
            if prv is None:
                break
            used.add(prv[0])
            members.append(prv[0])
            seq = prv[1][0] + seq

        contigs.append((seq, members))
    return contigs


rows = []
for abx in ANTIBIOTICS:
    chi2 = stable[abx]
    labels = np.load(f"{BASE}/labels_{abx}.npy")
    nR = int((labels == 1).sum())
    nS = int((labels == 0).sum())
    contigs = assemble(list(chi2), chi2)
    contigs.sort(key=lambda c: -max(chi2[m] for m in c[1]))

    print(f"=== {abx}: {len(chi2)} k-mers -> {len(contigs)} contigs "
          f"(longest {max(len(c[0]) for c in contigs)} bp)")
    with open(f"{OUT}/{abx}_contigs.fasta", "w") as fa:
        for n, (seq, members) in enumerate(contigs, 1):
            top = max(members, key=lambda m: chi2[m])
            col = pos[top]
            pR = 100 * present[labels == 1, col].sum() / nR
            pS = 100 * present[labels == 0, col].sum() / nS
            rec = dict(antibiotic=abx, contig=n, length=len(seq),
                       n_kmers=len(members),
                       mean_chi2=np.mean([chi2[m] for m in members]),
                       max_chi2=chi2[top], pct_R=pR, pct_S=pS,
                       enriched="R" if pR > pS else "S", sequence=seq)
            rows.append(rec)
            fa.write(f">{abx}_contig{n}_len{len(seq)}_n{len(members)}"
                     f"_chi2{chi2[top]:.0f}_R{pR:.0f}_S{pS:.0f}\n{seq}\n")
            if n <= 8:
                print(f"  contig {n:2d}  {len(seq):4d} bp  {len(members):3d} k-mers  "
                      f"chi2={chi2[top]:7.1f}  R={pR:5.1f}%  S={pS:5.1f}%  "
                      f"-> {rec['enriched']}")
    print(flush=True)

cols_out = ["antibiotic", "contig", "length", "n_kmers", "mean_chi2",
            "max_chi2", "pct_R", "pct_S", "enriched", "sequence"]
with open(f"{OUT}/contigs_all.tsv", "w") as t:
    t.write("\t".join(cols_out) + "\n")
    for r in rows:
        t.write("\t".join(
            f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c])
            for c in cols_out) + "\n")
print(f"wrote {OUT}/contigs_all.tsv ({len(rows)} contigs) "
      f"and per-antibiotic FASTAs")
[nelseifi@login8 staph_amr_project]$ cat mlsb_test.py
"""
Test whether erm(C) carriage shows the inducible-MLSb phenotype in this dataset.

erm(C) methylates the shared 23S target of erythromycin and clindamycin, so
constitutive expression would make carriers resistant to both. Inducible
expression makes carriers erythromycin-resistant but clindamycin-susceptible --
the phenotype a D-test detects. If that pattern holds here it explains why the
erm(C) plasmid is a top-ranked erythromycin feature but absent from
clindamycin's feature set.

Method: take named contigs from interpret2/contigs_all.tsv, recover their
constituent k-mer column indices from the k-mer index, score each genome as a
carrier when it contains at least half the contig's k-mers, then cross-tabulate
carriage against both phenotypes.

Usage:  python3 mlsb_test.py
Output: printed tables (nothing written to disk)
"""
import numpy as np

BASE = "/blue/boucher/nelseifi"
MAT = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
N_FEATURES = 11_351_471
K = 31
CARRIER_FRAC = 0.5

# (label shown in output) -> (antibiotic column, contig number) in contigs_all.tsv
TARGETS = {
    "erm(C) plasmid": ("erythromycin", "1"),
    "mecA block": ("methicillin", "1"),
}
COMP = str.maketrans("ACGT", "TGCA")


def rc(s):
    return s[::-1].translate(COMP)


# ------------------------------------------------- contig sequences -> k-mers
seqs = {}
with open(f"{BASE}/interpret2/contigs_all.tsv") as f:
    hdr = f.readline().rstrip("\n").split("\t")
    ia, ic, iseq = hdr.index("antibiotic"), hdr.index("contig"), hdr.index("sequence")
    for line in f:
        p = line.rstrip("\n").split("\t")
        for name, (abx, cn) in TARGETS.items():
            if p[ia] == abx and p[ic] == cn:
                seqs[name] = p[iseq]

lookup = {}                      # k-mer string (either orientation) -> name
for name, seq in seqs.items():
    n = 0
    for i in range(len(seq) - K + 1):
        km = seq[i:i + K]
        lookup[km] = name
        lookup[rc(km)] = name
        n += 1
    print(f"{name}: {len(seq)} bp -> {n} k-mers")

# ------------------------------------------- one pass over the k-mer index
cols_of = {name: [] for name in seqs}
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        name = lookup.get(s)
        if name is not None:
            cols_of[name].append(int(i_str))
for name in seqs:
    print(f"{name}: {len(cols_of[name])} k-mers found in the matrix index")

# -------------------------------------------------- carriage per genome
indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
n_gen = len(indptr) - 1

allcols = sorted({c for v in cols_of.values() for c in v})
lut = np.full(N_FEATURES, -1, dtype=np.int32)
lut[np.array(allcols, dtype=np.int64)] = np.arange(len(allcols), dtype=np.int32)
present = np.zeros((n_gen, len(allcols)), dtype=bool)
for r in range(n_gen):
    j = lut[np.asarray(cols[indptr[r]:indptr[r + 1]])]
    present[r, j[j >= 0]] = True

pos = {c: j for j, c in enumerate(allcols)}
carrier = {}
for name, cs in cols_of.items():
    idx = [pos[c] for c in cs]
    frac = present[:, idx].mean(axis=1)
    carrier[name] = frac >= CARRIER_FRAC
    print(f"\n{name}: {carrier[name].sum()} / {n_gen} genomes carry it "
          f"(>= {CARRIER_FRAC:.0%} of its k-mers)")

# ------------------------------------------------------------- phenotypes
ery = np.load(f"{BASE}/labels_erythromycin.npy")
cli = np.load(f"{BASE}/labels_clindamycin.npy")

for name, has in carrier.items():
    print(f"\n=== {name}")
    for abx, lab in (("erythromycin", ery), ("clindamycin", cli)):
        for cls, v in (("R", 1), ("S", 0)):
            sel = lab == v
            print(f"  {abx:13s} {cls}: {100 * has[sel].mean():5.1f}% carry "
                  f"({int(has[sel].sum())}/{int(sel.sum())})")

    both = (ery >= 0) & (cli >= 0)
    sub = both & has
    print(f"\n  cross-tab among carriers with both phenotypes (n={int(sub.sum())}):")
    print(f"    {'':12s}{'clinda-R':>10}{'clinda-S':>10}")
    for e_lab, e_v in (("ery-R", 1), ("ery-S", 0)):
        row = [int((sub & (ery == e_v) & (cli == c_v)).sum()) for c_v in (1, 0)]
        print(f"    {e_lab:12s}{row[0]:>10}{row[1]:>10}")

    nonsub = both & ~has
    print(f"\n  same table for NON-carriers (n={int(nonsub.sum())}):")
    print(f"    {'':12s}{'clinda-R':>10}{'clinda-S':>10}")
    for e_lab, e_v in (("ery-R", 1), ("ery-S", 0)):
        row = [int((nonsub & (ery == e_v) & (cli == c_v)).sum()) for c_v in (1, 0)]
        print(f"    {e_lab:12s}{row[0]:>10}{row[1]:>10}")
