"""
erm(B) follow-up: is it a real clindamycin determinant, and how much resistance
do the known erm genes leave unexplained?

Stratifying clindamycin on mecA carriage surfaced a contig the pipeline had
never reported: 523 bp, carried by ~29% of resistant and ~2% of susceptible
mecA-negative genomes, identified by blastx as Erm(B) at 100% identity. erm(B),
like erm(C), methylates the ribosome and confers resistance to erythromycin and
clindamycin together. This script asks four things of it:

  1. OVERLAP. Do erm(B) and erm(C) cover different resistant genomes or the
     same ones? That fixes the combined coverage of the known erm genes and so
     how much clindamycin resistance remains unexplained.

  2. EXTERNAL ODDS RATIO. The evidence that erm(C) is causal and mecA is not
     was that, in the independent NCBI genomes, erm(C)'s clindamycin odds ratio
     strengthened (4.9 -> 162) while mecA's collapsed (16.8 -> 1.5). Linkage is
     a property of a population; causation is not. Prediction stated in
     advance: if erm(B) is causal, its external clindamycin OR stays well above
     1 (95% CI excluding 1); a linked marker would drift toward 1.

  3. POOLED RANK. Where erm(B)'s best k-mer sits in the original pooled
     clindamycin ranking -- the measure of how deeply mecA buried it.

  4. CLINDAMYCIN-RESISTANT, ERYTHROMYCIN-SUSCEPTIBLE GENOMES. Every erm gene
     confers both resistances, so a genome resistant to clindamycin but not
     erythromycin cannot be explained by any erm gene. Lincosamide-specific
     mechanisms -- lnu(A), vga(A), lsa(E) -- give exactly that pattern. If the
     group is large enough, its distinguishing features are ranked and
     assembled for BLAST.

Carriage is scored on an evenly spaced sample of each locus's k-mers (the same
sample on both sides): a genome carries the locus if at least half are present.
External genomes are read directly from the raw assemblies, because erm(B) was
never selected by any model and so is absent from external_features.npz.

Sanity checks built in: mecA (C-terminal) should reproduce roughly 16.8 / 1.5
and erm(C) roughly 4.9 / 162 for clindamycin -- the figures linkage_external.py
reported on all matched external genomes. If they do not, the external scan is
wrong and nothing else here should be trusted.

Usage:  python3 erm_followup.py [n_sample=24] [min_group=15]
Output: printed report; if (4) runs,
        /blue/boucher/nelseifi/interpret2/clindaR_eryS.fasta
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

BASE = "/blue/boucher/nelseifi"
MAT = f"{BASE}/kmx_output"
EXT = f"{BASE}/external"
SUFFIX = "k31_min5_max3000_d0"
N_FEATURES = 11_351_471
K = 31
COMP = str.maketrans("ACGT", "TGCA")
FRAC = 0.5

NSAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 24
MIN_GROUP = int(sys.argv[2]) if len(sys.argv) > 2 else 15

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
n_train = len(indptr) - 1


def rc(s):
    return s[::-1].translate(COMP)


# ------------------------------------------------------------- locus sequences
loci = {}
ref = {}
with open(f"{BASE}/interpret2/contigs_all.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        key = f"{r['antibiotic']}_c{r['contig']}"
        ref[key] = r["sequence"]
        if key == "erythromycin_c1":
            loci["erm(C)"] = r["sequence"]
        elif key == "methicillin_c1":
            loci["mecA (C-term)"] = r["sequence"]
        elif key == "methicillin_c2":
            loci["mecA (N-term)"] = r["sequence"]

hdr, buf, records = None, [], []
with open(f"{BASE}/interpret2/stratified_mecA-negative.fasta") as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if hdr:
                records.append((hdr, "".join(buf)))
            hdr, buf = line[1:], []
        else:
            buf.append(line)
if hdr:
    records.append((hdr, "".join(buf)))
ermb = next(s for h, s in records if "_rank1_" in h)
loci["erm(B)"] = ermb
ref["erm(B)"] = ermb

ORDER = ["erm(B)", "erm(C)", "mecA (C-term)", "mecA (N-term)"]
for n in ORDER:
    print(f"{n:15s} {len(loci[n]):>6,} bp", flush=True)

# ----------------------- resolve every locus k-mer to its matrix column, once
want = {}
for name in ORDER:
    s = loci[name]
    for p in range(len(s) - K + 1):
        km = s[p:p + K]
        want[km] = (name, p)
        want[rc(km)] = (name, p)
resolved = defaultdict(dict)
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        hit = want.get(s)
        if hit:
            resolved[hit[0]][hit[1]] = int(i_str)

sample = {}
for name in ORDER:
    pos = sorted(resolved[name])
    take = pos if len(pos) <= NSAMPLE else [
        pos[int(round(i))] for i in np.linspace(0, len(pos) - 1, NSAMPLE)]
    sample[name] = [(p, resolved[name][p], loci[name][p:p + K]) for p in take]
    print(f"  {name:15s} {len(pos):>5,} k-mers in matrix, "
          f"{len(sample[name])} sampled", flush=True)

# ---------------------------------------------------- training-side carriage
allc = sorted({c for name in ORDER for _, c, _ in sample[name]})
lut = np.full(N_FEATURES, -1, dtype=np.int32)
lut[np.array(allc, dtype=np.int64)] = np.arange(len(allc), dtype=np.int32)
pres = np.zeros((n_train, len(allc)), dtype=bool)
for r in range(n_train):
    j = lut[np.asarray(cols[indptr[r]:indptr[r + 1]])]
    pres[r, j[j >= 0]] = True
cpos = {c: j for j, c in enumerate(allc)}
carry = {name: pres[:, [cpos[c] for _, c, _ in sample[name]]].mean(1) >= FRAC
         for name in ORDER}
meca = carry["mecA (C-term)"] | carry["mecA (N-term)"]
print(f"\ntraining carriers of {n_train}: " +
      ", ".join(f"{n} {int(carry[n].sum())}" for n in ORDER) +
      f", mecA either {int(meca.sum())}", flush=True)


def odds(car, y):
    a = int((car & (y == 1)).sum())
    b = int((car & (y == 0)).sum())
    c = int((~car & (y == 1)).sum())
    d = int((~car & (y == 0)).sum())
    o = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
    se = np.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (d + .5))
    return a, b, c, d, o, np.exp(np.log(o) - 1.96 * se), np.exp(np.log(o) + 1.96 * se)


def assoc_table(title, car, y):
    nR, nS = int((y == 1).sum()), int((y == 0).sum())
    print(f"\n  {title}  (R={nR}, S={nS})")
    print(f"    {'locus':15s} {'R carry':>14} {'S carry':>14} {'OR':>8}  95% CI")
    out = {}
    for name in ORDER:
        a, b, c, d, o, lo, hi = odds(car[name], y)
        out[name] = (o, lo, hi)
        print(f"    {name:15s} {a:>4}/{nR:<4} {a/max(nR,1):5.1%} "
              f"{b:>4}/{nS:<4} {b/max(nS,1):5.1%} {o:>8.1f}  [{lo:.1f}, {hi:.1f}]")
    return out


def overlap(title, cb, cc, y):
    R = y == 1
    nR = int(R.sum())
    both = int((cb & cc & R).sum())
    bo = int((cb & ~cc & R).sum())
    co = int((~cb & cc & R).sum())
    nei = int((~cb & ~cc & R).sum())
    print(f"\n  {title}: resistant genomes by erm gene (n={nR})")
    print(f"    erm(B) only {bo:4d}   erm(C) only {co:4d}   both {both:4d}   "
          f"neither {nei:4d}")
    print(f"    covered by either: {(nR-nei)/max(nR,1):.1%}   "
          f"unexplained: {nei/max(nR,1):.1%}", flush=True)


clab = np.load(f"{BASE}/labels_clindamycin.npy")
elab = np.load(f"{BASE}/labels_erythromycin.npy")

print(f"\n{'='*76}\nTRAINING (BV-BRC)", flush=True)
rows = np.where(clab >= 0)[0]
y = clab[rows]
tr_clinda = assoc_table("clindamycin", {n: carry[n][rows] for n in ORDER}, y)
for tag, mask in (("all clindamycin-labelled", np.ones(len(rows), bool)),
                  ("mecA-negative", ~meca[rows]),
                  ("mecA-positive", meca[rows])):
    overlap(tag, carry["erm(B)"][rows][mask], carry["erm(C)"][rows][mask],
            y[mask])
erows = np.where(elab >= 0)[0]
tr_ery = assoc_table("erythromycin", {n: carry[n][erows] for n in ORDER},
                     elab[erows])


# -------------------------------- pooled rank: how deeply was erm(B) buried?
def counts(rs):
    acc = np.zeros(N_FEATURES, dtype=np.float64)
    for i in range(0, len(rs), 400):
        c = np.concatenate([np.asarray(cols[indptr[r]:indptr[r + 1]])
                            for r in rs[i:i + 400]])
        acc += np.bincount(c, minlength=N_FEATURES)
        del c
    return acc


def chi_from(a, b, pR):
    ""Carrier-only chi-square from carrier counts (the score chi2_rank.py uses)."""
    tot = a + b
    eR, eS = pR * tot, (1 - pR) * tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = (a - eR) ** 2 / eR + (b - eS) ** 2 / eS
    chi[~np.isfinite(chi)] = 0.0
    return chi


def chi2_carrier(rs, ys):
    return chi_from(counts(rs[ys == 1]), counts(rs[ys == 0]),
                    float((ys == 1).mean()))


chi = chi2_carrier(rows, y)
rank = np.argsort(np.argsort(-chi))
print(f"\n  pooled clindamycin ranking (all {len(rows)} labelled genomes, "
      f"carrier-only chi-square):")
for name in ORDER:
    cc = np.array(list(resolved[name].values()), dtype=np.int64)
    best = cc[np.argmax(chi[cc])]
    print(f"    {name:15s} best k-mer chi2 {chi[best]:>7.1f}  "
          f"rank {int(rank[best]):>9,}", flush=True)
del chi, rank

# ------------------- clindamycin-R / erythromycin-S: what no erm gene explains
print(f"\n{'='*76}\nCLINDAMYCIN vs ERYTHROMYCIN PHENOTYPES (training)", flush=True)
both_lab = np.where((clab >= 0) & (elab >= 0))[0]
cR, eR = clab[both_lab] == 1, elab[both_lab] == 1
print(f"  genomes labelled for both: {len(both_lab)}")
print(f"    clinda R / ery R: {int((cR & eR).sum()):5d}    "
      f"clinda R / ery S: {int((cR & ~eR).sum()):5d}")
print(f"    clinda S / ery R: {int((~cR & eR).sum()):5d}    "
      f"clinda S / ery S: {int((~cR & ~eR).sum()):5d}")
grp = both_lab[cR & ~eR]
for tag, g in (("clinda-R / ery-S", grp),
               ("clinda-S / ery-R", both_lab[~cR & eR])):
    if len(g):
        print(f"  {tag} (n={len(g)}): erm(B) {int(carry['erm(B)'][g].sum())}, "
              f"erm(C) {int(carry['erm(C)'][g].sum())}, "
              f"mecA {int(meca[g].sum())}", flush=True)

if len(grp) >= MIN_GROUP:
    print(f"\n  ranking features: clinda-R/ery-S (n={len(grp)}) "
          f"vs all clindamycin-susceptible", flush=True)
    susc = np.where(clab == 0)[0]
    a_cnt, b_cnt = counts(grp), counts(susc)
    chi = chi_from(a_cnt, b_cnt, len(grp) / (len(grp) + len(susc)))
    top = np.argsort(-chi)[:500]
    need = {int(c) for c in top}
    seq_of = {}
    with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
        next(f)
        for line in f:
            i_str, s = line.rstrip("\n").split(",", 1)
            i = int(i_str)
            if i in need:
                seq_of[i] = s
                if len(seq_of) == len(need):
                    break
    col_of = {s: i for i, s in seq_of.items()}

    pool = set(seq_of.values())
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

    used, ctgs = set(), []
    for seed in sorted(pool):
        if seed in used:
            continue
        used.add(seed)
        mem, ctg = [seed], seed
        while True:
            h = take(by_pre, ctg[-(K - 1):], used)
            if not h:
                break
            used.add(h[0]); mem.append(h[0]); ctg += h[1][-1]
        while True:
            h = take(by_suf, ctg[:K - 1], used)
            if not h:
                break
            used.add(h[0]); mem.append(h[0]); ctg = h[1][0] + ctg
        ctgs.append((ctg, mem))

    refk = {}
    for nm, s in ref.items():
        ks = set()
        for p in range(len(s) - K + 1):
            ks.add(s[p:p + K]); ks.add(rc(s[p:p + K]))
        refk[nm] = ks
    recs = []
    for ctg, mem in ctgs:
        best = max((col_of[m] for m in mem), key=lambda c: chi[c])
        ks = set(mem) | {rc(m) for m in mem}
        hits = sorted((len(ks & v) // 2, nm) for nm, v in refk.items() if ks & v)[-2:]
        lab = "; ".join(f"{nm}:{c}" for c, nm in reversed(hits)) or "UNLABELLED"
        recs.append((chi[best], ctg, a_cnt[best] / len(grp),
                     b_cnt[best] / len(susc), lab))
    recs.sort(key=lambda t: -t[0])
    print(f"  top 500 features -> {len(recs)} contigs")
    print(f"    {'#':>3} {'bp':>6} {'chi2':>7} {'in grp':>7} {'in S':>6}  matches")
    for i, (c2, ctg, fg, fs, lab) in enumerate(recs[:12]):
        print(f"    {i+1:>3} {len(ctg):>6,} {c2:>7.1f} {fg:>6.1%} {fs:>6.1%}  "
              f"{lab[:44]}")
    fn = f"{BASE}/interpret2/clindaR_eryS.fasta"
    with open(fn, "w") as out:
        for i, (c2, ctg, fg, fs, lab) in enumerate(recs[:20]):
            out.write(f">clindaR_eryS_rank{i+1}_len{len(ctg)}_chi2{c2:.0f}"
                      f"_grp{fg:.3f}_S{fs:.3f}_{lab.split(':')[0]}\n")
            for p in range(0, len(ctg), 70):
                out.write(ctg[p:p + 70] + "\n")
    print(f"  wrote {fn}", flush=True)
    del chi
else:
    print(f"  clinda-R/ery-S group below {MIN_GROUP}; not ranked", flush=True)


# ------------------------------------------------------- external genomes
def accession_of(path):
    for part in path.split(os.sep):
        if part.startswith(("GCA_", "GCF_")):
            return part
    return os.path.basename(path)


def read_genome(path):
    parts, buf = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if buf:
                    parts.append("".join(buf)); buf = []
            else:
                buf.append(line.strip())
    if buf:
        parts.append("".join(buf))
    return "N".join(parts).upper()


probes = {n: [(km, rc(km)) for _, _, km in sample[n]] for n in ORDER}
files = sorted(glob.glob(f"{EXT}/raw/**/*.fna", recursive=True))
print(f"\n{'='*76}\nEXTERNAL (NCBI): scanning {len(files)} assemblies", flush=True)
ext_carry = {n: {} for n in ORDER}
for i, path in enumerate(files):
    acc = accession_of(path)
    g = read_genome(path)
    for n in ORDER:
        hit = sum(1 for f_, r_ in probes[n] if f_ in g or r_ in g)
        ext_carry[n][acc] = hit / len(probes[n]) >= FRAC
    if i % 200 == 0:
        print(f"  [{i+1}/{len(files)}]", flush=True)

dup = {}
with open(f"{EXT}/dedup_report.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        dup[r["assembly"]] = r["flagged_duplicate"] == "1"
pheno = defaultdict(dict)
with open(f"{BASE}/external_phenotypes.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["phenotype"] in ("Resistant", "Susceptible"):
            pheno[r["antibiotic"]][r["assembly"]] = 1 if r["phenotype"] == "Resistant" else 0


def ext_arrays(abx, dedup):
    accs = [a for a in pheno.get(abx, {}) if a in ext_carry[ORDER[0]]
            and (not dedup or not dup.get(a, False))]
    yy = np.array([pheno[abx][a] for a in accs])
    car = {n: np.array([ext_carry[n][a] for a in accs]) for n in ORDER}
    return accs, yy, car


res = {}
for dedup in (False, True):
    tag = "deduplicated" if dedup else "all matched"
    for abx in ("clindamycin", "erythromycin"):
        accs, yy, car = ext_arrays(abx, dedup)
        if len(accs) == 0:
            continue
        res[(abx, dedup)] = assoc_table(f"{abx}, {tag}", car, yy)
        if abx == "clindamycin":
            ext_meca = car["mecA (C-term)"] | car["mecA (N-term)"]
            overlap(f"clindamycin, {tag}", car["erm(B)"], car["erm(C)"], yy)
            overlap(f"clindamycin, {tag}, mecA-negative",
                    car["erm(B)"][~ext_meca], car["erm(C)"][~ext_meca],
                    yy[~ext_meca])

caccs = set(pheno.get("clindamycin", {})) & set(pheno.get("erythromycin", {}))
caccs = [a for a in caccs if a in ext_carry[ORDER[0]] and not dup.get(a, False)]
rs_grp = [a for a in caccs if pheno["clindamycin"][a] == 1 and pheno["erythromycin"][a] == 0]
print(f"\n  external clinda-R / ery-S (deduplicated): {len(rs_grp)} of "
      f"{len(caccs)} with both labels; erm(B) "
      f"{sum(ext_carry['erm(B)'][a] for a in rs_grp)}, erm(C) "
      f"{sum(ext_carry['erm(C)'][a] for a in rs_grp)}", flush=True)

# ------------------------------------------------------------------ summary
print(f"\n{'='*76}\nSUMMARY: clindamycin odds ratio, training -> external\n")
print(f"  {'locus':15s} {'training':>18}   {'external (all)':>18}   "
      f"{'external (dedup)':>18}")
for n in ORDER:
    t = tr_clinda[n]
    ea = res.get(("clindamycin", False), {}).get(n)
    ed = res.get(("clindamycin", True), {}).get(n)
    fmt = lambda v: f"{v[0]:7.1f} [{v[1]:.1f},{v[2]:.1f}]" if v else "--"
    print(f"  {n:15s} {fmt(t):>18}   {fmt(ea):>18}   {fmt(ed):>18}")
print("\n  sanity: mecA (C-term) should read ~16.8 -> ~1.5 and erm(C) ~4.9 -> ~162 "
      "(training -> external all, as reported by linkage_external.py)")
ed = res.get(("clindamycin", True), {}).get("erm(B)")
if ed:
    verdict = ("consistent with causal (external CI excludes 1)" if ed[1] > 1
               else "NOT confirmed (external CI includes 1)")
    print(f"\n  pre-stated prediction for erm(B): external clindamycin OR CI "
          f"excludes 1 -> {verdict}")
