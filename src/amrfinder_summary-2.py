"""
AMRFinderPlus follow-up: which catalogued genes explain clindamycin resistance,
and what fills the gap that erm(B) and erm(C) leave?

The k-mer analyses found that erm(B) and erm(C) together account for about
half of training and a quarter of external clindamycin resistance, and that
most of the remainder is resistant to erythromycin too -- the pattern of an
MLS_B gene the pipeline never surfaced. This script reads NCBI AMRFinderPlus
calls (run_amrfinder.sh; software 4.2.7, database 2026-10-29.1) on every
training and external genome and asks:

  1. SANITY. Do the gene calls line up with the matrix rows and reproduce the
     k-mer findings? mecA should be in ~95% of cefoxitin-resistant and ~15% of
     susceptible genomes; for clindamycin, mecA's odds ratio should collapse
     externally (~16.8 -> ~1.5) and erm(C)'s should rise (~4.9 -> ~162). If
     not, the rows are misaligned and nothing below should be trusted.
  2. MLS GENES. Every macrolide/lincosamide/streptogramin element found, with
     its clindamycin odds ratio in training and external genomes.
  3. COVERAGE. What fraction of clindamycin-resistant genomes carry any known
     lincosamide determinant, and which genes fill the erm(B)/erm(C) gap.
  4. GENE SCAN. The top AMRFinderPlus elements by 2x2 chi-square for each
     drug -- the database's answer to the question the k-mer ranking asked.
  5. LINKAGE. For each gap-filling gene: does it travel with mecA, and with
     what else? Hypothesis stated in advance: if erm(A) fills the gap, it is
     the copy on Tn554 inside SCCmec type II, so its carriers should be mostly
     mecA-positive and it should co-occur with spc, ant(9)-Ia.
  6. ERYTHROMYCIN. The same coverage question for macrolide determinants.

A lincosamide determinant is any element whose AMRFinderPlus class or subclass
includes LINCOSAMIDE, plus every erm gene (all erm methylases confer MLS_B
resistance, inducibly or constitutively). A macrolide determinant is the same
with MACROLIDE, excluding lmrS, a chromosomal efflux pump carried by ~94% of
genomes whatever their phenotype. Any call counts as carriage,
including partial hits at contig ends. Odds ratios use the Haldane +0.5
correction with Woolf 95% intervals, as in erm_followup.py.

Can be run while the AMRFinderPlus job is still going; section 0 says how
complete the input is.

Usage:  python3 amrfinder_summary.py
Output: printed report; data/amrfinder_calls.tsv, data/amrfinder_elements.tsv
"""
import csv
import hashlib
import os
from collections import Counter, defaultdict

import numpy as np

BASE = "/blue/boucher/nelseifi"
PROJ = os.path.expanduser("~/staph_amr_project")
TRAIN_LIST = f"{PROJ}/genome_fasta_list_fasta.txt"
ALL_LIST = f"{PROJ}/amrfinder/genomes.txt"
OUTDIR = f"{PROJ}/amrfinder/out"
EXT = f"{BASE}/external"
DRUGS = ["cefoxitin", "oxacillin", "methicillin", "ciprofloxacin",
         "clindamycin", "erythromycin"]
KNOWN = {"erm(B)", "erm(C)"}
LINE = "=" * 78


def out_file(path):
    # must match run_amrfinder.sh: printf '%s' "$f" | md5sum | cut -c1-16
    return f"{OUTDIR}/{hashlib.md5(path.encode()).hexdigest()[:16]}.tsv"


def accession_of(path):
    for part in path.split(os.sep):
        if part.startswith(("GCA_", "GCF_")):
            return part
    return os.path.basename(path)


def read_list(fn):
    with open(fn) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


info = {}                      # symbol -> (type, subtype, class, subclass)
methods = defaultdict(Counter)


def calls(path):
    """Set of element symbols for one genome, or None if not run yet."""
    fn = out_file(path)
    if not os.path.exists(fn):
        return None
    found = set()
    with open(fn) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sym = r.get("Element symbol") or r.get("Gene symbol")
            if not sym:
                continue
            found.add(sym)
            info.setdefault(sym, (r.get("Type", ""), r.get("Subtype", ""),
                                  r.get("Class", ""), r.get("Subclass", "")))
            methods[sym][r.get("Method", "")] += 1
    return found


# ------------------------------------------------------------------ load calls
train_paths = read_list(TRAIN_LIST)
train_set = set(train_paths)
ext_paths = [p for p in read_list(ALL_LIST) if p not in train_set]

tr = [calls(p) for p in train_paths]
tr_done = np.array([c is not None for c in tr])
tr = [c if c is not None else set() for c in tr]

ext, ext_files = {}, Counter()
for p in ext_paths:
    a = accession_of(p)
    ext_files[a] += 1
    c = calls(p)
    if c is not None:
        ext.setdefault(a, set()).update(c)

dup = {}
with open(f"{EXT}/dedup_report.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        dup[r["assembly"]] = r["flagged_duplicate"] == "1"
pheno = defaultdict(dict)
with open(f"{BASE}/external_phenotypes.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["phenotype"] in ("Resistant", "Susceptible"):
            pheno[r["antibiotic"]][r["assembly"]] = int(r["phenotype"] == "Resistant")

_labels = {}


def train_labels(abx):
    if abx not in _labels:
        fn = f"{BASE}/labels_{abx}.npy"
        lab = np.load(fn) if os.path.exists(fn) else None
        if lab is not None and len(lab) != len(train_paths):
            raise SystemExit(f"{fn} has {len(lab)} rows but the genome list has "
                             f"{len(train_paths)}; row order unknown, stopping")
        _labels[abx] = lab
    return _labels[abx]


def data(abx, where):
    """(y, element sets, ids) for genomes with a phenotype and AMRFinderPlus
    output. where: 'train', 'ext' (deduplicated) or 'ext_all'."""
    if where == "train":
        lab = train_labels(abx)
        if lab is None:
            return None
        rows = np.where((lab >= 0) & tr_done)[0]
        return lab[rows].astype(int), [tr[i] for i in rows], list(rows)
    accs = sorted(a for a in pheno.get(abx, {}) if a in ext
                  and (where == "ext_all" or not dup.get(a, False)))
    if not accs:
        return None
    return np.array([pheno[abx][a] for a in accs]), [ext[a] for a in accs], accs


# ---------------------------------------------------------------- statistics
def has(sets, sym):
    return np.fromiter((sym in s for s in sets), dtype=bool, count=len(sets))


def anyof(sets, syms):
    return np.fromiter((bool(s & syms) for s in sets), dtype=bool, count=len(sets))


def odds(car, y):
    a = int((car & (y == 1)).sum())
    b = int((car & (y == 0)).sum())
    c = int((~car & (y == 1)).sum())
    d = int((~car & (y == 0)).sum())
    o = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
    se = np.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (d + .5))
    return a, b, c, d, o, np.exp(np.log(o) - 1.96 * se), np.exp(np.log(o) + 1.96 * se)


def chi2x2(car, y):
    a, b, c, d = odds(car, y)[:4]
    den = (a + b) * (c + d) * (a + c) * (b + d)
    return (a + b + c + d) * (a * d - b * c) ** 2 / den if den else 0.0


def fmt_or(v):
    return f"{v[4]:7.1f} [{v[5]:.1f}, {v[6]:.1f}]"


def pct(k, n):
    return f"{int(k):>4}/{int(n):<4} {k / max(n, 1):6.1%}"


def cls_text(sym):
    return f"{info[sym][2]}/{info[sym][3]}".upper()


def is_linco(sym):
    return sym.startswith("erm(") or "LINCOSAMIDE" in cls_text(sym)


INTRINSIC = {"lmrS"}   # chromosomal efflux pump in ~94% of genomes; not acquired


def is_macro(sym):
    if sym in INTRINSIC:
        return False
    return sym.startswith("erm(") or "MACROLIDE" in cls_text(sym)


def is_mls(sym):
    t = cls_text(sym)
    return sym.startswith("erm(") or any(
        k in t for k in ("LINCOSAMIDE", "MACROLIDE", "STREPTOGRAMIN"))


LINCO = {s for s in info if is_linco(s)}
MACRO = {s for s in info if is_macro(s)}
WHERE = (("train", "training (BV-BRC)"), ("ext", "external (NCBI, deduplicated)"))

# ------------------------------------------------------------------ 0. coverage
print(f"{LINE}\n0. COVERAGE")
print(f"  training: {int(tr_done.sum())}/{len(train_paths)} genomes have output")
print(f"  external: {len(ext)}/{len(ext_files)} accessions have output "
      f"({len(ext_paths)} files)")
print(f"  distinct elements called: {len(info)}")
if tr_done.sum() < len(train_paths) or len(ext) < len(ext_files):
    print("  ** input incomplete: the AMRFinderPlus job is still running or some "
          "genomes failed; treat every number below as provisional **")

# -------------------------------------------------------------------- 1. sanity
print(f"\n{LINE}\n1. SANITY: do the gene calls reproduce the k-mer findings?")
for abx in ("cefoxitin", "oxacillin", "methicillin"):
    d = data(abx, "train")
    if d:
        y, s, _ = d
        a, b, c, dd = odds(has(s, "mecA"), y)[:4]
        print(f"  {abx:12s} training: mecA in R {pct(a, a + c)}   in S {pct(b, b + dd)}"
              + ("   (k-mer pipeline: 94.7% vs 15.1%)" if abx == "cefoxitin" else ""))
T_all, E_all = data("clindamycin", "train"), data("clindamycin", "ext_all")
for sym, expect in (("mecA", "~16.8 -> ~1.5"), ("erm(C)", "~4.9 -> ~162")):
    if T_all and E_all:
        print(f"  clindamycin OR {sym:7s} training {fmt_or(odds(has(T_all[1], sym), T_all[0]))}"
              f" -> external (all) {fmt_or(odds(has(E_all[1], sym), E_all[0]))}"
              f"   (k-mer: {expect})")
print("  (gene calls and k-mer carriage are different detectors, so expect the "
      "same direction and rough size, not identical numbers)")

# ---------------------------------------------------------------- 2. MLS genes
print(f"\n{LINE}\n2. EVERY MLS ELEMENT FOUND, with its clindamycin association")
T, E = data("clindamycin", "train"), data("clindamycin", "ext")
proc = [g for g, ok in zip(tr, tr_done) if ok]
mls = sorted((s for s in info if is_mls(s)), key=lambda s: -sum(s in g for g in proc))
print(f"  {'element':13s} {'n train':>7}  {'training R carry':>17} {'S carry':>15} "
      f"{'OR [95% CI]':>20}   {'external R carry':>17} {'S carry':>15} {'OR [95% CI]':>20}")
for sym in mls:
    row = f"  {sym:13s} {sum(sym in g for g in proc):>7}  "
    for d in (T, E):
        if d:
            a, b, c, dd, o, lo, hi = odds(has(d[1], sym), d[0])
            row += f"{pct(a, a + c):>17} {pct(b, b + dd):>15} {fmt_or((0, 0, 0, 0, o, lo, hi)):>20}   "
    print(row)
    print(f"  {'':13s} class: {info[sym][2]} / {info[sym][3]}   "
          f"lincosamide determinant: {'yes' if sym in LINCO else 'no'}")

# ---------------------------------------------------------------- 3. coverage
print(f"\n{LINE}\n3. CLINDAMYCIN: how much resistance do catalogued genes explain?")
for where, label in WHERE:
    d = data("clindamycin", where)
    if not d:
        continue
    y, sets, ids = d
    R, S = y == 1, y == 0
    known, anyl = anyof(sets, KNOWN), anyof(sets, LINCO)
    print(f"\n  {label}: R={int(R.sum())}, S={int(S.sum())}")
    print(f"    R carrying erm(B) or erm(C):            {pct((known & R).sum(), R.sum())}"
          f"   (k-mer estimate: ~51% training / ~25% external)")
    print(f"    R carrying any lincosamide determinant: {pct((anyl & R).sum(), R.sum())}")
    print(f"    S carrying any lincosamide determinant: {pct((anyl & S).sum(), S.sum())}"
          f"   (erm genes are often inducible; without a D-test these read susceptible)")
    gap = R & ~known
    cnt = Counter(e for g, m in zip(sets, gap) if m for e in g if e in LINCO)
    Ssets = [g for g, m in zip(sets, S) if m]
    print(f"    gap = {int(gap.sum())} resistant genomes without erm(B)/erm(C); "
          f"lincosamide determinants carried there:")
    for sym, k in cnt.most_common(8):
        print(f"      {sym:13s} in gap {pct(k, gap.sum())}   in S {pct(has(Ssets, sym).sum(), len(Ssets))}")
    if not cnt:
        print("      none")
    un = R & ~anyl
    ery = (train_labels("erythromycin") if where == "train" else None)
    tag = Counter()
    for i, m in zip(ids, un):
        if m:
            v = (ery[i] if ery is not None else
                 pheno.get("erythromycin", {}).get(i, -1))
            tag["ery R" if v == 1 else "ery S" if v == 0 else "no ery label"] += 1
    print(f"    still unexplained (R, no lincosamide determinant): {pct(un.sum(), R.sum())}"
          f"  -> {dict(tag)}, mecA+ {int((un & has(sets, 'mecA')).sum())}")
    if un.sum() >= 10:
        sc = []
        for sym in info:
            car = has(sets, sym)
            if car.sum() >= 5:
                sub = un | S
                sc.append((chi2x2(car[sub], (un[sub]).astype(int)), sym,
                           car[un].mean(), car[S].mean()))
        sc.sort(reverse=True)
        print("    elements most enriched in the unexplained group vs susceptible:")
        for c2, sym, fu, fs in sc[:6]:
            print(f"      {sym:13s} chi2 {c2:7.1f}   unexplained {fu:6.1%}   S {fs:6.1%}   "
                  f"[{info[sym][0]} {info[sym][2]}]")

# ---------------------------------------------------------------- 4. gene scan
print(f"\n{LINE}\n4. TOP AMRFinderPlus ELEMENTS PER DRUG (2x2 chi-square)")
for abx in DRUGS:
    for where, label in WHERE:
        d = data(abx, where)
        if not d:
            continue
        y, sets, _ = d
        minc = 10 if where == "train" else 5
        sc = []
        for sym in info:
            car = has(sets, sym)
            if car.sum() >= minc:
                sc.append((chi2x2(car, y), sym, odds(car, y)))
        sc.sort(key=lambda t: -t[0])
        print(f"\n  {abx}, {label} (R={int((y == 1).sum())}, S={int((y == 0).sum())})")
        for c2, sym, o in sc[:6]:
            a, b, c, dd = o[:4]
            print(f"    {sym:13s} chi2 {c2:7.1f}  R {a / max(a + c, 1):6.1%}  "
                  f"S {b / max(b + dd, 1):6.1%}  OR {fmt_or(o)}")

# ------------------------------------------------------------------ 5. linkage
print(f"\n{LINE}\n5. LINKAGE: what do the gap-filling genes travel with?")
ext_dd = [g for a, g in ext.items() if not dup.get(a, False)]
cands = sorted((s for s in LINCO - KNOWN if sum(s in g for g in proc) >= 10),
               key=lambda s: -sum(s in g for g in proc))
if "erm(A)" in info and "erm(A)" not in cands:
    cands.insert(0, "erm(A)")
if not cands:
    print("  no other lincosamide determinant carried by >= 10 training genomes")
for sym in cands:
    for label, sets in (("training", proc), ("external", ext_dd)):
        x, m = has(sets, sym), has(sets, "mecA")
        if x.sum() == 0:
            print(f"\n  {sym}, {label}: not carried")
            continue
        print(f"\n  {sym}, {label}: carried by {int(x.sum())} of {len(sets)}")
        print(f"    mecA among carriers {pct((x & m).sum(), x.sum())}   "
              f"carriage in mecA+ {pct((x & m).sum(), m.sum())}   "
              f"in mecA- {pct((x & ~m).sum(), (~m).sum())}")
        jac = []
        for o in info:
            if o == sym:
                continue
            oc = has(sets, o)
            inter = int((x & oc).sum())
            if inter >= 5:
                jac.append((inter / int((x | oc).sum()), o, inter, int(oc.sum())))
        jac.sort(reverse=True)
        print("    travels with (Jaccard):  " + ",  ".join(
            f"{o} {j:.2f} ({i}/{n})" for j, o, i, n in jac[:6]))

# ------------------------------------------------------------- 6. erythromycin
print(f"\n{LINE}\n6. ERYTHROMYCIN: coverage by macrolide determinants")
for where, label in WHERE:
    d = data("erythromycin", where)
    if not d:
        continue
    y, sets, _ = d
    R, S = y == 1, y == 0
    anym = anyof(sets, MACRO)
    print(f"  {label}: R={int(R.sum())}  R with any macrolide determinant "
          f"{pct((anym & R).sum(), R.sum())}   S with one {pct((anym & S).sum(), S.sum())}")

# ------------------------------------------------------------------- outputs
os.makedirs(f"{PROJ}/data", exist_ok=True)
with open(f"{PROJ}/data/amrfinder_calls.tsv", "w") as f:
    f.write("set\tgenome\tmatrix_row\tn_elements\telements\n")
    for i, (p, g, ok) in enumerate(zip(train_paths, tr, tr_done)):
        if ok:
            gid = os.path.basename(p).rsplit(".", 1)[0]
            f.write(f"training\t{gid}\t{i}\t{len(g)}\t{','.join(sorted(g))}\n")
    for a in sorted(ext):
        f.write(f"external{'_dup' if dup.get(a, False) else ''}\t{a}\t\t"
                f"{len(ext[a])}\t{','.join(sorted(ext[a]))}\n")
with open(f"{PROJ}/data/amrfinder_elements.tsv", "w") as f:
    f.write("element\ttype\tsubtype\tclass\tsubclass\ttraining_carriers\t"
            "external_carriers_dedup\tmethods\n")
    for sym in sorted(info, key=lambda s: -sum(s in g for g in proc)):
        f.write(f"{sym}\t" + "\t".join(info[sym]) +
                f"\t{sum(sym in g for g in proc)}\t{sum(sym in g for g in ext_dd)}\t"
                + ",".join(f"{k}:{v}" for k, v in methods[sym].most_common()) + "\n")
print(f"\n{LINE}\nwrote {PROJ}/data/amrfinder_calls.tsv and "
      f"{PROJ}/data/amrfinder_elements.tsv")
