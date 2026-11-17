"""
Is there learnable clindamycin signal once mecA is taken out of the picture?

Two fixes for clindamycin have already failed. Excluding the mecA / beta-lactam
features did not improve external AUROC beyond seed noise, and the standard 2x2
chi-square ranked erm(C) lower, not higher. Both failed for the same reason:
within the training population mecA genuinely is the better predictor of
clindamycin resistance (phi 0.61 vs 0.35 for erm(C)), so any criterion that
optimises prediction over the pooled population selects it.

The standard response to a known confounder is to stratify on it. Within the
mecA-negative genomes nobody carries mecA, so it cannot contribute; within the
mecA-positive genomes everybody does, so it cannot discriminate. A feature that
still predicts clindamycin inside a stratum is doing work the linkage cannot
explain.

Before building stratified selection into the pipeline, this checks whether
there is anything to find. For each stratum it:

  1. ranks features with AURA's carrier-conditioned chi-square, reports where
     erm(C) lands, and assembles the top features into contigs labelled
     against the known loci (unlabelled contigs are BLAST candidates -- other
     MLSb determinants such as erm(A), erm(B) or lnu(A) would surface here);
  2. estimates within-stratum AUROC by 5-fold cross-validation grouped by
     lineage cluster, with feature selection refit inside every fold so the
     test genomes never influence which k-mers are chosen.

Pre-registered decision rule: proceed to stratified selection only if the
mecA-negative stratum reaches mean CV AUROC >= 0.80 at either feature count.
That is the current external mean (0.784): a mecA-free model must at least
match what the confounded one achieves to be worth building.

Training and test matrices are both CSR, so absent k-mers carry the same
"missing" semantics at fit and predict time (the dense-scoring pitfall that
produced the 0.500 AUROCs does not arise here).

Usage:  python3 stratified_clinda.py [n_seeds=3] [top_k ...]   (default 5000 50000)
Output: printed report, plus
        /blue/boucher/nelseifi/interpret2/stratified_<stratum>.fasta
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

BASE = "/blue/boucher/nelseifi"
MAT = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
N_FEATURES = 11_351_471
K = 31
COMP = str.maketrans("ACGT", "TGCA")
ABX = "clindamycin"
FRAC = 0.5
N_ASSEMBLE = 2000
NFOLD = 5

NSEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TOP_KS = [int(a) for a in sys.argv[2:]] or [5_000, 50_000]

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
n_train = len(indptr) - 1


def rc(s):
    return s[::-1].translate(COMP)


# --------------------------------- reference loci: labels and mecA carriage
ref = {}
NAMED = {"methicillin_c1": "mecA (C-term)", "methicillin_c2": "mecA (N-term)",
         "erythromycin_c1": "erm(C)"}
with open(f"{BASE}/interpret2/contigs_all.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        s = r["sequence"]
        kms = set()
        for i in range(len(s) - K + 1):
            km = s[i:i + K]
            kms.add(km)
            kms.add(rc(km))
        ref[f"{r['antibiotic']}_c{r['contig']}"] = kms

want = {}
for key, name in NAMED.items():
    for km in ref.get(key, ()):
        want[km] = name
locus_cols = defaultdict(list)
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        n = want.get(s)
        if n:
            locus_cols[n].append(int(i_str))
for name in NAMED.values():
    print(f"{name:15s} {len(locus_cols.get(name, [])):>6,} matrix columns",
          flush=True)

meca_cols = np.array(sorted(set(locus_cols["mecA (C-term)"]) |
                            set(locus_cols["mecA (N-term)"])), dtype=np.int64)
lut = np.full(N_FEATURES, -1, dtype=np.int32)
lut[meca_cols] = np.arange(len(meca_cols), dtype=np.int32)
pres = np.zeros((n_train, len(meca_cols)), dtype=bool)
for r in range(n_train):
    j = lut[np.asarray(cols[indptr[r]:indptr[r + 1]])]
    pres[r, j[j >= 0]] = True
pos = {c: j for j, c in enumerate(meca_cols)}
meca = np.zeros(n_train, dtype=bool)
for name in ("mecA (C-term)", "mecA (N-term)"):
    sub = pres[:, [pos[c] for c in locus_cols[name]]]
    meca |= sub.mean(axis=1) >= FRAC
print(f"\nmecA carriers (either contig): {int(meca.sum()):,}/{n_train}",
      flush=True)

# ------------------------------------------------------------------ strata
labels = np.load(f"{BASE}/labels_{ABX}.npy")
rows = np.where(labels >= 0)[0]
strata = {"mecA-negative": rows[~meca[rows]],
          "mecA-positive": rows[meca[rows]]}
for name, sr in strata.items():
    sy = labels[sr]
    print(f"  {name:14s} n={len(sr):5d}  R={int((sy == 1).sum()):4d}  "
          f"S={int((sy == 0).sum()):4d}", flush=True)

# --------------------------------------------------- lineage groups for CV
groups = None
try:
    with open(f"{BASE}/lineage_clusters.csv") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []
        recs = list(rd)
    ccol = next((c for c in fields if "cluster" in c.lower()), None)
    icol = next((c for c in fields if c.lower() in
                 ("row", "index", "row_index", "matrix_row")), None)
    if ccol:
        code = {}
        g = np.full(n_train, -1, dtype=np.int64)
        if icol:
            for r in recs:
                g[int(r[icol])] = code.setdefault(r[ccol], len(code))
        elif len(recs) == n_train:
            for i, r in enumerate(recs):
                g[i] = code.setdefault(r[ccol], len(code))
        if (g >= 0).all():
            groups = g
            how = f"column '{icol}'" if icol else "row order"
            print(f"\nlineage groups: {len(code)} clusters "
                  f"(column '{ccol}', aligned by {how})", flush=True)
except FileNotFoundError:
    pass
if groups is None:
    print("\nWARNING: lineage clusters not usable -- falling back to plain "
          "stratified folds, which can leak near-identical genomes", flush=True)


# ----------------------------------------------------------------- helpers
def counts(rs):
    acc = np.zeros(N_FEATURES, dtype=np.float64)
    for i in range(0, len(rs), 400):
        c = np.concatenate([np.asarray(cols[indptr[r]:indptr[r + 1]])
                            for r in rs[i:i + 400]])
        acc += np.bincount(c, minlength=N_FEATURES)
        del c
    return acc


def chi2_carrier(rs, ys):
    """Carrier-only chi-square, as in chi2_rank.py."""
    a, b = counts(rs[ys == 1]), counts(rs[ys == 0])
    tot, pR = a + b, float((ys == 1).mean())
    eR, eS = pR * tot, (1 - pR) * tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = (a - eR) ** 2 / eR + (b - eS) ** 2 / eS
    chi[~np.isfinite(chi)] = 0.0
    return chi, a, b


def submatrix(rs, keep):
    l2 = np.full(N_FEATURES, -1, dtype=np.int32)
    l2[keep] = np.arange(len(keep), dtype=np.int32)
    data, ind, ptr = [], [], [0]
    for r in rs:
        c = np.asarray(cols[indptr[r]:indptr[r + 1]])
        mk = l2[c] >= 0
        ind.append(l2[c[mk]])
        data.append(np.ones(int(mk.sum()), dtype=np.float32))
        ptr.append(ptr[-1] + int(mk.sum()))
    return csr_matrix((np.concatenate(data), np.concatenate(ind),
                       np.array(ptr)), shape=(len(rs), len(keep)))


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

    used, out = set(), []
    for seed in sorted(pool):
        if seed in used:
            continue
        used.add(seed)
        members, contig = [seed], seed
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
        out.append((contig, members))
    return out


# ------------------------------------------- part 1: ranking within strata
erm_cols = np.array(locus_cols.get("erm(C)", []), dtype=np.int64)
ranked = {}
for name, sr in strata.items():
    sy = labels[sr]
    chi, a, b = chi2_carrier(sr, sy)
    nR, nS = float((sy == 1).sum()), float((sy == 0).sum())
    rank = np.argsort(np.argsort(-chi))
    print(f"\n{'='*74}\n{name}: ranking", flush=True)
    if len(erm_cols):
        best = erm_cols[np.argmax(chi[erm_cols])]
        print(f"  erm(C): chi2 {chi[best]:.1f}, rank {int(rank[best]):,}, "
              f"carried by {a[best]/nR:.1%} of R / {b[best]/nS:.1%} of S "
              f"in this stratum", flush=True)
    top = np.argsort(-chi)[:N_ASSEMBLE]
    ranked[name] = (top, chi, a, b, nR, nS, rank)

need = {int(c) for v in ranked.values() for c in v[0]}
seq_of, col_of = {}, {}
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        i = int(i_str)
        if i in need:
            seq_of[i] = s
            col_of[s] = i
            if len(seq_of) == len(need):
                break

for name, (top, chi, a, b, nR, nS, rank) in ranked.items():
    ctgs = assemble({seq_of[int(c)] for c in top})
    recs = []
    for seq, members in ctgs:
        mc = [col_of[m] for m in members]
        best = max(mc, key=lambda c: chi[c])
        kms = set(members) | {rc(m) for m in members}
        hits = sorted((len(kms & kk) // 2, nm) for nm, kk in ref.items()
                      if kms & kk)[-2:]
        lab = "; ".join(f"{nm}:{c}" for c, nm in reversed(hits)) or "UNLABELLED"
        recs.append((chi[best], int(rank[best]), seq, len(members),
                     a[best] / nR, b[best] / nS, lab))
    recs.sort(key=lambda t: -t[0])
    print(f"\n  {name}: top {N_ASSEMBLE:,} features -> {len(recs)} contigs")
    print(f"  {'#':>3} {'bp':>6} {'chi2':>7} {'rank':>7} {'R':>6} {'S':>6}  matches")
    for i, (c2, rk, seq, nk, cR, cS, lab) in enumerate(recs[:15]):
        print(f"  {i+1:>3} {len(seq):>6,} {c2:>7.1f} {rk:>7,} "
              f"{cR:>5.1%} {cS:>5.1%}  {lab[:46]}")
    fn = f"{BASE}/interpret2/stratified_{name}.fasta"
    with open(fn, "w") as out:
        for i, (c2, rk, seq, nk, cR, cS, lab) in enumerate(recs[:25]):
            out.write(f">{name}_rank{i+1}_len{len(seq)}_chi2{c2:.0f}"
                      f"_R{cR:.3f}_S{cS:.3f}_{lab.split(':')[0]}\n")
            for p in range(0, len(seq), 70):
                out.write(seq[p:p + 70] + "\n")
    print(f"  wrote {fn}", flush=True)

# ------------------------------- part 2: within-stratum cross-validation
print(f"\n{'='*74}\nwithin-stratum CV ({NFOLD}-fold, "
      f"{'lineage-grouped' if groups is not None else 'NOT grouped'}, "
      f"{NSEEDS} seeds, selection refit inside each fold)", flush=True)

summary = {}
for name, sr in strata.items():
    sy = labels[sr]
    sg = groups[sr] if groups is not None else None
    aucs = {k: [] for k in TOP_KS}
    for seed in range(NSEEDS):
        if sg is not None:
            splitter = StratifiedGroupKFold(NFOLD, shuffle=True, random_state=seed)
            folds = splitter.split(sr, sy, sg)
        else:
            splitter = StratifiedKFold(NFOLD, shuffle=True, random_state=seed)
            folds = splitter.split(sr, sy)
        for fi, (tr, te) in enumerate(folds):
            if len(set(sy[te].tolist())) < 2 or len(set(sy[tr].tolist())) < 2:
                continue
            chi, _, _ = chi2_carrier(sr[tr], sy[tr])
            order = np.argsort(-chi)
            for k in TOP_KS:
                keep = np.sort(order[:k][chi[order[:k]] > 0])
                Xtr, Xte = submatrix(sr[tr], keep), submatrix(sr[te], keep)
                clf = xgb.XGBClassifier(n_estimators=300, max_depth=6,
                                        learning_rate=0.1, subsample=0.8,
                                        colsample_bytree=0.8,
                                        eval_metric="logloss", n_jobs=2,
                                        random_state=seed, tree_method="hist")
                clf.fit(Xtr, sy[tr])
                p = clf.predict_proba(Xte)[:, 1]
                aucs[k].append(roc_auc_score(sy[te], p))
            print(f"  {name} seed {seed} fold {fi}: " +
                  "  ".join(f"k={k:,} {aucs[k][-1]:.4f}" for k in TOP_KS),
                  flush=True)
    summary[name] = aucs

print(f"\n{'='*74}\nsummary: within-stratum AUROC (mean +/- sd over folds x seeds)\n")
for name, aucs in summary.items():
    for k in TOP_KS:
        v = np.array(aucs[k])
        if len(v):
            print(f"  {name:14s} k={k:>6,}  {v.mean():.4f} +/- {v.std(ddof=1):.4f}"
                  f"  (n={len(v)})")
neg = summary.get("mecA-negative", {})
best = max((np.mean(neg[k]) for k in TOP_KS if neg.get(k)), default=float("nan"))
verdict = ("PASS: proceed to stratified selection" if best >= 0.80 else
           "FAIL: report clindamycin as not adequately predictable without "
           "lineage markers")
print(f"\npre-registered rule (mecA-negative mean >= 0.80 at either k): "
      f"best {best:.4f} -> {verdict}")
