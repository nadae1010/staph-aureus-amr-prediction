"""
How much of the exclusion effect is just XGBoost?

clindamycin_v2_loci and clindamycin_v2_phi exclude sets differing by 7 k-mers
out of 119,795, keep the same 200,000 features, and reach the same top
chi-square of 307.4. Their external AUROCs are 0.783 and 0.758 -- a gap of
0.025, wider than the +0.017 that the paired bootstrap called significant.

Two near-identical models cannot really differ by that much, so the gap is
fitting variance: with subsample 0.8 and colsample_bytree 0.8, a trivial change
to the input column set reshuffles which features each tree sees. The bootstrap
could not detect this because it resamples external genomes while holding the
fitted model fixed; it measures evaluation variance only.

This measures the missing term directly. Each feature set is refit with ten
random seeds -- nothing else changed -- and scored on the same external
genomes. The spread across seeds is the noise floor that any exclusion effect
has to clear.

If the spread is comparable to the differences between exclusion variants, then
those differences carry no information and the exclusion experiments are a
negative result: at this feature count, fitting variance exceeds the effect of
removing the confounder. That is itself worth reporting, and it would apply to
any k-mer model of this shape, not only to this one.

No re-featurization is needed. Each variant's kmers.txt fixes the column order,
and the external matrix already holds those columns under the same order, so
the saved presence vectors can be scored directly.

Usage:  python3 seed_variance.py [n_seeds=10] [variant ...]
Output: printed per-seed AUROCs and their spread
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score

BASE = "/blue/boucher/nelseifi"
MAT = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
N_FEATURES = 11_351_471

NSEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
VARIANTS = sys.argv[2:] or ["clindamycin", "clindamycin_v2_loci",
                            "clindamycin_v2_phi"]

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")

# --------------------------------------------------------------- external side
d = np.load(f"{BASE}/external/external_features.npz", allow_pickle=True)
ext = d["ext"]
accs = [str(a) for a in d["accessions"]]
idx_of = {a: i for i, a in enumerate(accs)}

dup = {}
with open(f"{BASE}/external/dedup_report.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        dup[r["assembly"]] = r["flagged_duplicate"] == "1"

pheno = defaultdict(dict)
with open(f"{BASE}/external_phenotypes.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        pheno[r["antibiotic"]][r["assembly"]] = r["phenotype"]

# ------------------------------- every variant's k-mer list, resolved in one scan
want = {}
order = {}
for v in VARIANTS:
    kl = [l.strip() for l in open(f"{BASE}/models/{v}/kmers.txt") if l.strip()]
    order[v] = kl
    for j, s in enumerate(kl):
        want.setdefault(s, []).append((v, j))
print(f"{len(want):,} distinct k-mers across {len(VARIANTS)} variants",
      flush=True)

col_of = {v: np.full(len(order[v]), -1, dtype=np.int64) for v in VARIANTS}
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        hits = want.get(s)
        if hits:
            for v, j in hits:
                col_of[v][j] = int(i_str)
for v in VARIANTS:
    miss = int((col_of[v] < 0).sum())
    print(f"  {v:28s} {len(order[v]):>7,} k-mers, {miss} unresolved", flush=True)


def submatrix(rows, keep):
    """CSR over `keep` columns, IN THE GIVEN ORDER (matches kmers.txt)."""
    lut = np.full(N_FEATURES, -1, dtype=np.int32)
    lut[keep] = np.arange(len(keep), dtype=np.int32)
    data, ind, ptr = [], [], [0]
    for r in rows:
        c = np.asarray(cols[indptr[r]:indptr[r + 1]])
        mk = lut[c] >= 0
        ind.append(lut[c[mk]])
        data.append(np.ones(int(mk.sum()), dtype=np.float32))
        ptr.append(ptr[-1] + int(mk.sum()))
    return csr_matrix((np.concatenate(data), np.concatenate(ind),
                       np.array(ptr)), shape=(len(rows), len(keep)))


summary = {}
for v in VARIANTS:
    abx = "clindamycin" if v.startswith("clindamycin") else "erythromycin"
    labels = np.load(f"{BASE}/labels_{abx}.npy")
    trows = np.where(labels >= 0)[0]
    ytr = labels[trows]
    X = submatrix(trows, col_of[v])

    erows, ye = [], []
    for acc, lab in pheno.get(abx, {}).items():
        i = idx_of.get(acc)
        if i is not None and lab in ("Resistant", "Susceptible") and not dup[acc]:
            erows.append(i)
            ye.append(1 if lab == "Resistant" else 0)
    erows, ye = np.array(erows), np.array(ye)
    Xe = ext[np.ix_(erows, d[f"model_{v}"])].astype(np.float32)
    De = xgb.DMatrix(Xe, missing=0)

    print(f"\n{'='*70}\n{v}\n  train {X.shape}, external n={len(ye)} "
          f"({ye.mean():.1%} R)", flush=True)

    aucs, f1s = [], []
    for s in range(NSEEDS):
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=6,
                                learning_rate=0.1, subsample=0.8,
                                colsample_bytree=0.8, eval_metric="logloss",
                                n_jobs=4, random_state=s, tree_method="hist")
        clf.fit(X, ytr)
        p = clf.get_booster().predict(De)
        a = roc_auc_score(ye, p)
        f = f1_score(ye, (p >= 0.5).astype(int), zero_division=0)
        aucs.append(a)
        f1s.append(f)
        print(f"    seed {s}: AUROC {a:.4f}  F1 {f:.4f}", flush=True)
    aucs = np.array(aucs)
    summary[v] = aucs
    print(f"  mean {aucs.mean():.4f}  sd {aucs.std(ddof=1):.4f}  "
          f"min {aucs.min():.4f}  max {aucs.max():.4f}  "
          f"range {aucs.max()-aucs.min():.4f}", flush=True)

print(f"\n{'='*70}\nnoise floor vs measured effects")
for v, a in summary.items():
    print(f"  {v:28s} {a.mean():.4f} +/- {a.std(ddof=1):.4f}  "
          f"[{a.min():.4f}, {a.max():.4f}]")
if len(summary) > 1:
    base = VARIANTS[0]
    for v in VARIANTS[1:]:
        diff = summary[v].mean() - summary[base].mean()
        pooled = np.sqrt((summary[v].var(ddof=1) +
                          summary[base].var(ddof=1)) / 2)
        print(f"  {v} - {base}: {diff:+.4f}  "
              f"(pooled seed sd {pooled:.4f}, "
              f"{abs(diff)/pooled:.1f} sd)" if pooled > 0 else "")
