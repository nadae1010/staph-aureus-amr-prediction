"""
Error bars for every external figure.

Clindamycin's external AUROC was measured across ten random seeds and moved
0.758 to 0.804 (mean 0.784, sd 0.014) with nothing changed but the seed. The
reported 0.758 turned out to be the minimum of the ten. The other four
antibiotics are still single fits, so their headline numbers carry no
statement of uncertainty at all.

That gap matters in both directions. If ciprofloxacin and cefoxitin prove
stable, as their far smaller feature sets suggest they should, the table can be
reported with tight intervals and the four successes are firmer than they look
today. If any of them wobbles like clindamycin, a figure currently presented as
a point estimate is not one.

Each model is refit with N seeds -- no other change -- and scored on the same
deduplicated external genomes. Everything else is held fixed: the feature set
comes from the model's own kmers.txt, which also fixes the column order the
external matrix already stores, so nothing needs re-featurising.

Brier score is reported alongside AUROC. AUROC measures ranking only; Brier
measures whether the predicted probabilities are themselves accurate, which is
what a threshold-based decision would actually depend on. Lower is better.

Usage:  python3 seed_variance_all.py [n_seeds=10] [variant ...]
        default variants: the five externally testable baseline models
Output: printed per-seed values and summary, plus
        /blue/boucher/nelseifi/external/seed_variance.tsv
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score

BASE = "/blue/boucher/nelseifi"
MAT = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
N_FEATURES = 11_351_471

NSEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
VARIANTS = sys.argv[2:] or ["ciprofloxacin", "oxacillin", "cefoxitin",
                            "erythromycin", "clindamycin"]

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")

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

# ------------------------------- each variant's k-mers, resolved in one scan
order, abx_of = {}, {}
want = {}
for v in VARIANTS:
    meta = json.load(open(f"{BASE}/models/{v}/meta.json"))
    abx_of[v] = meta.get("antibiotic", v)
    kl = [l.strip() for l in open(f"{BASE}/models/{v}/kmers.txt") if l.strip()]
    order[v] = kl
    for j, s in enumerate(kl):
        want.setdefault(s, []).append((v, j))
    print(f"{v:16s} -> {abx_of[v]:14s} {len(kl):>7,} k-mers", flush=True)
print(f"\n{len(want):,} distinct k-mers across {len(VARIANTS)} models",
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
    if miss:
        print(f"  WARNING {v}: {miss} k-mers unresolved", flush=True)


def submatrix(rows, keep):
    """CSR over `keep` columns in the given order, matching kmers.txt."""
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


rows_out = []
for v in VARIANTS:
    abx = abx_of[v]
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
    if not erows:
        print(f"\n{v}: no external phenotypes matched, skipped", flush=True)
        continue
    erows, ye = np.array(erows), np.array(ye)
    Xe = ext[np.ix_(erows, d[f"model_{v}"])].astype(np.float32)
    De = xgb.DMatrix(Xe, missing=0)

    print(f"\n{'='*72}\n{v}  train {X.shape}  external n={len(ye)} "
          f"({ye.mean():.1%} R)", flush=True)

    aucs, briers, f1s = [], [], []
    for s in range(NSEEDS):
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=6,
                                learning_rate=0.1, subsample=0.8,
                                colsample_bytree=0.8, eval_metric="logloss",
                                n_jobs=4, random_state=s, tree_method="hist")
        clf.fit(X, ytr)
        p = clf.get_booster().predict(De)
        a = roc_auc_score(ye, p)
        b = brier_score_loss(ye, p)
        f = f1_score(ye, (p >= 0.5).astype(int), zero_division=0)
        aucs.append(a)
        briers.append(b)
        f1s.append(f)
        print(f"  seed {s}: AUROC {a:.4f}  Brier {b:.4f}  F1 {f:.4f}",
              flush=True)

    aucs, briers, f1s = np.array(aucs), np.array(briers), np.array(f1s)
    print(f"  AUROC  mean {aucs.mean():.4f}  sd {aucs.std(ddof=1):.4f}  "
          f"[{aucs.min():.4f}, {aucs.max():.4f}]  range {aucs.ptp():.4f}")
    print(f"  Brier  mean {briers.mean():.4f}  sd {briers.std(ddof=1):.4f}")
    print(f"  F1     mean {f1s.mean():.4f}  sd {f1s.std(ddof=1):.4f}",
          flush=True)
    rows_out.append((v, abx, len(ye), len(order[v]), aucs, briers, f1s))

print(f"\n{'='*72}\nsummary — external AUROC over {NSEEDS} seeds\n")
print(f"  {'model':16s} {'n':>5} {'features':>9} {'mean':>7} {'sd':>7} "
      f"{'min':>7} {'max':>7}")
for v, abx, n, nf, aucs, briers, f1s in rows_out:
    print(f"  {v:16s} {n:>5} {nf:>9,} {aucs.mean():>7.4f} "
          f"{aucs.std(ddof=1):>7.4f} {aucs.min():>7.4f} {aucs.max():>7.4f}")

with open(f"{BASE}/external/seed_variance.tsv", "w") as fh:
    fh.write("model\tantibiotic\tn_external\tn_features\tauroc_mean\tauroc_sd\t"
             "auroc_min\tauroc_max\tbrier_mean\tbrier_sd\tf1_mean\tf1_sd\n")
    for v, abx, n, nf, aucs, briers, f1s in rows_out:
        fh.write(f"{v}\t{abx}\t{n}\t{nf}\t{aucs.mean():.4f}\t"
                 f"{aucs.std(ddof=1):.4f}\t{aucs.min():.4f}\t{aucs.max():.4f}\t"
                 f"{briers.mean():.4f}\t{briers.std(ddof=1):.4f}\t"
                 f"{f1s.mean():.4f}\t{f1s.std(ddof=1):.4f}\n")
print(f"\nwrote {BASE}/external/seed_variance.tsv")
