"""
Held-out-source evaluation: train on every source study except one, test on
the held-out study. Measures how much performance survives when the model
cannot recognise the collection a genome came from.

Chi-square + top-K selection are refit on the training genomes only, exactly
as in the random-split pipeline. No validation set, so n_estimators is fixed
(no early stopping) -- results are therefore slightly conservative.

Usage:  python3 source_holdout.py <antibiotic> [top_k=10000] [min_test=100]
Output: /blue/boucher/nelseifi/results_source/<abx>.json
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb

BASE, SUFFIX = "/blue/boucher/nelseifi", "k31_min5_max3000_d0"
MAT, N_FEATURES = f"{BASE}/kmx_output", 11_351_471
abx      = sys.argv[1]
TOP_K    = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
MIN_TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 100

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols   = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
labels = np.load(f"{BASE}/labels_{abx}.npy")

aligned = pd.read_csv(f"{BASE}/phenotypes_matrix_aligned.csv", dtype=str)
aligned["matrix_row"] = aligned["matrix_row"].astype(int)
src = pd.read_csv(f"{BASE}/genome_sources.csv", dtype=str)
src = src[src["Antibiotic"] == abx][["Genome ID", "source"]]
m = aligned.merge(src, on="Genome ID", how="inner")
m = m[labels[m["matrix_row"].values] >= 0]
rows   = m["matrix_row"].values
groups = m["source"].values
y      = labels[rows]
print(f"{abx}: {len(rows)} labelled genomes with a known source, "
      f"{len(set(groups))} sources")

def class_counts(rs):
    acc = np.zeros(N_FEATURES, dtype=np.float64)
    for i in range(0, len(rs), 400):
        c = np.concatenate([np.asarray(cols[indptr[r]:indptr[r+1]]) for r in rs[i:i+400]])
        acc += np.bincount(c, minlength=N_FEATURES)
    return acc

def submatrix(rs, keep):
    lut = np.full(N_FEATURES, -1, dtype=np.int32)
    lut[keep] = np.arange(len(keep), dtype=np.int32)
    data, ind, ptr = [], [], [0]
    for r in rs:
        c = np.asarray(cols[indptr[r]:indptr[r+1]])
        mk = lut[c] >= 0
        ind.append(lut[c[mk]]); data.append(np.ones(int(mk.sum()), dtype=np.float32))
        ptr.append(ptr[-1] + int(mk.sum()))
    return csr_matrix((np.concatenate(data), np.concatenate(ind), np.array(ptr)),
                      shape=(len(rs), len(keep)))

out = []
for s in sorted(set(groups)):
    te = np.where(groups == s)[0]
    tr = np.where(groups != s)[0]
    if len(te) < MIN_TEST or len(set(y[tr])) < 2:
        continue
    ytr, yte = y[tr], y[te]
    oR = class_counts(rows[tr][ytr == 1]); oS = class_counts(rows[tr][ytr == 0])
    tot = oR + oS; pR = (ytr == 1).mean()
    eR, eS = pR * tot, (1 - pR) * tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = (oR - eR)**2 / eR + (oS - eS)**2 / eS
    chi2[~np.isfinite(chi2)] = 0.0
    keep = np.sort(np.argpartition(chi2, -TOP_K)[-TOP_K:])

    clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", n_jobs=4,
                            random_state=0, tree_method="hist")
    clf.fit(submatrix(rows[tr], keep), ytr)
    Xte = submatrix(rows[te], keep)
    pred = clf.predict(Xte)
    single = len(set(yte)) < 2
    r = dict(source=s, n_test=int(len(te)), test_pct_R=float(yte.mean()),
             n_train=int(len(tr)),
             accuracy=float((pred == yte).mean()),
             precision=None if single else float(precision_score(yte, pred, zero_division=0)),
             recall=None if single else float(recall_score(yte, pred, zero_division=0)),
             f1=None if single else float(f1_score(yte, pred, zero_division=0)),
             auroc=None if single else float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])))
    out.append(r)
    f1s = "single-class" if single else f"F1={r['f1']:.3f} AUROC={r['auroc']:.3f}"
    print(f"  held out {s[:52]:52s} n={r['n_test']:4d} "
          f"({r['test_pct_R']:.0%} R) acc={r['accuracy']:.3f} {f1s}", flush=True)

os.makedirs(f"{BASE}/results_source", exist_ok=True)
json.dump({"antibiotic": abx, "top_k": TOP_K, "holdouts": out},
          open(f"{BASE}/results_source/{abx}.json", "w"), indent=2)
scored = [r["f1"] for r in out if r["f1"] is not None]
if scored:
    print(f"  mean F1 across {len(scored)} held-out sources: {np.mean(scored):.3f}")
