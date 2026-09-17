"""
Train + evaluate one antibiotic across all 5 seeds.
Features: top-K chi-square k-mers (selected on training data only).
Model: XGBoost. Val used for early stopping; test touched once, at the end.

Usage: python3 train_pilot.py <antibiotic> [top_k]
"""
import json, os, sys, time
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score,
                             average_precision_score)
import xgboost as xgb

BASE   = "/blue/boucher/nelseifi"
MAT    = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
SEEDS  = [0, 1, 2, 3, 4]

abx   = sys.argv[1]
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols   = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
vals   = np.load(f"{MAT}/data_{SUFFIX}.npy",   mmap_mode="r")
labels = np.load(f"{BASE}/labels_{abx}.npy")

def submatrix(rows, keep):
    """Rows x selected-features dense-ish CSR, remapping kept columns to 0..K-1."""
    lut = np.full(11_351_471, -1, dtype=np.int32)
    lut[keep] = np.arange(len(keep), dtype=np.int32)
    data, indices, indptr_out = [], [], [0]
    for r in rows:
        c = np.asarray(cols[indptr[r]:indptr[r+1]])
        v = np.asarray(vals[indptr[r]:indptr[r+1]])
        m = lut[c] >= 0
        indices.append(lut[c[m]]); data.append(v[m])
        indptr_out.append(indptr_out[-1] + int(m.sum()))
    return csr_matrix((np.concatenate(data), np.concatenate(indices),
                       np.array(indptr_out)), shape=(len(rows), len(keep)))

results = []
for seed in SEEDS:
    t0 = time.time()
    sp = json.load(open(f"{BASE}/splits/{abx}_seed{seed}.json"))
    ch = np.load(f"{BASE}/chi2/{abx}_seed{seed}_chi2top.npz")
    keep = np.sort(ch["idx"][:TOP_K].astype(np.int64))

    tr, va, te = (np.array(sp[k]) for k in ("train", "val", "test"))
    Xtr, Xva, Xte = submatrix(tr, keep), submatrix(va, keep), submatrix(te, keep)
    ytr, yva, yte = labels[tr], labels[va], labels[te]

    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", early_stopping_rounds=25,
        n_jobs=4, random_state=seed, tree_method="hist")
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)[:, 1]
    r = dict(seed=seed,
             precision=precision_score(yte, pred),
             recall=recall_score(yte, pred),
             f1=f1_score(yte, pred),
             auroc=roc_auc_score(yte, prob),
             auprc=average_precision_score(yte, prob),
             cm=confusion_matrix(yte, pred).tolist(),
             best_iter=int(clf.best_iteration))
    results.append(r)
    print(f"  seed {seed}: P={r['precision']:.3f} R={r['recall']:.3f} "
          f"F1={r['f1']:.3f} AUROC={r['auroc']:.3f} AUPRC={r['auprc']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

print(f"\n{abx} (top {TOP_K:,} k-mers), mean +/- sd over {len(SEEDS)} seeds:")
for m in ("precision", "recall", "f1", "auroc", "auprc"):
    v = np.array([r[m] for r in results])
    print(f"  {m:10s} {v.mean():.3f} +/- {v.std():.3f}")

os.makedirs(f"{BASE}/results", exist_ok=True)
json.dump({"antibiotic": abx, "top_k": TOP_K, "seeds": results},
          open(f"{BASE}/results/{abx}_top{TOP_K}.json", "w"), indent=2)
