"""
Train and evaluate on lineage-grouped splits (80/20, 5 seeds).

Chi-square ranking and top-K selection are refit on the training rows of each
lineage split -- the chi2/ files from the strain-grouped pipeline cannot be
reused, since those were fit on different training sets.

No validation split under this design, so n_estimators is fixed (no early
stopping). For comparability, the strain-grouped arm should be rerun the same
way if a like-for-like comparison is required.

Usage:  python3 train_lineage.py <antibiotic> [top_k=10000]
Output: /blue/boucher/nelseifi/results_lineage/<abx>_top<K>.json
"""
import json, os, sys, time
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)
import xgboost as xgb

BASE, SUFFIX = "/blue/boucher/nelseifi", "k31_min5_max3000_d0"
MAT, N_FEATURES = f"{BASE}/kmx_output", 11_351_471
abx   = sys.argv[1]
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
SEEDS = [0, 1, 2, 3, 4]

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols   = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
labels = np.load(f"{BASE}/labels_{abx}.npy")

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

results = []
for seed in SEEDS:
    t0 = time.time()
    sp = json.load(open(f"{BASE}/splits_lineage/{abx}_seed{seed}.json"))
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    ytr, yte = labels[tr], labels[te]

    oR, oS = class_counts(tr[ytr == 1]), class_counts(tr[ytr == 0])
    tot, pR = oR + oS, (ytr == 1).mean()
    eR, eS = pR * tot, (1 - pR) * tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = (oR - eR)**2 / eR + (oS - eS)**2 / eS
    chi2[~np.isfinite(chi2)] = 0.0
    keep = np.sort(np.argpartition(chi2, -TOP_K)[-TOP_K:])

    clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", n_jobs=4,
                            random_state=seed, tree_method="hist")
    clf.fit(submatrix(tr, keep), ytr)
    Xte = submatrix(te, keep)
    pred, prob = clf.predict(Xte), clf.predict_proba(Xte)[:, 1]
    r = dict(seed=seed, n_train=int(len(tr)), n_test=int(len(te)),
             train_pct_R=float(ytr.mean()), test_pct_R=float(yte.mean()),
             precision=float(precision_score(yte, pred, zero_division=0)),
             recall=float(recall_score(yte, pred, zero_division=0)),
             f1=float(f1_score(yte, pred, zero_division=0)),
             auroc=float(roc_auc_score(yte, prob)),
             auprc=float(average_precision_score(yte, prob)),
             cm=confusion_matrix(yte, pred).tolist())
    results.append(r)
    print(f"  seed {seed}: P={r['precision']:.3f} R={r['recall']:.3f} "
          f"F1={r['f1']:.3f} AUROC={r['auroc']:.3f} AUPRC={r['auprc']:.3f} "
          f"(test {r['test_pct_R']:.0%} R, {time.time()-t0:.0f}s)", flush=True)

print(f"\n{abx} lineage-grouped (top {TOP_K:,} k-mers), mean +/- sd:")
for m in ("precision", "recall", "f1", "auroc", "auprc"):
    v = np.array([r[m] for r in results])
    print(f"  {m:10s} {v.mean():.3f} +/- {v.std():.3f}")

os.makedirs(f"{BASE}/results_lineage", exist_ok=True)
json.dump({"antibiotic": abx, "top_k": TOP_K, "grouping": "kmer_lineage_cluster",
           "seeds": results},
          open(f"{BASE}/results_lineage/{abx}_top{TOP_K}.json", "w"), indent=2)

