"""
Compare classifiers on the same leakage-controlled splits and the same
chi-square-selected features. Mirrors the model set used in the AURA paper:
naive Bayes, LASSO logistic regression, linear SVM, random forest, XGBoost.

Features are presence/absence, so BernoulliNB is the appropriate NB variant.
XGBoost uses the validation split for early stopping; the other models have
no tuning, so their validation split is unused (noted as a limitation).

Usage:  python3 model_compare.py <antibiotic> [top_k=10000]
Output: /blue/boucher/nelseifi/results_models/<abx>.json
"""
import json, os, sys, time, warnings
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)
import xgboost as xgb
warnings.filterwarnings("ignore")

BASE, SUFFIX = "/blue/boucher/nelseifi", "k31_min5_max3000_d0"
MAT, N_FEATURES = f"{BASE}/kmx_output", 11_351_471
abx   = sys.argv[1]
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
SEEDS = [0, 1, 2, 3, 4]

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols   = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
labels = np.load(f"{BASE}/labels_{abx}.npy")

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

def models(seed):
    return {
        "naive_bayes": BernoulliNB(),
        "lasso_logreg": LogisticRegression(penalty="l1", solver="liblinear",
                                           C=0.1, max_iter=2000),
        "linear_svm": LinearSVC(C=0.1, max_iter=5000),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=4,
                                                random_state=seed),
        "xgboost": xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.8,
                                     eval_metric="logloss", early_stopping_rounds=25,
                                     n_jobs=4, random_state=seed, tree_method="hist"),
    }

results = {k: [] for k in models(0)}
for seed in SEEDS:
    sp = json.load(open(f"{BASE}/splits/{abx}_seed{seed}.json"))
    ch = np.load(f"{BASE}/chi2/{abx}_seed{seed}_chi2top.npz")
    keep = np.sort(ch["idx"][:TOP_K].astype(np.int64))
    tr, va, te = (np.array(sp[k]) for k in ("train", "val", "test"))
    Xtr, Xva, Xte = submatrix(tr, keep), submatrix(va, keep), submatrix(te, keep)
    ytr, yva, yte = labels[tr], labels[va], labels[te]
    print(f"  seed {seed}:", flush=True)

    for name, clf in models(seed).items():
        t0 = time.time()
        if name == "xgboost":
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        else:
            clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        score = (clf.decision_function(Xte) if name == "linear_svm"
                 else clf.predict_proba(Xte)[:, 1])
        r = dict(seed=seed,
                 precision=float(precision_score(yte, pred, zero_division=0)),
                 recall=float(recall_score(yte, pred, zero_division=0)),
                 f1=float(f1_score(yte, pred, zero_division=0)),
                 auroc=float(roc_auc_score(yte, score)),
                 auprc=float(average_precision_score(yte, score)),
                 cm=confusion_matrix(yte, pred).tolist(),
                 seconds=round(time.time() - t0, 1))
        results[name].append(r)
        print(f"    {name:14s} F1={r['f1']:.3f} AUROC={r['auroc']:.3f} "
              f"AUPRC={r['auprc']:.3f} ({r['seconds']}s)", flush=True)

print(f"\n{abx} (top {TOP_K:,} k-mers), mean +/- sd over {len(SEEDS)} seeds:")
print(f"  {'model':14s}{'F1':>16}{'AUROC':>16}{'AUPRC':>16}")
for name, rs in results.items():
    g = lambda m: np.array([r[m] for r in rs])
    print(f"  {name:14s}" + "".join(f"{g(m).mean():>10.3f}+/-{g(m).std():.3f}"
                                    for m in ("f1", "auroc", "auprc")))

os.makedirs(f"{BASE}/results_models", exist_ok=True)
json.dump({"antibiotic": abx, "top_k": TOP_K, "models": results},
          open(f"{BASE}/results_models/{abx}.json", "w"), indent=2)
