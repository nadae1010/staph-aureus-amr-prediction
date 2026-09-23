"""
Held-out-source evaluation with threshold calibration.

Same design as source_holdout.py, but the decision threshold is chosen on a
calibration split carved out of the TRAINING studies (grouped by source, so
no study spans train and calibration). The held-out study is used only for
final scoring. Reports metrics at both 0.5 and the calibrated threshold.

Usage:  python3 source_holdout_cal.py <antibiotic> [top_k=10000] [min_test=100]
Output: /blue/boucher/nelseifi/results_source_cal/<abx>.json

RESULT: calibration did not help. Mean F1 fell for 4 of 6 antibiotics
(clindamycin 0.776->0.654, methicillin 0.765->0.714). A threshold tuned on
other studies does not transfer to a study with different resistance
prevalence. Kept as a documented negative result; AUROC/AUPRC are the
appropriate cross-study metrics.
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
rng = np.random.default_rng(0)

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols   = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
labels = np.load(f"{BASE}/labels_{abx}.npy")

aligned = pd.read_csv(f"{BASE}/phenotypes_matrix_aligned.csv", dtype=str)
aligned["matrix_row"] = aligned["matrix_row"].astype(int)
src = pd.read_csv(f"{BASE}/genome_sources.csv", dtype=str)
src = src[src["Antibiotic"] == abx][["Genome ID", "source"]]
m = aligned.merge(src, on="Genome ID", how="inner")
m = m[labels[m["matrix_row"].values] >= 0]
rows, groups = m["matrix_row"].values, m["source"].values
y = labels[rows]
print(f"{abx}: {len(rows)} genomes, {len(set(groups))} sources")

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

def best_threshold(yv, pv):
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(grid, key=lambda t: f1_score(yv, (pv >= t).astype(int), zero_division=0)))

out = []
for s in sorted(set(groups)):
    te = np.where(groups == s)[0]
    pool = np.where(groups != s)[0]
    if len(te) < MIN_TEST or len(set(y[pool])) < 2:
        continue
    # carve calibration studies (~25% of training genomes) off the training pool
    pool_srcs = sorted(set(groups[pool]))
    rng.shuffle(pool_srcs)
    cal_srcs, n = [], 0
    for c in pool_srcs:
        if n >= 0.25 * len(pool):
            break
        cal_srcs.append(c); n += int((groups[pool] == c).sum())
    cal = pool[np.isin(groups[pool], cal_srcs)]
    tr  = pool[~np.isin(groups[pool], cal_srcs)]
    if len(set(y[tr])) < 2 or len(set(y[cal])) < 2:
        tr, cal = pool, pool           # fall back: calibrate on training itself

    ytr, ycal, yte = y[tr], y[cal], y[te]
    oR, oS = class_counts(rows[tr][ytr == 1]), class_counts(rows[tr][ytr == 0])
    tot, pR = oR + oS, (ytr == 1).mean()
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
    thr = best_threshold(ycal, clf.predict_proba(submatrix(rows[cal], keep))[:, 1])

    p = clf.predict_proba(submatrix(rows[te], keep))[:, 1]
    single = len(set(yte)) < 2
    r = dict(source=s, n_test=int(len(te)), test_pct_R=float(yte.mean()),
             threshold=thr,
             f1_at_half=None if single else float(f1_score(yte, (p >= .5).astype(int), zero_division=0)),
             f1_cal=None if single else float(f1_score(yte, (p >= thr).astype(int), zero_division=0)),
             precision_cal=None if single else float(precision_score(yte, (p >= thr).astype(int), zero_division=0)),
             recall_cal=None if single else float(recall_score(yte, (p >= thr).astype(int), zero_division=0)),
             auroc=None if single else float(roc_auc_score(yte, p)),
             acc_cal=float(((p >= thr).astype(int) == yte).mean()))
    out.append(r)
    msg = "single-class" if single else (f"F1 {r['f1_at_half']:.3f} -> {r['f1_cal']:.3f} "
                                        f"AUROC={r['auroc']:.3f}")
    print(f"  {s[:50]:50s} n={r['n_test']:4d} ({r['test_pct_R']:.0%} R) thr={thr:.2f}  {msg}", flush=True)

os.makedirs(f"{BASE}/results_source_cal", exist_ok=True)
json.dump({"antibiotic": abx, "top_k": TOP_K, "holdouts": out},
          open(f"{BASE}/results_source_cal/{abx}.json", "w"), indent=2)
h = [r["f1_at_half"] for r in out if r["f1_at_half"] is not None]
c = [r["f1_cal"] for r in out if r["f1_cal"] is not None]
if c:
    print(f"  mean F1: {np.mean(h):.3f} (0.5) -> {np.mean(c):.3f} (calibrated), "
          f"{len(c)} sources")
