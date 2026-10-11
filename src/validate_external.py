"""
External validation of the exported models on NCBI Pathogen Detection isolates.

Two stages:

1. Deduplication. BV-BRC ingests the same public assemblies, so some external
   genomes are already in the 4,610 used for training. Each external genome is
   compared to every training genome by Jaccard similarity over a genome-wide
   k-mer panel; anything at or above the threshold is treated as not external.
   Metrics are reported before and after removal so the effect is visible.

2. Scoring. Each model is applied to the external presence vectors and scored
   against the NCBI phenotypes.

Expect the decision threshold to transfer poorly where external resistance
prevalence differs sharply from training -- ciprofloxacin is 5.8% resistant
here versus 42.8% in training. AUROC and AUPRC are the metrics to read.

Usage:  python3 validate_external.py [jaccard_threshold=0.99]
Output: printed report, plus /blue/boucher/nelseifi/external/external_results.json
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np
import xgboost as xgb
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)

BASE = "/blue/boucher/nelseifi"
EXT = f"{BASE}/external"
THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.99

d = np.load(f"{EXT}/external_features.npz", allow_pickle=True)
ext, train_panel = d["ext"], d["train_panel"]
panel_cols, accs = d["panel_cols"], [str(a) for a in d["accessions"]]
antibiotics = sorted(k[len("model_"):] for k in d.files if k.startswith("model_"))
print(f"external genomes: {len(accs)} | union k-mers: {ext.shape[1]:,} | "
      f"models: {', '.join(antibiotics)}\n")

# ----------------------------------------------------------- deduplication
E = ext[:, panel_cols].astype(np.float32)
T = train_panel.astype(np.float32)
inter = E @ T.T
se, st = E.sum(1), T.sum(1)
union = se[:, None] + st[None, :] - inter
with np.errstate(divide="ignore", invalid="ignore"):
    jac = np.where(union > 0, inter / union, 0.0)
best = jac.max(axis=1)
dup = best >= THRESH
print(f"Jaccard vs training (panel of {len(panel_cols):,} k-mers)")
print(f"  max similarity: median {np.median(best):.3f}, "
      f"90th pct {np.percentile(best, 90):.3f}, max {best.max():.3f}")
for t in (0.999, 0.99, 0.95, 0.90):
    print(f"  >= {t:.3f}: {int((best >= t).sum()):4d} genomes")
print(f"\nflagged as duplicates at {THRESH}: {int(dup.sum())} / {len(accs)}; "
      f"{int((~dup).sum())} remain external\n")

with open(f"{EXT}/dedup_report.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["assembly", "max_jaccard_vs_training", "flagged_duplicate"])
    for a, b, q in zip(accs, best, dup):
        w.writerow([a, f"{b:.4f}", int(q)])

# ------------------------------------------------------------- phenotypes
pheno = defaultdict(dict)
with open(f"{BASE}/external_phenotypes.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        pheno[r["antibiotic"]][r["assembly"]] = r["phenotype"]

idx_of = {a: i for i, a in enumerate(accs)}
results = {}
for abx in antibiotics:
    cols = d[f"model_{abx}"]
    booster = xgb.Booster()
    booster.load_model(f"{BASE}/models/{abx}/model.json")
    meta = json.load(open(f"{BASE}/models/{abx}/meta.json"))

    rows, y = [], []
    for acc, lab in pheno.get(abx, {}).items():
        i = idx_of.get(acc)
        if i is not None and lab in ("Resistant", "Susceptible"):
            rows.append(i)
            y.append(1 if lab == "Resistant" else 0)
    if not rows:
        print(f"{abx}: no external phenotypes matched a downloaded genome\n")
        continue
    rows, y = np.array(rows), np.array(y)

    X = ext[np.ix_(rows, cols)].astype(np.float32)
    prob = booster.predict(xgb.DMatrix(X))

    def report(mask, tag):
        yy, pp = y[mask], prob[mask]
        if len(set(yy.tolist())) < 2:
            print(f"  {tag:14s} n={len(yy):4d}  single-class, skipped")
            return None
        pred = (pp >= 0.5).astype(int)
        r = dict(n=int(len(yy)), pct_R=float(yy.mean()),
                 precision=float(precision_score(yy, pred, zero_division=0)),
                 recall=float(recall_score(yy, pred, zero_division=0)),
                 f1=float(f1_score(yy, pred, zero_division=0)),
                 auroc=float(roc_auc_score(yy, pp)),
                 auprc=float(average_precision_score(yy, pp)),
                 cm=confusion_matrix(yy, pred).tolist())
        print(f"  {tag:14s} n={r['n']:4d} ({r['pct_R']:5.1%} R)  "
              f"F1={r['f1']:.3f}  AUROC={r['auroc']:.3f}  AUPRC={r['auprc']:.3f}  "
              f"P={r['precision']:.3f} R={r['recall']:.3f}")
        return r

    print(f"=== {abx}  (trained on {meta['n_genomes']} genomes, "
          f"{meta['pct_resistant']:.1%} R, top_k={meta['top_k']:,})")
    all_r = report(np.ones(len(rows), bool), "all matched")
    ded_r = report(~dup[rows], "deduplicated")
    cov = X.mean(axis=1)
    print(f"  model-k-mer coverage: median {np.median(cov):.3f}, "
          f"min {cov.min():.3f}\n")
    results[abx] = {"all": all_r, "deduplicated": ded_r,
                    "n_flagged_duplicate": int(dup[rows].sum())}

json.dump({"jaccard_threshold": THRESH, "results": results},
          open(f"{EXT}/external_results.json", "w"), indent=2)
print(f"wrote {EXT}/external_results.json and {EXT}/dedup_report.tsv")
