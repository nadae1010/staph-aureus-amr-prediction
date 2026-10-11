"""
Paired bootstrap on the mecA-exclusion variants.

Excluding mecA moved clindamycin's external AUROC from 0.758 to 0.775 (narrow)
and 0.773 (broad), while the erythromycin control moved 0.961 -> 0.958 / 0.956.
The signs are what the confounder hypothesis predicts, but +0.017 is about one
standalone standard error, so the raw numbers cannot distinguish a real
improvement from noise.

The comparison is paired -- the same external genomes scored by three models
sharing ~97% of their features -- so the difference has far less variance than
either AUROC does alone. This resamples the genomes with replacement, recomputes
every model's AUROC on each resample, and reports the distribution of the
DIFFERENCE from the baseline model. A 95% interval excluding zero supports the
claim that removing a confounded feature improved generalisation; an interval
straddling zero means the exclusion result is a null and should be written up
as one.

Note what this does and does not test. It asks whether the observed difference
is stable under resampling of these external genomes. It does not account for
the single draw of training data, nor for having tried two exclusion widths --
with two arms tested, a one-sided p near 0.05 is weaker than it looks.

Usage:  python3 bootstrap_variants.py [n_boot=2000] [seed=0]
Output: printed intervals
"""
import csv
import sys
from collections import defaultdict

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

BASE = "/blue/boucher/nelseifi"
EXT = f"{BASE}/external"
NBOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0

FAMILIES = {
    "clindamycin": ["clindamycin", "clindamycin_nomecA_narrow",
                    "clindamycin_nomecA_broad"],
    "erythromycin": ["erythromycin", "erythromycin_nomecA_narrow",
                     "erythromycin_nomecA_broad"],
}

d = np.load(f"{EXT}/external_features.npz", allow_pickle=True)
ext = d["ext"]
accs = [str(a) for a in d["accessions"]]
idx_of = {a: i for i, a in enumerate(accs)}

# reuse the deduplication decision from validate_external.py
dup = {}
with open(f"{EXT}/dedup_report.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        dup[r["assembly"]] = r["flagged_duplicate"] == "1"

pheno = defaultdict(dict)
with open(f"{BASE}/external_phenotypes.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        pheno[r["antibiotic"]][r["assembly"]] = r["phenotype"]

rng = np.random.default_rng(SEED)

for abx, variants in FAMILIES.items():
    rows, y = [], []
    for acc, lab in pheno.get(abx, {}).items():
        i = idx_of.get(acc)
        if i is not None and lab in ("Resistant", "Susceptible") and not dup[acc]:
            rows.append(i)
            y.append(1 if lab == "Resistant" else 0)
    rows, y = np.array(rows), np.array(y)
    print(f"\n{'='*66}\n{abx}: n={len(rows)}  R={int(y.sum())}  "
          f"S={int((y==0).sum())}", flush=True)

    probs = {}
    for v in variants:
        booster = xgb.Booster()
        booster.load_model(f"{BASE}/models/{v}/model.json")
        X = ext[np.ix_(rows, d[f"model_{v}"])].astype(np.float32)
        probs[v] = booster.predict(xgb.DMatrix(X, missing=0))
        print(f"  {v:34s} AUROC = {roc_auc_score(y, probs[v]):.4f}", flush=True)

    # paired resampling: one index draw, scored by every variant
    boot = {v: np.empty(NBOOT) for v in variants}
    n = 0
    while n < NBOOT:
        s = rng.integers(0, len(y), len(y))
        ys = y[s]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        for v in variants:
            boot[v][n] = roc_auc_score(ys, probs[v][s])
        n += 1

    base = variants[0]
    print(f"\n  {NBOOT} paired resamples, difference vs {base}:")
    for v in variants[1:]:
        diff = boot[v] - boot[base]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        frac = float((diff > 0).mean())
        verdict = "excludes zero" if lo > 0 or hi < 0 else "STRADDLES ZERO"
        print(f"    {v:34s} {diff.mean():+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P(>0)={frac:.3f}  {verdict}")
