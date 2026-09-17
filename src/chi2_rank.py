"""
Chi-square ranking of k-mers per antibiotic per seed, fit on TRAINING ROWS ONLY.
Saves the top-N ranked k-mer column indices + scores so downstream models can
try different cutoffs without recomputing.

Matrix: 4,610 genomes x 11,351,471 k-mers (CSR, k=31, min5/max3000)
Usage:  python3 chi2_rank.py <antibiotic>
"""
import json, os, sys, time
import numpy as np

BASE   = "/blue/boucher/nelseifi"
MAT    = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
SPLITS = f"{BASE}/splits"
OUT    = f"{BASE}/chi2"
SEEDS  = [0, 1, 2, 3, 4]
TOP_N  = 500_000
N_FEATURES = 11_351_471

abx = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols   = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
vals   = np.load(f"{MAT}/data_{SUFFIX}.npy",   mmap_mode="r")
labels = np.load(f"{BASE}/labels_{abx}.npy")
print(f"{abx}: n_features={N_FEATURES:,}", flush=True)

def class_sums(rows):
    """Sum feature values over the given matrix rows."""
    acc = np.zeros(N_FEATURES, dtype=np.float64)
    CH = 400
    for i in range(0, len(rows), CH):
        chunk = rows[i:i+CH]
        c = np.concatenate([cols[indptr[r]:indptr[r+1]] for r in chunk])
        v = np.concatenate([vals[indptr[r]:indptr[r+1]] for r in chunk])
        acc += np.bincount(c, weights=v.astype(np.float64), minlength=N_FEATURES)
        del c, v
    return acc

for seed in SEEDS:
    t0 = time.time()
    with open(f"{SPLITS}/{abx}_seed{seed}.json") as f:
        sp = json.load(f)
    train = np.array(sp["train"])
    y = labels[train]
    rows_R, rows_S = train[y == 1], train[y == 0]

    obs_R = class_sums(rows_R)
    obs_S = class_sums(rows_S)

    total = obs_R + obs_S
    pR = len(rows_R) / len(train)
    exp_R, exp_S = pR * total, (1 - pR) * total

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = (obs_R - exp_R)**2 / exp_R + (obs_S - exp_S)**2 / exp_S
    chi2[~np.isfinite(chi2)] = 0.0

    k = min(TOP_N, N_FEATURES)
    top = np.argpartition(chi2, -k)[-k:]
    top = top[np.argsort(chi2[top])[::-1]]

    np.savez_compressed(f"{OUT}/{abx}_seed{seed}_chi2top.npz",
                        idx=top.astype(np.uint32),
                        score=chi2[top].astype(np.float32),
                        n_features=N_FEATURES,
                        n_train=len(train), n_train_R=len(rows_R))
    print(f"  seed {seed}: {len(train)} train rows, "
          f"top score {chi2[top[0]]:.1f}, {time.time()-t0:.0f}s", flush=True)
