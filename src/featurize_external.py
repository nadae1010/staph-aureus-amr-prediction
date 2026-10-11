"""
One pass over the external assemblies, producing every k-mer presence vector
the validation needs.

Scanning a 2.8 Mb assembly costs a few seconds, so doing it once per antibiotic
would mean five full passes. This builds the union of
  (a) a genome-wide panel used for deduplication against the training set, and
  (b) the selected k-mers of all exported models
and records presence for all of them in a single sweep.

The training-side panel matrix is saved too, so the deduplication step never
has to touch the 38 GB k-mer matrix again.

Usage:  python3 featurize_external.py [panel_size=20000]
Output: /blue/boucher/nelseifi/external/external_features.npz
          ext        bool  (n_external x n_union)   external presence
          train_panel bool (4610 x panel_size)      training presence, panel only
          panel_cols int32 (panel_size,)            panel positions within union
          accessions  str                           external genome order
          model_cols  int32 per antibiotic          model positions within union
"""
import glob
import os
import sys

import numpy as np

BASE = "/blue/boucher/nelseifi"
MAT = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
EXT = f"{BASE}/external"
N_FEATURES = 11_351_471
K = 31
PANEL = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
COMP = str.maketrans("ACGT", "TGCA")
rng = np.random.default_rng(0)

ANTIBIOTICS = [d for d in sorted(os.listdir(f"{BASE}/models"))
               if os.path.isfile(f"{BASE}/models/{d}/kmers.txt")]
print("models found:", ", ".join(ANTIBIOTICS), flush=True)

# ---------------------------------------------- pick the deduplication panel
indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
n_train = len(indptr) - 1

freq = np.zeros(N_FEATURES, dtype=np.int32)
for i in range(0, n_train, 400):
    c = np.concatenate([np.asarray(cols[indptr[r]:indptr[r + 1]])
                        for r in range(i, min(i + 400, n_train))])
    freq += np.bincount(c, minlength=N_FEATURES).astype(np.int32)
    del c
band = np.where((freq >= int(0.05 * n_train)) & (freq <= int(0.95 * n_train)))[0]
panel_idx = np.sort(rng.choice(band, size=min(PANEL, len(band)), replace=False))
print(f"panel: {len(panel_idx):,} k-mers drawn from {len(band):,} informative",
      flush=True)

# -------------------------------- sequences: panel (from index) + model files
want = {int(i) for i in panel_idx}
panel_seq = {}
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, s = line.rstrip("\n").split(",", 1)
        i = int(i_str)
        if i in want:
            panel_seq[i] = s
            if len(panel_seq) == len(want):
                break

union, pos = [], {}


def add(seq):
    j = pos.get(seq)
    if j is None:
        j = len(union)
        pos[seq] = j
        union.append(seq)
    return j


panel_cols = np.array([add(panel_seq[int(i)]) for i in panel_idx], dtype=np.int32)
model_cols = {}
for abx in ANTIBIOTICS:
    kl = [l.strip() for l in open(f"{BASE}/models/{abx}/kmers.txt") if l.strip()]
    model_cols[abx] = np.array([add(s) for s in kl], dtype=np.int32)
    print(f"  {abx:14s} {len(kl):7,} k-mers", flush=True)
print(f"union: {len(union):,} distinct k-mers\n", flush=True)

lookup = {}
for s, j in pos.items():
    lookup[s] = j
    lookup[s[::-1].translate(COMP)] = j

# ------------------------------------------- training presence, panel only
lut = np.full(N_FEATURES, -1, dtype=np.int32)
lut[panel_idx] = np.arange(len(panel_idx), dtype=np.int32)
train_panel = np.zeros((n_train, len(panel_idx)), dtype=bool)
for r in range(n_train):
    j = lut[np.asarray(cols[indptr[r]:indptr[r + 1]])]
    train_panel[r, j[j >= 0]] = True
print(f"training panel matrix {train_panel.shape}, "
      f"mean {train_panel.sum(1).mean():.0f} k-mers/genome\n", flush=True)


# ------------------------------------------------------ scan the assemblies
def accession_of(path):
    for part in path.split(os.sep):
        if part.startswith(("GCA_", "GCF_")):
            return part
    return os.path.basename(path)


files = sorted(glob.glob(f"{EXT}/raw/**/*.fna", recursive=True))
print(f"{len(files)} external assemblies\n", flush=True)

ext = np.zeros((len(files), len(union)), dtype=bool)
accs = []
for n, path in enumerate(files):
    accs.append(accession_of(path))
    buf = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if buf:
                    s = "".join(buf).upper()
                    for i in range(len(s) - K + 1):
                        j = lookup.get(s[i:i + K])
                        if j is not None:
                            ext[n, j] = True
                    buf = []
            else:
                buf.append(line.strip())
    if buf:
        s = "".join(buf).upper()
        for i in range(len(s) - K + 1):
            j = lookup.get(s[i:i + K])
            if j is not None:
                ext[n, j] = True
    if n % 50 == 0 or n == len(files) - 1:
        print(f"  [{n+1}/{len(files)}] {accs[-1]}  "
              f"{int(ext[n].sum()):,} union k-mers present", flush=True)

np.savez_compressed(
    f"{EXT}/external_features.npz",
    ext=ext, train_panel=train_panel, panel_cols=panel_cols,
    accessions=np.array(accs),
    **{f"model_{a}": model_cols[a] for a in ANTIBIOTICS})
print(f"\nwrote {EXT}/external_features.npz")
