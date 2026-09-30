"""
Cluster genomes by whole-genome k-mer similarity as a proxy for lineage.

Uses a random subsample of informative k-mers (present in 5-95% of genomes,
so neither near-universal core nor near-unique noise), computes pairwise
Jaccard similarity, and applies average-linkage hierarchical clustering.
Reports cluster-size distributions at several distance cutoffs so a cutoff
can be chosen before splits are rebuilt.

Usage:  python3 lineage_cluster.py [n_features=100000] [cutoff=0.30]
Output: /blue/boucher/nelseifi/lineage_clusters.csv  (matrix_row, Genome ID, cluster)
"""
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

BASE, SUFFIX = "/blue/boucher/nelseifi", "k31_min5_max3000_d0"
MAT, N_FEATURES = f"{BASE}/kmx_output", 11_351_471
N_SUB  = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
CUTOFF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
rng = np.random.default_rng(0)

indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols   = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
n_gen  = len(indptr) - 1
print(f"{n_gen} genomes", flush=True)

# genome frequency of every k-mer, in chunks
freq = np.zeros(N_FEATURES, dtype=np.int32)
for i in range(0, n_gen, 400):
    c = np.concatenate([np.asarray(cols[indptr[r]:indptr[r+1]])
                        for r in range(i, min(i+400, n_gen))])
    freq += np.bincount(c, minlength=N_FEATURES).astype(np.int32)
    del c
lo, hi = int(0.05 * n_gen), int(0.95 * n_gen)
mid = np.where((freq >= lo) & (freq <= hi))[0]
print(f"{len(mid):,} k-mers in 5-95% of genomes", flush=True)
sel = np.sort(rng.choice(mid, size=min(N_SUB, len(mid)), replace=False))

# binary genome x subsampled-kmer matrix
lut = np.full(N_FEATURES, -1, dtype=np.int32)
lut[sel] = np.arange(len(sel), dtype=np.int32)
data, ind, ptr = [], [], [0]
for r in range(n_gen):
    c = np.asarray(cols[indptr[r]:indptr[r+1]])
    mk = lut[c] >= 0
    ind.append(lut[c[mk]]); data.append(np.ones(int(mk.sum()), dtype=np.float32))
    ptr.append(ptr[-1] + int(mk.sum()))
X = csr_matrix((np.concatenate(data), np.concatenate(ind), np.array(ptr)),
               shape=(n_gen, len(sel)))
print(f"submatrix {X.shape}, density {X.nnz/(X.shape[0]*X.shape[1]):.3f}", flush=True)

# pairwise Jaccard distance
inter = np.asarray((X @ X.T).todense(), dtype=np.float32)
sz = np.diag(inter).copy()
union = sz[:, None] + sz[None, :] - inter
with np.errstate(divide="ignore", invalid="ignore"):
    D = 1.0 - inter / union
D[~np.isfinite(D)] = 1.0
np.fill_diagonal(D, 0.0)
D = np.clip((D + D.T) / 2, 0, 1)
print(f"median pairwise distance {np.median(D[np.triu_indices(n_gen,1)]):.3f}", flush=True)

Z = linkage(squareform(D, checks=False), method="average")
print("\ncutoff  clusters  largest  singletons")
for c in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
    lab = fcluster(Z, t=c, criterion="distance")
    s = pd.Series(lab).value_counts()
    print(f"  {c:.2f}   {len(s):6d}   {s.max():6d}   {(s==1).sum():6d}")

lab = fcluster(Z, t=CUTOFF, criterion="distance")
aligned = pd.read_csv(f"{BASE}/phenotypes_matrix_aligned.csv", dtype=str)
out = pd.DataFrame({"matrix_row": np.arange(n_gen),
                    "Genome ID": aligned["Genome ID"].values,
                    "cluster": lab})
out.to_csv(f"{BASE}/lineage_clusters.csv", index=False)
print(f"\nwrote lineage_clusters.csv at cutoff {CUTOFF} "
      f"({out['cluster'].nunique()} clusters)")
