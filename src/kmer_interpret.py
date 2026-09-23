"""
Pull the most stable top chi-square k-mers for one antibiotic, get their
sequences, and compute prevalence in resistant vs susceptible genomes.
Stable = in the top 200 of at least 4 of 5 seeds.

Usage:  python3 kmer_interpret.py <antibiotic> [n_out=50]
Output: /blue/boucher/nelseifi/interpret/<abx>_topkmers.tsv and .fasta
"""
import os, sys
from collections import Counter
import numpy as np

BASE   = "/blue/boucher/nelseifi"
MAT    = f"{BASE}/kmx_output"
SUFFIX = "k31_min5_max3000_d0"
OUT    = f"{BASE}/interpret"
abx    = sys.argv[1]
N_OUT  = int(sys.argv[2]) if len(sys.argv) > 2 else 50
os.makedirs(OUT, exist_ok=True)

# 1. stable k-mers across seeds
counts, scores = Counter(), {}
for s in range(5):
    z = np.load(f"{BASE}/chi2/{abx}_seed{s}_chi2top.npz")
    for i, sc in zip(z["idx"][:200], z["score"][:200]):
        i = int(i); counts[i] += 1
        scores.setdefault(i, []).append(float(sc))
stable = [i for i, c in counts.items() if c >= 4]
stable.sort(key=lambda i: -np.mean(scores[i]))
stable = stable[:N_OUT]
print(f"{abx}: {len(stable)} stable k-mers (top-200 in >=4/5 seeds)")

# 2. sequences: file is "index,K-mer" with a header row
want = set(stable); seqs = {}
with open(f"{MAT}/set_of_all_unique_kmers_{SUFFIX}.csv") as f:
    next(f)
    for line in f:
        i_str, seq = line.rstrip().split(",", 1)
        i = int(i_str)
        if i in want:
            seqs[i] = seq.upper()
            if len(seqs) == len(want): break
print(f"  sequences found: {len(seqs)}/{len(want)}")

# 3. prevalence in R vs S (all tested genomes; descriptive only)
indptr = np.load(f"{MAT}/row_{SUFFIX}.npy")
cols   = np.load(f"{MAT}/column_{SUFFIX}.npy", mmap_mode="r")
labels = np.load(f"{BASE}/labels_{abx}.npy")
sel = np.array(sorted(want), dtype=np.uint32)
pos = {int(k): j for j, k in enumerate(sel)}
hitR, hitS = np.zeros(len(sel)), np.zeros(len(sel))
nR, nS = int((labels == 1).sum()), int((labels == 0).sum())
for r in np.where(labels >= 0)[0]:
    c = np.asarray(cols[indptr[r]:indptr[r+1]])
    for k in sel[np.isin(sel, c)]:
        (hitR if labels[r] == 1 else hitS)[pos[int(k)]] += 1

# 4. write outputs
with open(f"{OUT}/{abx}_topkmers.tsv", "w") as t, open(f"{OUT}/{abx}_topkmers.fasta", "w") as fa:
    t.write("rank\tkmer_idx\tsequence\tmean_chi2\tseeds_in_top200\tpct_R\tpct_S\tenriched_in\n")
    for rank, i in enumerate(stable, 1):
        j = pos[i]; pR = 100*hitR[j]/nR; pS = 100*hitS[j]/nS
        seq = seqs.get(i, "NA")
        t.write(f"{rank}\t{i}\t{seq}\t{np.mean(scores[i]):.1f}\t{counts[i]}\t"
                f"{pR:.1f}\t{pS:.1f}\t{'R' if pR > pS else 'S'}\n")
        fa.write(f">{abx}_rank{rank}_chi2_{np.mean(scores[i]):.0f}_R{pR:.0f}_S{pS:.0f}\n{seq}\n")
        if rank <= 15:
            print(f"  {rank:2d} {seq}  chi2={np.mean(scores[i]):7.1f}  R={pR:5.1f}%  S={pS:5.1f}%")
print(f"  wrote {OUT}/{abx}_topkmers.tsv / .fasta")
