"""Thesis figures for the Results chapter. Writes PDF (for Overleaf) and PNG previews."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 8.5,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
})
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
B250, B400, B550 = "#86b6ef", "#3987e5", "#1c5cab"
GREY_DATA = "#a3a29b"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=INK, ha="right", va="bottom")


# ---------------------------------------------------------------- Figure 1
def fig_pipeline():
    fig = plt.figure(figsize=(6.5, 6.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    def box(x0, y0, w, h, title, body, accent=False):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=("#eaf2fc" if accent else "white"),
                                    ec=(BLUE if accent else AXIS), lw=(1.4 if accent else 1.0)))
        ax.text(x0 + w / 2, y0 + h - 0.17, title, ha="center", va="top",
                fontsize=8.5, fontweight="bold", color=INK)
        ax.text(x0 + w / 2, y0 + h - 0.40, body, ha="center", va="top",
                fontsize=7.5, color=INK2, linespacing=1.35)
        return (x0, y0, w, h)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0,
                                    shrinkA=0, shrinkB=0, mutation_scale=9))

    stages = [(6.05, "A", "Data"), (4.75, "B", "Features"), (3.45, "C", "Ranking"),
              (2.15, "D", "Evaluation"), (0.75, "E", "Interpretation")]
    for y, letter, name in stages:
        ax.text(0.05, y, letter, fontsize=11, fontweight="bold", color=INK, va="center")
        ax.text(0.32, y, name, fontsize=8.5, color=INK2, va="center")

    L, R, W = 1.35, 4.05, 2.35
    box(L, 5.65, W, 0.85, "BV-BRC development cohort",
        "Laboratory phenotypes, 6 antibiotics\n4,610 assemblies")
    box(R, 5.65, W, 0.85, "NCBI external cohort",
        "Pathogen Detection AST Browser\n1,235 assemblies")
    box(L, 4.30, W, 0.95, "k-mer matrix (KMX, Gerbil)",
        "Canonical 31-mers, occurrence 5 to 3,000\n4,610 × 11,351,471, CSR, binarized")
    box(R, 4.30, W, 0.95, "Similarity screening",
        "Jaccard ≥ 0.99 to any development genome\n87 removed, 1,148 retained")
    box(L, 3.05, W, 0.85, "Feature ranking",
        "Carrier-conditioned χ² on training rows\nTop 500,000 k-mers per antibiotic")
    box(L, 1.55, W, 1.15, "Internal evaluation",
        "XGBoost (5 classifiers compared)\nStrain-grouped 70/10/20, 5 seeds\nLineage-grouped 80/20, 5 seeds\nHeld-out source studies")
    box(R, 1.55, W, 1.15, "External validation",
        "Final models on all labelled genomes\n5,000 to 200,000 k-mers per antibiotic\n10 refits per model\nAUROC, Brier score", accent=True)
    w3 = 1.55
    box(L, 0.15, w3, 1.0, "Contig assembly",
        "Top k-mers joined on\n30-bp overlaps\nBLAST identification")
    box(L + w3 + 0.18, 0.15, w3, 1.0, "Locus association",
        "Carriage in both cohorts\nOdds ratios with\nWoolf intervals")
    box(L + 2 * (w3 + 0.18), 0.15, w3, 1.0, "AMRFinderPlus",
        "Reference annotation\nof all genomes\nand contigs")

    cx_l, cx_r = L + W / 2, R + W / 2
    arrow(cx_l, 5.65, cx_l, 5.25)
    arrow(cx_r, 5.65, cx_r, 5.25)
    arrow(cx_l, 4.30, cx_l, 3.90)
    arrow(cx_l, 3.05, cx_l, 2.70)
    arrow(cx_r, 4.30, cx_r, 2.70)                       # screened genomes -> external
    ax.plot([L + W, cx_r], [3.47, 3.47], color=MUTED, lw=1.0)   # ranked k-mers -> external
    ax.plot(cx_r, 3.47, "o", ms=3.5, color=MUTED)
    bus = 1.36
    e_centres = [L + w3 / 2 + i * (w3 + 0.18) for i in range(3)]
    ax.plot([cx_l, cx_l], [1.55, bus], color=MUTED, lw=1.0)
    ax.plot([cx_r, cx_r], [1.55, bus], color=MUTED, lw=1.0)
    ax.plot([e_centres[0], e_centres[-1]], [bus, bus], color=MUTED, lw=1.0)
    for x in e_centres:
        arrow(x, bus, x, 1.15)
    save(fig, "fig_pipeline")


# ---------------------------------------------------------------- Figure 2
FEATS = [1_000, 5_000, 10_000, 50_000, 200_000, 500_000]
BUDGET = {
    "Methicillin":   [0.995, 0.996, 0.997, 0.999],
    "Cefoxitin":     [0.981, 0.987, 0.989, 0.989],
    "Ciprofloxacin": [0.967, 0.969, 0.970, 0.972],
    "Oxacillin":     [0.916, 0.933, 0.948, 0.954],
    "Erythromycin":  [0.730, 0.922, 0.931, 0.964],
    "Clindamycin":   [0.826, 0.887, 0.892, 0.919, 0.942, 0.944],
}
MARK = {"Methicillin": "o", "Cefoxitin": "s", "Ciprofloxacin": "^", "Oxacillin": "D",
        "Erythromycin": "o", "Clindamycin": "s"}
DESIGN = {  # strain@10k, lineage@10k, lineage@50k, external mean, external sd
    "Methicillin":   (0.997, 0.981, None, None, None),
    "Cefoxitin":     (0.989, 0.985, None, 0.9647, 0.0025),
    "Ciprofloxacin": (0.971, 0.949, None, 0.9912, 0.0008),
    "Oxacillin":     (0.948, 0.883, None, 0.9875, 0.0006),
    "Erythromycin":  (0.931, 0.907, 0.954, 0.9606, 0.0009),
    "Clindamycin":   (0.892, 0.755, 0.857, 0.7841, 0.0138),
}


def fig_performance():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.1), gridspec_kw={"wspace": 0.42})

    for name, vals in BUDGET.items():
        mls = name in ("Erythromycin", "Clindamycin")
        col = {"Erythromycin": B400, "Clindamycin": B550}.get(name, GREY_DATA)
        a.plot(FEATS[:len(vals)], vals, color=col, lw=1.8 if mls else 1.2,
               marker=MARK[name], ms=4.5, mfc=col, mec="white", mew=0.6,
               zorder=3 if mls else 2, label=name)
    a.set_xscale("log")
    a.set_xticks(FEATS)
    a.set_xticklabels(["1k", "5k", "10k", "50k", "200k", "500k"])
    a.set_ylim(0.70, 1.005)
    a.set_xlim(800, 650_000)
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_xlabel("Selected k-mers (log scale)")
    a.set_ylabel("AUROC, strain-grouped (5-seed mean)")
    a.text(1_000 * 1.12, 0.730, "Erythromycin", color=B400, fontsize=7.5, va="center")
    a.text(500_000, 0.944 - 0.020, "Clindamycin", color=B550, fontsize=7.5, ha="right", va="center")
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.legend(loc="lower right", frameon=False, fontsize=7, handlelength=1.6,
             borderaxespad=0.2, bbox_to_anchor=(1.0, 0.05))
    panel(a, "a")

    names = list(DESIGN)[::-1]
    for i, name in enumerate(names):
        s, l10, l50, ext, sd = DESIGN[name]
        b.axhline(i, color=GRID, lw=0.6, zorder=0)
        b.plot(s, i, "o", ms=5.5, mfc="none", mec=INK2, mew=1.1, zorder=5)
        b.plot(l10, i, "s", ms=5, mfc=GREY_DATA, mec="white", mew=0.5, zorder=3)
        if l50 is not None:
            b.plot(l50, i, "^", ms=5.5, mfc=INK2, mec="white", mew=0.5, zorder=3)
        if ext is not None:
            if sd and sd > 0.005:
                b.plot([ext - sd, ext + sd], [i, i], color=BLUE, lw=1.6, zorder=3)
            b.plot(ext, i, "D", ms=5.5, mfc=BLUE, mec="white", mew=0.6, zorder=4)
        else:
            b.text(0.702, i - 0.32, "no external labels", fontsize=6.5, color=MUTED, va="center")
    b.set_yticks(range(len(names)))
    b.set_yticklabels(names)
    b.set_xlim(0.70, 1.005)
    b.set_ylim(-0.6, len(names) - 0.4)
    b.set_xlabel("AUROC")
    b.spines["left"].set_visible(False)
    handles = [
        Line2D([], [], ls="", marker="o", ms=5.5, mfc="white", mec=INK2, label="Strain-grouped, 10k"),
        Line2D([], [], ls="", marker="s", ms=5, mfc=GREY_DATA, mec="white", label="Lineage-grouped, 10k"),
        Line2D([], [], ls="", marker="^", ms=5.5, mfc=INK2, mec="white", label="Lineage-grouped, 50k"),
        Line2D([], [], ls="", marker="D", ms=5.5, mfc=BLUE, mec="white", label="External, 10-refit mean"),
    ]
    b.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.30), ncol=2,
             frameon=False, fontsize=7, handletextpad=0.3, columnspacing=0.9)
    panel(b, "b")
    save(fig, "fig_performance")


# ---------------------------------------------------------------- Figure 3: classifiers
MODELS = {  # F1, AUROC (5-seed means, top 10,000 k-mers; model_compare.py)
    "Methicillin":   {"NB": (0.994, 0.994), "LR": (0.994, 0.997), "SVM": (0.994, 0.996), "RF": (0.994, 0.996), "XGB": (0.994, 0.997)},
    "Cefoxitin":     {"NB": (0.936, 0.905), "LR": (0.949, 0.984), "SVM": (0.948, 0.984), "RF": (0.955, 0.984), "XGB": (0.956, 0.989)},
    "Ciprofloxacin": {"NB": (0.578, 0.711), "LR": (0.944, 0.962), "SVM": (0.943, 0.970), "RF": (0.940, 0.969), "XGB": (0.945, 0.970)},
    "Oxacillin":     {"NB": (0.898, 0.912), "LR": (0.898, 0.933), "SVM": (0.890, 0.935), "RF": (0.888, 0.945), "XGB": (0.900, 0.946)},
    "Erythromycin":  {"NB": (0.892, 0.903), "LR": (0.895, 0.925), "SVM": (0.888, 0.924), "RF": (0.888, 0.922), "XGB": (0.893, 0.927)},
    "Clindamycin":   {"NB": (0.802, 0.807), "LR": (0.818, 0.886), "SVM": (0.806, 0.879), "RF": (0.819, 0.880), "XGB": (0.826, 0.891)},
}
MSTYLE = {  # label, marker, face, edge, size
    "NB":  ("Naive Bayes", "X", INK2, "white", 6),
    "LR":  ("LASSO logistic regression", "o", "none", MUTED, 5.5),
    "SVM": ("Linear SVM", "s", GREY_DATA, "white", 5),
    "RF":  ("Random forest", "^", MUTED, "white", 5.5),
    "XGB": ("XGBoost", "D", BLUE, "white", 5.5),
}


def fig_models():
    lo = 0.78
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True, gridspec_kw={"wspace": 0.12})
    names = list(MODELS)[::-1]
    for j, (ax, metric) in enumerate(zip(axes, ["F1 score", "AUROC"])):
        for i, name in enumerate(names):
            ax.axhline(i, color=GRID, lw=0.6, zorder=0)
            for key in ["LR", "SVM", "RF", "NB", "XGB"]:
                lab, mk, fc, ec, ms = MSTYLE[key]
                v = MODELS[name][key][j]
                z = 6 if key == "XGB" else 4
                if v < lo:
                    ax.plot(lo + 0.004, i, marker="<", ms=5.5, color=INK2, clip_on=False, zorder=z)
                    ax.text(lo + 0.011, i + 0.33, f"Naive Bayes {v:.3f}", fontsize=6.5, color=INK2, va="center")
                    continue
                ax.plot(v, i, marker=mk, ms=ms, mfc=fc, mec=ec, mew=1.0 if fc == "none" else 0.5,
                        ls="", zorder=z)
        ax.set_xlim(lo, 1.005)
        ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
        ax.set_xticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"])
        ax.set_xlabel(metric)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.6, len(names) - 0.4)
        panel(ax, "ab"[j])
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    handles = [Line2D([], [], ls="", marker=MSTYLE[k][1], ms=MSTYLE[k][4], mfc=MSTYLE[k][2],
                      mec=(MSTYLE[k][3] if MSTYLE[k][2] != "none" else MUTED), label=MSTYLE[k][0])
               for k in ["XGB", "LR", "SVM", "RF", "NB"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 1.07), ncol=5, frameon=False,
               fontsize=7, handletextpad=0.3, columnspacing=1.0)
    save(fig, "fig_models")


# ---------------------------------------------------------------- Figure 4: transfer
def gene(txt):
    """Italicise the gene symbol only: 'erm|(C)' -> $\\it{erm}$(C)."""
    sym, _, rest = txt.partition("|")
    return r"$\it{" + sym + "}$" + rest


def fig_transfer():
    pts = [  # drug, gene label, best-feature OR, external mean, sd, features
        ("Ciprofloxacin", "gyrA| S84L", 175.7, 0.9912, 0.0008, "5k"),
        ("Oxacillin", "mecA|", 116.9, 0.9875, 0.0006, "50k"),
        ("Cefoxitin", "mecA|", 99.8, 0.9647, 0.0025, "5k"),
        ("Erythromycin", "erm|(C)", 36.7, 0.9606, 0.0009, "50k"),
        ("Clindamycin", "mecA|, N-terminal", 14.9, 0.7841, 0.0138, "200k"),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.55})
    lab = {"Ciprofloxacin": ((0, 9), "center", "bottom"), "Oxacillin": ((-7, 6), "right", "bottom"),
           "Cefoxitin": ((0, -9), "center", "top"), "Erythromycin": ((-8, 0), "right", "center"),
           "Clindamycin": ((8, 0), "left", "center")}
    for drug, g, orr, auc, sd, k in pts:
        col = BLUE if drug == "Clindamycin" else INK2
        a.plot(orr, auc, "o", ms=6.5, mfc=col, mec="white", mew=0.8, zorder=3)
        (dx, dy), ha, va = lab[drug]
        a.annotate(drug + " (" + gene(g) + ")", (orr, auc), xytext=(dx, dy), textcoords="offset points",
                   ha=ha, va=va, fontsize=7, color=INK)
    a.set_xscale("log")
    a.set_xlim(8, 450)
    a.set_ylim(0.76, 1.02)
    a.set_xticks([10, 30, 100, 300])
    a.set_xticklabels(["10", "30", "100", "300"])
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.grid(color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_xlabel("Odds ratio of strongest feature\n(development cohort, log scale)")
    a.set_ylabel("External AUROC (10-refit mean)")
    panel(a, "a")

    rows = pts  # already ordered by odds ratio, strongest first
    for i, (drug, g, orr, auc, sd, k) in enumerate(rows):
        y = len(rows) - 1 - i
        col = BLUE if drug == "Clindamycin" else GREY_DATA
        b.barh(y, sd, height=0.55, color=col, edgecolor="white", linewidth=1.0)
        b.text(sd + 0.0003, y, f"{sd:.4f}  ({k} k-mers)", va="center", fontsize=7, color=INK2)
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([r[0] for r in rows][::-1])
    b.set_xlim(0, 0.019)
    b.set_xticks([0, 0.005, 0.010, 0.015])
    b.set_xticklabels(["0", "0.005", "0.010", "0.015"])
    b.set_xlabel("SD of external AUROC\nacross 10 refits")
    b.spines["left"].set_visible(False)
    b.grid(axis="x", color=GRID, lw=0.6)
    b.set_axisbelow(True)
    panel(b, "b")
    save(fig, "fig_transfer")


# ---------------------------------------------------------------- Figure 5: clindamycin
def fig_clindamycin():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.0),
                               gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.45})
    loci = [  # gene label, dev OR, lo, hi, ext OR, lo, hi, colour, style, x-offset
        ("mecA|", 16.8, 13.3, 21.2, 1.5, 1.2, 2.0, ORANGE, "-", -0.06),
        ("erm|(C)", 4.9, 3.7, 6.4, 461.4, 28.5, 7470.3, BLUE, "-", -0.02),
        ("erm|(B)", 8.9, 6.0, 13.3, 40.6, 2.3, 704.7, BLUE, "-", 0.02),
        ("erm|(A)", 19.6, 13.1, 29.3, 197.2, 67.5, 576.0, BLUE, "--", 0.06),
    ]
    for g, d, dlo, dhi, e, elo, ehi, col, ls, o in loci:
        x0, x1 = 0 + o, 1 + o
        a.plot([x0, x1], [d, e], color=col, lw=1.6, ls=ls, zorder=3)
        for x, v, lo, hi in ((x0, d, dlo, dhi), (x1, e, elo, ehi)):
            a.plot([x, x], [lo, hi], color=col, lw=1.0, alpha=0.55, zorder=2)
            a.plot(x, v, "o", ms=5, mfc=col, mec="white", mew=0.6, zorder=4)
        a.text(1.15, e, gene(g), fontsize=7.5, color=INK, va="center")
    a.axhline(1, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    a.text(-0.27, 1.12, "OR = 1", fontsize=6.5, color=MUTED, va="bottom")
    a.set_yscale("log")
    a.set_ylim(0.8, 12_000)
    a.set_yticks([1, 10, 100, 1000, 10000])
    a.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
    a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_xlim(-0.3, 1.5)
    a.set_xticks([0, 1])
    a.set_xticklabels(["Development", "External"])
    a.set_ylabel("Odds ratio, clindamycin resistance\n(log scale, 95% CI)")
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.spines["bottom"].set_visible(False)
    panel(a, "a")

    cols = ["#184f95", B400, B250, "#d6d5cf"]
    txt = ["white", "white", INK, INK]
    rows = [("External\n(n = 333)", [86, 139, 84, 24]),
            ("Development\n(n = 830)", [482, 275, 8, 65])]
    for i, (lab, counts) in enumerate(rows):
        n = sum(counts)
        left = 0.0
        for j, (c, col, tc) in enumerate(zip(counts, cols, txt)):
            w = 100 * c / n
            b.barh(i, w, left=left, height=0.55, color=col, edgecolor="white", linewidth=1.2)
            if w >= 12:
                b.text(left + w / 2, i, f"{w:.1f}%", ha="center", va="center", fontsize=7.5, color=tc)
            left += w
        b.text(101.5, i, f"{100 * counts[3] / n:.1f}%\nunexplained", ha="left", va="center",
               fontsize=6.8, color=INK2, clip_on=False)
    b.set_yticks([0, 1])
    b.set_yticklabels([r[0] for r in rows])
    b.set_xlim(0, 100)
    b.set_ylim(-0.5, 1.5)
    b.set_xticks([0, 25, 50, 75, 100])
    b.set_xticklabels(["0", "25", "50", "75", "100%"])
    b.set_xlabel("Clindamycin-resistant genomes")
    b.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="none") for c in cols]
    leg_labels = [gene("erm|(B)") + " or " + gene("erm|(C)"),
                  gene("erm|(A)") + ", without " + gene("erm|(B)") + "/(C)",
                  "Other lincosamide gene (" + gene("erm|(T)") + " externally)",
                  "No known gene"]
    b.legend(handles, leg_labels, loc="upper left", bbox_to_anchor=(-0.02, -0.20), ncol=2,
             frameon=False, fontsize=6.8, handlelength=1.0, handleheight=1.0, columnspacing=0.8)
    panel(b, "b")
    save(fig, "fig_clindamycin")


if __name__ == "__main__":
    fig_pipeline()
    fig_performance()
    fig_models()
    fig_transfer()
    fig_clindamycin()
    print("done:", sorted(f for f in os.listdir(OUT) if f.endswith((".pdf", ".png"))))
