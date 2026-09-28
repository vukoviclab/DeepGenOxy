#!/usr/bin/env python
"""
Sequence-similarity analysis
============================

Compares the generated / selected sequences in

    result_nn_r6.csv                          (SELEX round-6 pool, 19,983 seqs)
    result_nn_vae_ordered_base_on_arch.csv    (VAE-generated pool, 1,378 seqs)

against the 29 unique original OXT aptamer sequences in OXT.csv.

The OXT sequences are 30 nt with a constant CCCCCC clamp on both ends; the
first and last 6 C's are stripped, leaving the 18 nt variable core, which is
exactly the length of every sequence in the two result files.

Two metrics are computed for every (generated, OXT) pair:

  * Levenshtein (edit) distance  -- substitutions + insertions + deletions
  * Hamming distance             -- position-wise mismatches (equal length)

Both are converted to a percent-identity style similarity:

    similarity % = 100 * (1 - distance / 18)

Read-only with respect to the inputs; every output is written to this script's directory.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

OXT_CSV = os.path.join(BASE, "OXT.csv")
R6_CSV = os.path.join(BASE, "result_nn_r6.csv")
VAE_CSV = os.path.join(BASE, "result_nn_vae_ordered_base_on_arch.csv")

CLAMP = 6        # number of C's stripped from each end of the OXT sequences
L = 18           # length of the variable core / of every generated sequence

# --------------------------------------------------------------------------
# encoding + distance kernels
# --------------------------------------------------------------------------
_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode(seqs) -> np.ndarray:
    """(n,) sequences of equal length -> (n, L) uint8 array."""
    arr = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), -1)
    out = np.full(arr.shape, 255, dtype=np.uint8)
    for ch, v in _MAP.items():
        out[arr == ord(ch)] = v
    if (out == 255).any():
        raise ValueError("non-ACGT character encountered")
    return out


def hamming_matrix(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """(n,L) x (m,L) -> (n,m) int16 position-wise mismatch counts."""
    return (q[:, None, :] != r[None, :, :]).sum(axis=2).astype(np.int16)


def levenshtein_matrix(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    (n,L) x (m,L) -> (n,m) int16 edit distances.

    Full Wagner-Fischer DP, vectorised across the n query sequences: for each
    reference we sweep the DP table row by row, every cell update being a
    numpy op over the whole query batch at once.
    """
    n, lq = q.shape
    m, lr = r.shape
    out = np.empty((n, m), dtype=np.int16)

    for k in range(m):
        ref = r[k]
        prev = np.tile(np.arange(lq + 1, dtype=np.int32), (n, 1))
        cur = np.empty_like(prev)
        for i in range(1, lr + 1):
            cur[:, 0] = i
            ref_c = ref[i - 1]
            for j in range(1, lq + 1):
                cost = (q[:, j - 1] != ref_c).astype(np.int32)
                cur[:, j] = np.minimum(
                    np.minimum(cur[:, j - 1] + 1, prev[:, j] + 1),
                    prev[:, j - 1] + cost,
                )
            prev, cur = cur, prev
        out[:, k] = prev[:, lq].astype(np.int16)
    return out


def sim_pct(d, length=L):
    return 100.0 * (1.0 - d / length)


# --------------------------------------------------------------------------
# load references
# --------------------------------------------------------------------------
oxt_raw = pd.read_csv(OXT_CSV)
oxt_raw["core"] = oxt_raw["sequence"].str[CLAMP:-CLAMP]

# 30 rows but 29 unique sequences (one exact duplicate) -> keep first occurrence
oxt = oxt_raw.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)
oxt.insert(0, "oxt_id", [f"OXT_{i + 1:02d}" for i in range(len(oxt))])
assert len(oxt) == 29, f"expected 29 unique OXT sequences, got {len(oxt)}"
assert set(oxt["core"].str.len()) == {L}

dup_rows = oxt_raw[oxt_raw.duplicated(subset="sequence", keep=False)]

REF = encode(oxt["core"].tolist())
REF_IDS = oxt["oxt_id"].tolist()
REF_CORES = oxt["core"].tolist()

oxt[["oxt_id", "sequence", "core", "dff"]].to_csv(
    os.path.join(OUT, "oxt_reference_29.csv"), index=False
)

# --------------------------------------------------------------------------
# per-dataset analysis
# --------------------------------------------------------------------------
DATASETS = [
    ("R6", "SELEX round-6 pool", R6_CSV, "result_nn_r6.csv"),
    ("VAE", "VAE-generated pool", VAE_CSV, "result_nn_vae_ordered_base_on_arch.csv"),
]

results = {}

for tag, label, path, fname in DATASETS:
    df = pd.read_csv(path)
    seqs = df["sequence"].tolist()
    Q = encode(seqs)

    lev = levenshtein_matrix(Q, REF)
    ham = hamming_matrix(Q, REF)

    lev_min = lev.min(axis=1)
    ham_min = ham.min(axis=1)
    lev_arg = lev.argmin(axis=1)
    ham_arg = ham.argmin(axis=1)

    per = pd.DataFrame(
        {
            "sequence": seqs,
            "lev_min": lev_min,
            "lev_similarity_pct": sim_pct(lev_min).round(2),
            "nearest_oxt_lev": [REF_IDS[i] for i in lev_arg],
            "nearest_oxt_lev_core": [REF_CORES[i] for i in lev_arg],
            "n_oxt_tied_lev": (lev == lev_min[:, None]).sum(axis=1),
            "ham_min": ham_min,
            "ham_similarity_pct": sim_pct(ham_min).round(2),
            "nearest_oxt_ham": [REF_IDS[i] for i in ham_arg],
            "nearest_oxt_ham_core": [REF_CORES[i] for i in ham_arg],
            "n_oxt_tied_ham": (ham == ham_min[:, None]).sum(axis=1),
            "lev_mean_all29": lev.mean(axis=1).round(3),
            "ham_mean_all29": ham.mean(axis=1).round(3),
        }
    )

    # carry through useful metadata columns when present
    for col in ["architecture", "average_probability", "max_probability",
                "min_probability", "reads"]:
        if col in df.columns:
            per[col] = df[col].values

    per.to_csv(os.path.join(OUT, f"per_sequence_{tag.lower()}.csv"), index=False)

    # full distance matrices (rows = generated sequence, cols = 29 OXT)
    pd.DataFrame(lev, index=seqs, columns=REF_IDS).to_csv(
        os.path.join(OUT, f"matrix_levenshtein_{tag.lower()}.csv"),
        index_label="sequence",
    )
    pd.DataFrame(ham, index=seqs, columns=REF_IDS).to_csv(
        os.path.join(OUT, f"matrix_hamming_{tag.lower()}.csv"),
        index_label="sequence",
    )

    results[tag] = dict(
        label=label, fname=fname, df=df, per=per, lev=lev, ham=ham,
        lev_min=lev_min, ham_min=ham_min,
    )

# --------------------------------------------------------------------------
# summary tables
# --------------------------------------------------------------------------
def describe(v):
    v = np.asarray(v, dtype=float)
    return dict(
        n=int(v.size),
        min=float(v.min()),
        p1=float(np.percentile(v, 1)),
        q1=float(np.percentile(v, 25)),
        median=float(np.median(v)),
        mean=float(v.mean()),
        q3=float(np.percentile(v, 75)),
        max=float(v.max()),
        std=float(v.std(ddof=1)),
    )


summary_rows = []
for tag in results:
    R = results[tag]
    for metric, vals in [("Levenshtein", R["lev_min"]), ("Hamming", R["ham_min"])]:
        d = describe(vals)
        d.update(
            dataset=tag,
            source=R["fname"],
            metric=metric,
            mean_similarity_pct=round(sim_pct(d["mean"]), 2),
            best_similarity_pct=round(sim_pct(d["min"]), 2),
        )
        summary_rows.append(d)

summary = pd.DataFrame(summary_rows)[
    ["dataset", "source", "metric", "n", "min", "p1", "q1", "median", "mean",
     "q3", "max", "std", "mean_similarity_pct", "best_similarity_pct"]
].round(3)
summary.to_csv(os.path.join(OUT, "summary_statistics.csv"), index=False)

# distance distribution tables
dist_tables = {}
for tag in results:
    R = results[tag]
    rows = []
    n = len(R["lev_min"])
    for d in range(0, L + 1):
        lc = int((R["lev_min"] == d).sum())
        hc = int((R["ham_min"] == d).sum())
        if lc == 0 and hc == 0 and d > R["ham_min"].max():
            continue
        rows.append(
            dict(
                distance=d,
                similarity_pct=round(sim_pct(d), 1),
                lev_count=lc,
                lev_pct=round(100 * lc / n, 3),
                lev_cum_pct=round(100 * (R["lev_min"] <= d).sum() / n, 3),
                ham_count=hc,
                ham_pct=round(100 * hc / n, 3),
                ham_cum_pct=round(100 * (R["ham_min"] <= d).sum() / n, 3),
            )
        )
    t = pd.DataFrame(rows)
    dist_tables[tag] = t
    t.to_csv(os.path.join(OUT, f"distance_distribution_{tag.lower()}.csv"), index=False)

# per-OXT-reference view: how well is each original sequence recovered?
per_ref_rows = []
for i, rid in enumerate(REF_IDS):
    row = dict(oxt_id=rid, core=REF_CORES[i], dff=oxt["dff"].iloc[i])
    for tag in results:
        R = results[tag]
        lev_col, ham_col = R["lev"][:, i], R["ham"][:, i]
        row[f"{tag}_best_lev"] = int(lev_col.min())
        row[f"{tag}_best_ham"] = int(ham_col.min())
        row[f"{tag}_best_lev_sim_pct"] = round(sim_pct(lev_col.min()), 2)
        row[f"{tag}_mean_lev"] = round(float(lev_col.mean()), 3)
        row[f"{tag}_n_nearest_lev"] = int((R["per"]["nearest_oxt_lev"] == rid).sum())
        row[f"{tag}_n_within_3_lev"] = int((lev_col <= 3).sum())
        row[f"{tag}_n_within_5_lev"] = int((lev_col <= 5).sum())
    per_ref_rows.append(row)

per_ref = pd.DataFrame(per_ref_rows)
per_ref.to_csv(os.path.join(OUT, "per_oxt_reference_summary.csv"), index=False)

# per-architecture breakdown for the VAE pool
arch = (
    results["VAE"]["per"]
    .groupby("architecture")
    .agg(
        n=("sequence", "size"),
        lev_min_best=("lev_min", "min"),
        lev_min_mean=("lev_min", "mean"),
        lev_min_median=("lev_min", "median"),
        lev_min_max=("lev_min", "max"),
        ham_min_best=("ham_min", "min"),
        ham_min_mean=("ham_min", "mean"),
        ham_min_median=("ham_min", "median"),
        ham_min_max=("ham_min", "max"),
        mean_sim_pct=("lev_similarity_pct", "mean"),
    )
    .round(3)
    .reset_index()
    .sort_values("lev_min_mean")
)
arch.to_csv(os.path.join(OUT, "vae_by_architecture.csv"), index=False)

# closest hits overall
top_rows = []
for tag in results:
    p = results[tag]["per"].nsmallest(25, ["lev_min", "ham_min"]).copy()
    p.insert(0, "dataset", tag)
    top_rows.append(
        p[["dataset", "sequence", "lev_min", "lev_similarity_pct", "ham_min",
           "ham_similarity_pct", "nearest_oxt_lev", "nearest_oxt_lev_core"]]
    )
top_hits = pd.concat(top_rows, ignore_index=True)
top_hits.to_csv(os.path.join(OUT, "top_closest_matches.csv"), index=False)

# --------------------------------------------------------------------------
# supplementary: exclude the poly-C control references
#
# Two of the 29 OXT entries are poly-C controls rather than real aptamer
# candidates (OXT_19 = 18xC, OXT_29 = 18xC with a single G).  The R6 pool is
# full of poly-C artefacts, so those two references dominate its best hits and
# flatter the headline numbers.  Repeat the analysis over the 27 genuine
# aptamer references so the comparison reflects real sequence recovery.
# --------------------------------------------------------------------------
CONTROL_MASK = np.array(
    [c.count("C") >= L - 1 for c in REF_CORES]
)   # >=17 of 18 positions are C
CONTROL_IDS = [rid for rid, m in zip(REF_IDS, CONTROL_MASK) if m]
KEEP = ~CONTROL_MASK

nc_summary_rows = []
nc_dist_tables = {}
for tag in results:
    R = results[tag]
    lev_nc = R["lev"][:, KEEP].min(axis=1)
    ham_nc = R["ham"][:, KEEP].min(axis=1)
    R["lev_min_nc"], R["ham_min_nc"] = lev_nc, ham_nc

    for metric, vals in [("Levenshtein", lev_nc), ("Hamming", ham_nc)]:
        d = describe(vals)
        d.update(dataset=tag, source=R["fname"], metric=metric,
                 mean_similarity_pct=round(sim_pct(d["mean"]), 2),
                 best_similarity_pct=round(sim_pct(d["min"]), 2))
        nc_summary_rows.append(d)

    n = len(lev_nc)
    rows = []
    for d in range(0, L + 1):
        lc, hc = int((lev_nc == d).sum()), int((ham_nc == d).sum())
        if lc == 0 and hc == 0 and d > ham_nc.max():
            continue
        rows.append(dict(distance=d, similarity_pct=round(sim_pct(d), 1),
                         lev_count=lc, lev_pct=round(100 * lc / n, 3),
                         lev_cum_pct=round(100 * (lev_nc <= d).sum() / n, 3),
                         ham_count=hc, ham_pct=round(100 * hc / n, 3),
                         ham_cum_pct=round(100 * (ham_nc <= d).sum() / n, 3)))
    nc_dist_tables[tag] = pd.DataFrame(rows)

nc_summary = pd.DataFrame(nc_summary_rows)[
    ["dataset", "source", "metric", "n", "min", "p1", "q1", "median", "mean",
     "q3", "max", "std", "mean_similarity_pct", "best_similarity_pct"]
].round(3)
nc_summary.to_csv(
    os.path.join(OUT, "summary_statistics_excluding_polyC_controls.csv"), index=False
)
for tag in results:
    nc_dist_tables[tag].to_csv(
        os.path.join(OUT, f"distance_distribution_{tag.lower()}_excl_controls.csv"),
        index=False,
    )

# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_R6, C_VAE = "#4C72B0", "#DD8452"

# fig 1: distribution of minimum distance, both metrics, both datasets
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, metric, key in zip(axes, ["Levenshtein", "Hamming"], ["lev_min", "ham_min"]):
    bins = np.arange(-0.5, L + 1.5, 1)
    for tag, color in [("R6", C_R6), ("VAE", C_VAE)]:
        v = results[tag]["per"][key].values
        w = np.ones_like(v, dtype=float) * 100.0 / len(v)
        ax.hist(v, bins=bins, weights=w, alpha=0.6, label=f"{tag} (n={len(v):,})",
                color=color, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(f"minimum {metric} distance to any of the 29 OXT cores")
    ax.set_ylabel("% of sequences in pool")
    ax.set_title(f"{metric} distance", fontweight="bold")
    ax.set_xticks(range(0, L + 1, 2))
    ax.legend(frameon=False)
fig.suptitle("Distance from generated sequences to the nearest original OXT sequence (18 nt core)",
             fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_distance_distributions.png"), bbox_inches="tight")
plt.close(fig)

# fig 2: cumulative coverage
fig, ax = plt.subplots(figsize=(7, 4.4))
for tag, color in [("R6", C_R6), ("VAE", C_VAE)]:
    t = dist_tables[tag]
    ax.plot(t["distance"], t["lev_cum_pct"], "-o", ms=3.5, color=color,
            label=f"{tag} – Levenshtein")
    ax.plot(t["distance"], t["ham_cum_pct"], "--s", ms=3, color=color, alpha=0.65,
            label=f"{tag} – Hamming")
ax.set_xlabel("distance threshold d")
ax.set_ylabel("% of sequences with min distance ≤ d")
ax.set_title("Cumulative similarity to the OXT reference set", fontweight="bold")
ax.set_xticks(range(0, L + 1, 2))
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_cumulative.png"), bbox_inches="tight")
plt.close(fig)

# fig 3: VAE per-architecture
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
order = arch["architecture"].tolist()
for ax, key, metric in zip(axes, ["lev_min", "ham_min"], ["Levenshtein", "Hamming"]):
    data = [results["VAE"]["per"].loc[results["VAE"]["per"]["architecture"] == a, key].values
            for a in order]
    bp = ax.boxplot(data, labels=[f"{a}\n(n={len(d)})" for a, d in zip(order, data)],
                    showmeans=True, patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor(C_VAE)
        patch.set_alpha(0.45)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel(f"min {metric} distance to OXT")
    ax.set_title(f"{metric}", fontweight="bold")
fig.suptitle("VAE pool: distance to nearest OXT sequence, by generating architecture",
             fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_vae_by_architecture.png"), bbox_inches="tight")
plt.close(fig)

# fig 4: heatmap - best Levenshtein distance achieved against each OXT reference
fig, ax = plt.subplots(figsize=(9, 5.2))
mat = np.vstack([per_ref["R6_best_lev"].values, per_ref["VAE_best_lev"].values]).astype(float)
im = ax.imshow(mat, cmap="viridis_r", aspect="auto", vmin=0, vmax=mat.max())
ax.set_yticks([0, 1], ["R6 pool", "VAE pool"])
ax.set_xticks(range(len(REF_IDS)), REF_IDS, rotation=90, fontsize=7)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=7,
                color="white" if mat[i, j] > mat.max() * 0.55 else "black")
ax.set_title("Best (minimum) Levenshtein distance achieved against each original OXT sequence",
             fontweight="bold")
ax.grid(False)
fig.colorbar(im, ax=ax, label="edit distance", pad=0.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_per_reference_best.png"), bbox_inches="tight")
plt.close(fig)

# fig 5: how many sequences map to each OXT reference as nearest neighbour
fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
x = np.arange(len(REF_IDS))
for ax, tag, color in zip(axes, ["R6", "VAE"], [C_R6, C_VAE]):
    counts = per_ref[f"{tag}_n_nearest_lev"].values
    ax.bar(x, 100 * counts / counts.sum(), color=color, alpha=0.85)
    ax.set_ylabel(f"% of {tag} pool")
    ax.set_title(f"{tag}: nearest-OXT assignment (Levenshtein, ties → first match)",
                 fontweight="bold", fontsize=9.5)
axes[-1].set_xticks(x, REF_IDS, rotation=90, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_nearest_reference_counts.png"), bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------
# console digest (captured into the report)
# --------------------------------------------------------------------------
print("=" * 78)
print("OXT reference set")
print("=" * 78)
print(f"rows in OXT.csv                : {len(oxt_raw)}")
print(f"unique sequences               : {len(oxt)}")
print(f"duplicate dropped              : {dup_rows['sequence'].iloc[0]} "
      f"(rows {list(dup_rows.index + 2)})")
print(f"core length after stripping 6C : {L} nt\n")

print("=" * 78)
print("Summary statistics of the minimum distance to the 29 OXT cores")
print("=" * 78)
print(summary.to_string(index=False))
print()

for tag in results:
    print("=" * 78)
    print(f"{tag} — {results[tag]['label']}  ({results[tag]['fname']})")
    print("=" * 78)
    print(dist_tables[tag].to_string(index=False))
    print()

print("=" * 78)
print("VAE pool by generating architecture")
print("=" * 78)
print(arch.to_string(index=False))
print()

print("=" * 78)
print("Per-OXT-reference recovery")
print("=" * 78)
print(per_ref[["oxt_id", "core", "R6_best_lev", "R6_best_ham", "R6_n_within_3_lev",
               "VAE_best_lev", "VAE_best_ham", "VAE_n_within_3_lev"]].to_string(index=False))
print()

print("=" * 78)
print("10 closest sequences per pool")
print("=" * 78)
for tag in results:
    print(f"\n-- {tag} --")
    print(top_hits[top_hits.dataset == tag].head(10).to_string(index=False))

print("\nexact matches (distance 0):")
for tag in results:
    print(f"  {tag}: Levenshtein={int((results[tag]['lev_min'] == 0).sum())}, "
          f"Hamming={int((results[tag]['ham_min'] == 0).sum())}")

print()
print("=" * 78)
print(f"SUPPLEMENTARY — poly-C controls excluded ({', '.join(CONTROL_IDS)}); "
      f"{int(KEEP.sum())} genuine aptamer references")
print("=" * 78)
print(nc_summary.to_string(index=False))
print()
for tag in results:
    print(f"-- {tag} (controls excluded) --")
    print(nc_dist_tables[tag].to_string(index=False))
    print()

print("Outputs written to:", OUT)
