#!/usr/bin/env python
"""
Paper deliverable: nearest-original Hamming distance per generated sequence
===========================================================================

For each generated pool, the core table has exactly four columns:

    sequence              the generated sequence (18 nt)
    nearest_oxt_id        ID of the original OXT sequence it is most similar to
                          (smallest Hamming distance)
    nearest_oxt_sequence  that original sequence, trimmed of its 6 C clamps
    hamming_distance      the value of that smallest Hamming distance

Output is organised one folder per analysis, each self-contained (its CSVs, its
figure, its statistics, and a README):

    paper/
      00_reference_key/                  the 29 originals, IDs, control flags
      01_vae_all_29_references/          VAE  vs all 29 originals
      02_vae_excluding_polyC_controls/   VAE  vs the 27 genuine aptamers
      03_r6_all_29_references/           R6   vs all 29 originals
      04_r6_excluding_polyC_controls/    R6   vs the 27 genuine aptamers
      05_comparison_vae_vs_r6/           both pools on one figure

Figures are publication-grade 600 dpi PNG: ACS Nano / Origin house style, a full
box frame with inward ticks on all four sides, Helvetica-metric type, and a
colour-blind-safe palette validated with validate_palette.py.

Inputs are read-only.
"""

import contextlib
import io
import os
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import distance_analysis as da

ROOT = os.path.join(da.OUT, "paper")
os.makedirs(ROOT, exist_ok=True)
L = da.L

# palette validated with validate_palette.py (OKLab dE, Machado CVD simulation)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, INK3 = "#1a1a1a", "#55555a", "#8c8c92"
GRID = "#e8e8e6"

# ---- publication figure defaults -----------------------------------------
# House style follows the ACS Nano / Origin convention: a full box frame,
# major AND minor ticks pointing inward on all four sides, no gridlines, thick
# data traces, an unframed in-plot legend and bold lower-case panel letters.
# importing distance_analysis above applies ITS rcParams (which switch the top
# and right spines off). Reset to the library defaults first so none of that
# leaks in -- without this the box frame loses two sides and the inward top /
# right ticks end up floating with no axis line to sit on.
plt.rcdefaults()
plt.rcParams.update({
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    # keep math italics (the italic n) in the same family as the body type,
    # otherwise matplotlib silently embeds DejaVu alongside Nimbus Sans
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic",
    "mathtext.bf": "Nimbus Sans:bold",
    "font.size": 9,
    "axes.labelsize": 10, "axes.labelpad": 3.0,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 1.3, "axes.edgecolor": "black",
    "axes.labelcolor": "black", "text.color": "black",
    "xtick.color": "black", "ytick.color": "black",
    "xtick.top": True, "ytick.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
    "xtick.major.size": 3.2, "ytick.major.size": 3.2,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.pad": 4, "ytick.major.pad": 4,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.handlelength": 1.6, "legend.handletextpad": 0.6,
    "legend.labelspacing": 0.45, "legend.borderpad": 0.2,
    "axes.grid": False, "figure.dpi": 150, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

DATASETS = [
    ("vae", "VAE-generated", "result_nn_vae_ordered_base_on_arch.csv", da.VAE_CSV, ORANGE),
    ("r6", "SELEX round 6", "result_nn_r6.csv", da.R6_CSV, BLUE),
]


# --------------------------------------------------------------------------
# reference set
# --------------------------------------------------------------------------
oxt_raw = da.oxt_raw.copy()
oxt_raw["oxt_csv_row"] = np.arange(1, len(oxt_raw) + 1)
oxt = oxt_raw.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)
oxt["nearest_oxt_id"] = [f"OXT_{i + 1:02d}" for i in range(len(oxt))]
assert len(oxt) == 29

REF_CORES = oxt["core"].tolist()
REF_IDS = oxt["nearest_oxt_id"].tolist()
REF_FULL = oxt["sequence"].tolist()
REF_ROW = oxt["oxt_csv_row"].tolist()
REF = da.encode(REF_CORES)
IS_CTRL = np.array([c.count("C") >= L - 1 for c in REF_CORES])
CTRL_IDS = [i for i, m in zip(REF_IDS, IS_CTRL) if m]


def folder(name):
    p = os.path.join(ROOT, name)
    os.makedirs(p, exist_ok=True)
    return p


def nearest(ham, mask=None):
    h = ham[:, mask] if mask is not None else ham
    arg = h.argmin(axis=1)
    mn = h.min(axis=1)
    eq = h == mn[:, None]
    ties = eq.sum(axis=1)
    cols = np.flatnonzero(mask) if mask is not None else np.arange(h.shape[1])
    tied = [[REF_IDS[k] for k in cols[np.flatnonzero(row)]] for row in eq]
    return cols[arg], mn, ties, tied


def describe(v, dataset, variant):
    v = np.asarray(v, dtype=float)
    k2, p = stats.normaltest(v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # scipy 1.17 method= deprecation
        ad = stats.anderson(v, dist="norm")
    return dict(
        dataset=dataset, variant=variant, n=len(v),
        min=int(v.min()), max=int(v.max()), mean=round(v.mean(), 4),
        median=float(np.median(v)), mode=int(stats.mode(v, keepdims=False).mode),
        std=round(v.std(ddof=1), 4), variance=round(v.var(ddof=1), 4),
        q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
        iqr=float(np.percentile(v, 75) - np.percentile(v, 25)),
        skewness=round(float(stats.skew(v)), 4),
        excess_kurtosis=round(float(stats.kurtosis(v)), 4),
        dagostino_k2=round(float(k2), 2), dagostino_p=f"{p:.3e}",
        anderson_stat=round(float(ad.statistic), 3),
        anderson_crit_5pct=float(ad.critical_values[2]),
        normal_at_5pct="no" if ad.statistic > ad.critical_values[2] else "yes",
    )


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def trace(ax, counts, values, colour, label=None, as_percent=False, xmax=14,
          marker="o", normal_fit=True, mean_line=True):
    """One distribution drawn as a thick trace with a marker at each integer."""
    x = counts["hamming_distance"].values
    y = counts["count"].values
    keep = x <= xmax
    x, y = x[keep], y[keep]
    n = len(values)
    h = 100 * y / n if as_percent else y

    mean, sd = values.mean(), values.std(ddof=1)
    if normal_fit:
        xs = np.linspace(0, xmax, 400)
        pdf = stats.norm.pdf(xs, mean, sd) * (100 if as_percent else n)
        ax.plot(xs, pdf, color="0.55", lw=1.2, ls=(0, (4, 2.2)), zorder=2)
    if mean_line:
        ax.axvline(mean, color="black", lw=1.0, ls=(0, (1.2, 1.4)), zorder=3)

    ax.plot(x, h, color=colour, lw=2.2, zorder=4, label=label,
            marker=marker, ms=4.4, mfc=colour, mec="white", mew=0.7,
            clip_on=False)
    return h.max()


def finish(ax, ymax, xmax=14, ylabel="count", xlabel=True, ystep=None):
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(0, xmax + 1, 2))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    if ystep:
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ystep))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(ystep / 2))
    if xlabel:
        ax.set_xlabel("Hamming distance")
    ax.set_ylabel(ylabel)


def emptier_side(counts, values, as_percent=False, xmax=14, frac=0.45):
    """Which top corner is free of data — so the annotation never lands on the
    curve. Fixed placement fails here: the VAE peak sits left of centre and the
    R6 peak sits right of it, so the free corner differs between panels."""
    x = counts["hamming_distance"].values
    y = counts["count"].values
    keep = x <= xmax
    x, y = x[keep], y[keep]
    h = 100 * y / len(values) if as_percent else y
    cut = xmax * frac
    left = h[x <= cut].max(initial=0)
    right = h[x >= xmax - cut].max(initial=0)
    return "left" if left <= right else "right"


def note(ax, text, side="right", y=0.945):
    x, ha = (0.045, "left") if side == "left" else (0.955, "right")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="top",
            fontsize=8.2, linespacing=1.5, zorder=6)


def letter(ax, ch, x=-0.26, y=1.10):
    ax.text(x, y, ch, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", ha="left")


def save(fig, out, stem):
    fig.savefig(os.path.join(out, f"{stem}.png"))
    plt.close(fig)


def readme(path, title, body):
    with open(os.path.join(path, "README.md"), "w") as f:
        f.write(f"# {title}\n\n{body}\n")


# --------------------------------------------------------------------------
# 00 — reference key
# --------------------------------------------------------------------------
d00 = folder("00_reference_key")
key = oxt[["nearest_oxt_id", "oxt_csv_row", "sequence", "core", "dff"]].rename(
    columns={"sequence": "original_sequence_30nt", "core": "trimmed_sequence_18nt"}
).assign(is_polyC_control=IS_CTRL)
key.to_csv(os.path.join(d00, "oxt_id_key.csv"), index=False)

print(f"reference set  : {len(oxt)} unique originals from {len(oxt_raw)} rows "
      f"(row 10 duplicates row 1)")
print(f"poly-C controls: {CTRL_IDS}\n")

# --------------------------------------------------------------------------
# per dataset
# --------------------------------------------------------------------------
FOLDERS = {("vae", False): "01_vae_all_29_references",
           ("vae", True): "02_vae_excluding_polyC_controls",
           ("r6", False): "03_r6_all_29_references",
           ("r6", True): "04_r6_excluding_polyC_controls"}

results, stat_rows = {}, []
for tag, label, fname, path, colour in DATASETS:
    df = pd.read_csv(path)
    seqs = df["sequence"].tolist()
    Q = da.encode(seqs)
    ham = da.hamming_matrix(Q, REF).astype(int)
    lev = da.levenshtein_matrix(Q, REF).astype(int)
    results[tag] = dict(label=label, fname=fname, colour=colour, n=len(seqs))

    for excl in (False, True):
        out = folder(FOLDERS[(tag, excl)])
        mask = ~IS_CTRL if excl else None
        arg, mn, ties, tied = nearest(ham, mask)
        variant = ("27 genuine originals (poly-C controls excluded)" if excl
                   else "all 29 originals")
        suffix = "_excl_polyC" if excl else ""

        main = pd.DataFrame({
            "sequence": seqs,
            "nearest_oxt_id": [REF_IDS[i] for i in arg],
            "nearest_oxt_sequence": [REF_CORES[i] for i in arg],
            "hamming_distance": mn,
        })
        main.to_csv(os.path.join(out, f"{tag}_nearest_hamming{suffix}.csv"),
                    index=False)

        ext = main.copy()
        ext["oxt_csv_row"] = [REF_ROW[i] for i in arg]
        ext["nearest_oxt_sequence_30nt"] = [REF_FULL[i] for i in arg]
        ext["levenshtein_distance_to_that_oxt"] = [lev[r, i] for r, i in enumerate(arg)]
        ext["n_tied_nearest"] = ties
        ext["tied_oxt_ids"] = [";".join(t) for t in tied]
        ext["nearest_is_polyC_control"] = [bool(IS_CTRL[i]) for i in arg]
        if "architecture" in df.columns:
            ext["architecture"] = df["architecture"].values
        ext.to_csv(os.path.join(out, f"{tag}_nearest_hamming{suffix}_extended.csv"),
                   index=False)

        counts = (pd.Series(mn).value_counts().reindex(range(L + 1), fill_value=0)
                  .rename_axis("hamming_distance").reset_index(name="count"))
        counts["percent"] = (100 * counts["count"] / len(mn)).round(4)
        counts["cumulative_percent"] = counts["percent"].cumsum().round(4)
        counts.to_csv(os.path.join(out, f"hamming_counts_{tag}{suffix}.csv"),
                      index=False)

        row = describe(mn, label, variant)
        stat_rows.append(row)
        pd.DataFrame([row]).to_csv(os.path.join(out, "distribution_statistics.csv"),
                                   index=False)

        v = np.asarray(mn)
        fig, ax = plt.subplots(figsize=(3.35, 2.75))   # ACS single column
        top = trace(ax, counts, v, colour)
        ystep = 50 if tag == "vae" else 2000
        finish(ax, np.ceil(top * 1.18 / ystep) * ystep, ystep=ystep)
        note(ax, f"{label}\n$n$ = {len(v):,}\nmean = {v.mean():.2f}\n"
                 f"median = {np.median(v):.0f}",
             side=emptier_side(counts, v))
        save(fig, out, f"fig_hamming_{tag}{suffix}")

        n_ties = int((np.asarray(ties) > 1).sum())
        row["n_tied"] = n_ties

        print(f"{FOLDERS[(tag, excl)]:<34} rows={len(main):>6,}  "
              f"mean={row['mean']:.2f}  median={row['median']:.0f}  "
              f"skew={row['skewness']:+.2f}")

        results[tag][("counts", excl)] = counts
        results[tag][("mn", excl)] = np.asarray(mn)

# --------------------------------------------------------------------------
# 05 — comparison
# --------------------------------------------------------------------------
d05 = folder("05_comparison_vae_vs_r6")
XMAX = 13

# three panels, ACS double-column width: (a) VAE counts, (b) R6 counts,
# (c) both as percent of pool -- the pools differ 14-fold, so only the
# normalised panel compares their shapes fairly.
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

va, r6 = results["vae"], results["r6"]

top = trace(axes[0], va[("counts", False)], va[("mn", False)], va["colour"])
finish(axes[0], 300, ystep=100)
note(axes[0], "VAE\n$n$ = 1,378",
     side=emptier_side(va[("counts", False)], va[("mn", False)]))
letter(axes[0], "a")

top = trace(axes[1], r6[("counts", False)], r6[("mn", False)], r6["colour"])
finish(axes[1], 8000, ystep=2000)
note(axes[1], "SELEX R6\n$n$ = 19,983",
     side=emptier_side(r6[("counts", False)], r6[("mn", False)]))
letter(axes[1], "b")

for tag, mk in (("vae", "o"), ("r6", "s")):
    R = results[tag]
    trace(axes[2], R[("counts", False)], R[("mn", False)], R["colour"],
          label="VAE" if tag == "vae" else "SELEX R6", as_percent=True,
          marker=mk, normal_fit=False, mean_line=False)
finish(axes[2], 40, ylabel="sequences (%)", ystep=10)
axes[2].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 1.02),
               fontsize=8.2)
letter(axes[2], "c")

fig.tight_layout(w_pad=1.9)
save(fig, d05, "fig_hamming_comparison")

cmp = results["vae"][("counts", False)][["hamming_distance"]].copy()
for tag in ("vae", "r6"):
    c = results[tag][("counts", False)]
    cmp[f"{tag}_count"] = c["count"].values
    cmp[f"{tag}_percent"] = c["percent"].values
cmp.to_csv(os.path.join(d05, "hamming_counts_comparison.csv"), index=False)

stats_df = pd.DataFrame(stat_rows)
stats_df.to_csv(os.path.join(d05, "distribution_statistics_all.csv"), index=False)

# (documented in the single top-level README)

print(f"\n{'05_comparison_vae_vs_r6':<34} two-panel figure + combined tables")

# --------------------------------------------------------------------------
# reference-key README + top-level index; drop the old flat layout
# --------------------------------------------------------------------------
# (documented in the single top-level README)

for stale in os.listdir(ROOT):
    p = os.path.join(ROOT, stale)
    if os.path.isfile(p) and stale != "README.md":
        os.remove(p)

# The two headline four-column tables also sit at the top level, where they are
# easy to find. Byte-identical to the copies inside 01_ and 03_; those folders
# stay self-contained so each analysis can be handed over on its own.
for tag, src_folder in (("vae", "01_vae_all_29_references"),
                        ("r6", "03_r6_all_29_references")):
    name = f"{tag}_nearest_hamming.csv"
    shutil.copyfile(os.path.join(ROOT, src_folder, name),
                    os.path.join(ROOT, name))
    print(f"top level -> {name}")

readme(ROOT, "Paper figures and tables — nearest-original Hamming distance", f"""
## The two requested tables

| File | Sequences from | Rows |
|---|---|---:|
| **[vae_nearest_hamming.csv](vae_nearest_hamming.csv)** | `result_nn_vae_ordered_base_on_arch.csv` | {results['vae']['n']:,} |
| **[r6_nearest_hamming.csv](r6_nearest_hamming.csv)** | `result_nn_r6.csv` | {results['r6']['n']:,} |

Four columns, identical headers in both, rows in source-file order:

| Column | Meaning |
|---|---|
| `sequence` | the generated sequence (18 nt) |
| `nearest_oxt_id` | ID of the original OXT sequence it is most similar to |
| `nearest_oxt_sequence` | that original sequence, trimmed to its 18 nt core |
| `hamming_distance` | the smallest Hamming distance (0–18) — **plot this** |

Plot column 4 as a histogram: x = Hamming distance, y = count. Those counts are
already tabulated in `hamming_counts_vae.csv` / `hamming_counts_r6.csv` (inside
folders 01 and 03) if you would rather not recompute from {results['r6']['n']:,} rows,
and the finished figures are in the same folders.

These two files are byte-identical copies of the ones in `01_…` and `03_…`; the
folders keep their own copy so each analysis stays self-contained.

---

Everything about every folder is documented below — this is the only README.

| Folder | Analysis | n | mean | median |
|---|---|---:|---:|---:|
| [00_reference_key](00_reference_key/) | the 29 originals, IDs, control flags | 29 | — | — |
| [01_vae_all_29_references](01_vae_all_29_references/) | VAE vs all 29 originals | {results['vae']['n']:,} | {stats_df.iloc[0]['mean']:.2f} | {stats_df.iloc[0]['median']:.0f} |
| [02_vae_excluding_polyC_controls](02_vae_excluding_polyC_controls/) | VAE vs the 27 genuine aptamers | {results['vae']['n']:,} | {stats_df.iloc[1]['mean']:.2f} | {stats_df.iloc[1]['median']:.0f} |
| [03_r6_all_29_references](03_r6_all_29_references/) | SELEX round 6 vs all 29 originals | {results['r6']['n']:,} | {stats_df.iloc[2]['mean']:.2f} | {stats_df.iloc[2]['median']:.0f} |
| [04_r6_excluding_polyC_controls](04_r6_excluding_polyC_controls/) | SELEX round 6 vs the 27 genuine aptamers | {results['r6']['n']:,} | {stats_df.iloc[3]['mean']:.2f} | {stats_df.iloc[3]['median']:.0f} |
| [05_comparison_vae_vs_r6](05_comparison_vae_vs_r6/) | both pools, three-panel figure | — | — | — |

**Which to put in the paper.** Folders 02 and 04 exclude the two poly-C controls
(`OXT_19`, `OXT_29`), which are not aptamer candidates and which account for the
entire low-distance tail of the round-6 pool. That is the more defensible pair for
publication; 01 and 03 are the unfiltered equivalents.

## What each analysis folder contains

Every sequence in the source CSV scored by its **smallest Hamming distance** to any
original OXT sequence (each trimmed of its leading and trailing 6 C's, leaving the
18 nt core). Folders 01–04 all hold the same four file types:

| File | Rows | Contents |
|---|---:|---|
| `*_nearest_hamming*.csv` | 1,378 or 19,983 | **the four requested columns** |
| `*_nearest_hamming*_extended.csv` | same | adds OXT.csv row, 30 nt original, Levenshtein, tie disclosure |
| `hamming_counts_*.csv` | 19 | the figure's x/y: distance, count, percent, cumulative |
| `distribution_statistics.csv` | 1 | n, mean, median, mode, SD, IQR, skew, kurtosis, normality |
| `fig_hamming_*.png` | — | the distribution figure |

### The four columns

| Column | Meaning |
|---|---|
| `sequence` | the generated sequence (18 nt), in source-file order |
| `nearest_oxt_id` | ID of the closest original — see [00_reference_key](00_reference_key/) |
| `nearest_oxt_sequence` | that original, trimmed to its 18 nt core |
| `hamming_distance` | the smallest Hamming distance (0–18) |

### 00_reference_key

`oxt_id_key.csv` maps every `nearest_oxt_id` to its row in `OXT.csv`, its untrimmed
30 nt sequence, its 18 nt core, its `dff` value, and whether it is a poly-C control.
`OXT.csv` holds 30 rows but only **29 unique** sequences — row 10 repeats row 1 — so
IDs run `OXT_01`…`OXT_29` over the deduplicated set.

Two entries are **poly-C controls** rather than aptamer candidates: `OXT_19`
(`CCCCCCCCCCCCCCCCCC`) and `OXT_29` (`CCCCCCCCCCCCCCGCCC`). They dominate the
low-distance tail of the round-6 pool, which is why every analysis is provided both
with and without them.

### 05_comparison_vae_vs_r6

Three panels: **a** VAE counts, **b** SELEX R6 counts, **c** both as percent of pool.
Panel c is normalised because the pools differ 14-fold ({results['vae']['n']:,} vs
{results['r6']['n']:,}); only percentages compare their shapes fairly. Also holds
`hamming_counts_comparison.csv` and `distribution_statistics_all.csv` (all four
analyses in one table).

## Caveats

- **Ties.** Some sequences are equidistant from more than one original
  ({stat_rows[0]['n_tied']:,} of {results['vae']['n']:,} VAE, {stat_rows[2]['n_tied']:,} of
  {results['r6']['n']:,} R6). Column 2 reports the lowest-numbered of them;
  `n_tied_nearest` and `tied_oxt_ids` in the extended files record the rest.
  **Column 4 is unaffected.**
- **Discreteness.** The distance is an integer bounded to 0–18, so formal normality
  tests are of limited value; the shape statistics carry the evidence.

## Figure specification

Styled to the ACS Nano / Origin house convention: full box frame, major and
minor ticks pointing **inward on all four sides**, no gridlines, thick data
traces with a marker at every integer, unframed in-plot legend, bold lower-case
panel letters, and no chart titles.

- **Sizing at 100%** — single panels 3.3 in (ACS single column, 8.45 cm);
  the three-panel comparison 7.0 in (double column, 17.8 cm). No rescaling
  needed, so type stays at its intended size.
- **Type** — Nimbus Sans (Helvetica metrics) throughout, including the math
  italics, embedded as subsetted TrueType (Type0). Matplotlib's default Type 3
  is rejected by most journal production systems.
- **Format** — 600 dpi PNG.
- **Colour** — palette checked with `../validate_palette.py` (OKLab ΔE under
  simulated protanopia/deuteranopia), so the two series stay distinguishable
  in colour-blind and greyscale reproduction.

The dashed grey curve on each single-panel figure is a normal distribution with
the same mean and SD — it is a reference for judging shape, not a fitted model.
The dotted vertical line marks the mean.

Regenerate everything with:

```bash
/home/payam/miniconda3/envs/main/bin/python ../make_paper_csv.py
```
""".strip())

# keep README.pdf in step with README.md so the two cannot drift apart
try:
    import md_to_pdf
    md_to_pdf.convert(os.path.join(ROOT, "README.md"))
except SystemExit as e:
    print(f"README.pdf skipped: {e}")
except Exception as e:
    print(f"README.pdf skipped: {type(e).__name__}: {e}")

print("\ndistribution statistics")
print(stats_df[["dataset", "variant", "n", "mean", "median", "std",
                "skewness", "excess_kurtosis"]].to_string(index=False))
print(f"\nOutputs -> {ROOT}")
