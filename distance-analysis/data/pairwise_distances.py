#!/usr/bin/env python
"""
Full per-sequence pairwise distance table
=========================================

For EVERY (OXT sequence, generated sequence) pair, emit one row carrying both
the Hamming and the Levenshtein distance.

  1. each of the 29 unique OXT sequences is trimmed of its leading and
     trailing 6 C's, leaving the 18 nt core
  2. every trimmed sequence is compared against every sequence in both
     result_nn CSV files
  3. both distances are computed for every comparison
  4. results are written as long-format ("tidy") CSVs -- one row per pair

  29 x 19,983 = 579,507 pairs   (R6)
  29 x  1,378 =  39,962 pairs   (VAE)
  ---------------------------------
                619,469 pairs   total

The distance kernels are imported from distance_analysis.py so both scripts
use identical, already-validated math.

Inputs are read-only; all output goes to this script's directory.
"""

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import distance_analysis as da   # kernels + the 29 trimmed references

OUT = da.OUT
L = da.L

# --------------------------------------------------------------------------
# step 1 -- the 29 trimmed reference sequences
# --------------------------------------------------------------------------
oxt = da.oxt          # deduplicated, 29 rows, with 'core' column
REF = da.REF          # (29, 18) encoded
REF_IDS = da.REF_IDS
REF_CORES = da.REF_CORES

print("step 1 - OXT references")
print(f"  rows in OXT.csv                 : {len(da.oxt_raw)}")
print(f"  unique sequences                : {len(oxt)}")
print(f"  trimmed (first/last 6 C removed): {L} nt each")
print(f"  example: {oxt['sequence'].iloc[0]} -> {oxt['core'].iloc[0]}")
print()

DATASETS = [
    ("r6", "result_nn_r6.csv", da.R6_CSV),
    ("vae", "result_nn_vae_ordered_base_on_arch.csv", da.VAE_CSV),
]

COLS = [
    "oxt_id",
    "oxt_sequence_original",
    "oxt_sequence_trimmed",
    "result_file",
    "result_sequence",
    "architecture",
    "hamming_distance",
    "levenshtein_distance",
    "hamming_similarity_pct",
    "levenshtein_similarity_pct",
]

parts = []
for tag, fname, path in DATASETS:
    df = pd.read_csv(path)
    seqs = df["sequence"].tolist()
    n = len(seqs)
    Q = da.encode(seqs)

    # steps 2 + 3 -- every trimmed reference vs every generated sequence
    lev = da.levenshtein_matrix(Q, REF)     # (n, 29)
    ham = da.hamming_matrix(Q, REF)         # (n, 29)

    arch = (
        df["architecture"].astype(str).values
        if "architecture" in df.columns
        else np.full(n, "", dtype=object)
    )

    # long format, ordered reference-major: all n comparisons for OXT_01,
    # then all n for OXT_02, ...
    long = pd.DataFrame(
        {
            "oxt_id": np.repeat(REF_IDS, n),
            "oxt_sequence_original": np.repeat(oxt["sequence"].values, n),
            "oxt_sequence_trimmed": np.repeat(REF_CORES, n),
            "result_file": fname,
            "result_sequence": np.tile(seqs, len(REF_IDS)),
            "architecture": np.tile(arch, len(REF_IDS)),
            "hamming_distance": ham.T.ravel(),
            "levenshtein_distance": lev.T.ravel(),
        }
    )
    long["hamming_similarity_pct"] = (100 * (1 - long.hamming_distance / L)).round(2)
    long["levenshtein_similarity_pct"] = (
        100 * (1 - long.levenshtein_distance / L)
    ).round(2)
    long = long[COLS]

    out_path = os.path.join(OUT, f"pairwise_distances_{tag}.csv")
    long.to_csv(out_path, index=False)
    parts.append(long)

    print(f"step 2/3 - {fname}")
    print(f"  generated sequences : {n:,}")
    print(f"  pairs written       : {len(long):,}  ({len(REF_IDS)} x {n:,})")
    print(f"  hamming     min/mean/max : {long.hamming_distance.min()} / "
          f"{long.hamming_distance.mean():.3f} / {long.hamming_distance.max()}")
    print(f"  levenshtein min/mean/max : {long.levenshtein_distance.min()} / "
          f"{long.levenshtein_distance.mean():.3f} / {long.levenshtein_distance.max()}")
    print(f"  -> {os.path.basename(out_path)} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)")
    print()

# step 4 -- combined table
combined = pd.concat(parts, ignore_index=True)
combined_path = os.path.join(OUT, "pairwise_distances_all.csv")
combined.to_csv(combined_path, index=False)
print("step 4 - combined table")
print(f"  total pairs : {len(combined):,}")
print(f"  -> pairwise_distances_all.csv "
      f"({os.path.getsize(combined_path) / 1e6:.1f} MB)")
print()

# --------------------------------------------------------------------------
# companion summary: one row per OXT sequence per result file
# --------------------------------------------------------------------------
summ = (
    combined.groupby(["oxt_id", "oxt_sequence_trimmed", "result_file"])
    .agg(
        n_comparisons=("result_sequence", "size"),
        ham_min=("hamming_distance", "min"),
        ham_mean=("hamming_distance", "mean"),
        ham_max=("hamming_distance", "max"),
        lev_min=("levenshtein_distance", "min"),
        lev_mean=("levenshtein_distance", "mean"),
        lev_max=("levenshtein_distance", "max"),
    )
    .round(3)
    .reset_index()
)
# closest generated sequence for each (oxt, file), by Levenshtein then Hamming
best = (
    combined.sort_values(["levenshtein_distance", "hamming_distance"])
    .groupby(["oxt_id", "result_file"], as_index=False)
    .first()[["oxt_id", "result_file", "result_sequence",
              "levenshtein_distance", "hamming_distance"]]
    .rename(columns={"result_sequence": "closest_sequence",
                     "levenshtein_distance": "closest_lev",
                     "hamming_distance": "closest_ham"})
)
summ = summ.merge(best, on=["oxt_id", "result_file"], how="left")
summ["best_lev_similarity_pct"] = (100 * (1 - summ.closest_lev / L)).round(2)
summ["n_within_1_lev"] = (
    combined[combined.levenshtein_distance <= 1]
    .groupby(["oxt_id", "result_file"]).size()
    .reindex(pd.MultiIndex.from_frame(summ[["oxt_id", "result_file"]]))
    .fillna(0).astype(int).values
)
summ["n_within_3_lev"] = (
    combined[combined.levenshtein_distance <= 3]
    .groupby(["oxt_id", "result_file"]).size()
    .reindex(pd.MultiIndex.from_frame(summ[["oxt_id", "result_file"]]))
    .fillna(0).astype(int).values
)
summ_path = os.path.join(OUT, "pairwise_summary_by_oxt.csv")
summ.to_csv(summ_path, index=False)

print("companion - per-OXT-sequence summary")
print(f"  rows: {len(summ)} (29 sequences x 2 result files)")
print(f"  -> pairwise_summary_by_oxt.csv")
print()
print(summ[["oxt_id", "oxt_sequence_trimmed", "result_file", "lev_min",
            "ham_min", "lev_mean", "closest_sequence"]].to_string(index=False))

# --------------------------------------------------------------------------
# consistency check against the matrices written by distance_analysis.py
# --------------------------------------------------------------------------
print()
print("consistency check vs distance_analysis.py outputs")
ok = True
for tag, fname, _ in DATASETS:
    per = pd.read_csv(os.path.join(OUT, f"per_sequence_{tag}.csv"))
    sub = combined[combined.result_file == fname]
    got = (
        sub.groupby("result_sequence")
        .agg(lev=("levenshtein_distance", "min"), ham=("hamming_distance", "min"))
        .reindex(per.sequence)
    )
    a = bool((got.lev.values == per.lev_min.values).all())
    b = bool((got.ham.values == per.ham_min.values).all())
    ok &= a and b
    print(f"  {tag:<4} min-Levenshtein agrees: {a}   min-Hamming agrees: {b}")
print("  " + ("ALL CONSISTENT" if ok else "MISMATCH"))

print(f"\nOutputs written to: {OUT}")
sys.exit(0 if ok else 1)
