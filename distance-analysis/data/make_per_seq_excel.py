#!/usr/bin/env python
"""
One Excel workbook per original OXT sequence, PER SOURCE FILE
=============================================================

For every row of OXT.csv two workbooks are written into per_seq/:

    oxt_<row>_r6.xlsx    sequences from result_nn_r6.csv                    (19,983)
    oxt_<row>_vae.xlsx   sequences from result_nn_vae_ordered_base_on_arch  ( 1,378)

where <row> is the 1-based row index of that sequence in OXT.csv (header
excluded). The two sources are never mixed in one file.

Each workbook has four sheets:

  1. "Distances"           exactly three columns --
                             Sequence | Levenshtein Distance | Hamming Distance
                           Row 2 is the reference sequence itself, trimmed of
                           its leading/trailing 6 C's, with both distances 0.
                           Rows 3+ are that source file's sequences.
                           Characters are coloured per position: GREEN where
                           the character matches the reference, AMBER where it
                           differs.

  2. "Source Location"     where every sequence came from -- source file,
                           physical row in that CSV, line number in the file,
                           and the CSV's own row_number column.

  3. "Aligned (highlighted)"
                           the same rows with one column per position, so each
                           character can carry a real LIGHT GREEN / LIGHT
                           YELLOW cell highlight (the .xlsx format supports
                           per-character font colour but not per-character
                           background fill, so an in-cell highlight cannot be
                           expressed on sheet 1).

  4. "Notes"               provenance and colour legend.

Inputs are read-only; output goes to data/per_seq/.
"""

import contextlib
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import xlsxwriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import distance_analysis as da

OUT = os.path.join(da.OUT, "per_seq")
os.makedirs(OUT, exist_ok=True)
L = da.L

GREEN_BG, GREEN_FG = "#C6EFCE", "#006100"      # light green fill / dark green text
YELLOW_BG, YELLOW_FG = "#FFEB9C", "#9C6500"    # light yellow fill / dark amber text
HDR_BG = "#DCE6F1"

SOURCES = [
    ("r6", "result_nn_r6.csv", da.R6_CSV),
    ("vae", "result_nn_vae_ordered_base_on_arch.csv", da.VAE_CSV),
]

LOC_COLS = [
    ("workbook_row", 14, "row of this sequence on the Distances sheet"),
    ("sequence", 24, "the 18 nt sequence"),
    ("source_file", 40, "CSV the sequence was read from"),
    ("csv_data_row", 14, "1-based position in that CSV, header excluded"),
    ("csv_line_number", 16, "line number in the file itself (header = line 1)"),
    ("row_number_in_csv", 18, "the CSV's own row_number column (-1 where unset)"),
    ("architecture", 14, "the CSV's architecture column (-1 where unset)"),
]


def build(path, seqs, lev, ham, ref_core, loc, meta):
    """Write one workbook. seqs/lev/ham/loc include the reference row first."""
    wb = xlsxwriter.Workbook(path, {"constant_memory": True})

    f_hdr = wb.add_format({"bold": True, "bg_color": HDR_BG, "border": 1,
                           "align": "center", "valign": "vcenter", "text_wrap": True})
    f_num = wb.add_format({"align": "center"})
    f_seq = wb.add_format({"font_name": "Consolas"})
    r_match = wb.add_format({"font_name": "Consolas", "font_color": GREEN_FG, "bold": True})
    r_diff = wb.add_format({"font_name": "Consolas", "font_color": YELLOW_FG, "bold": True})
    c_match = wb.add_format({"font_name": "Consolas", "bg_color": GREEN_BG,
                             "font_color": GREEN_FG, "align": "center",
                             "border": 1, "border_color": "#FFFFFF"})
    c_diff = wb.add_format({"font_name": "Consolas", "bg_color": YELLOW_BG,
                            "font_color": YELLOW_FG, "align": "center", "bold": True,
                            "border": 1, "border_color": "#FFFFFF"})

    # ---- sheet 1: the three requested columns ----
    ws = wb.add_worksheet("Distances")
    ws.freeze_panes(1, 0)
    ws.set_column(0, 0, 26)
    ws.set_column(1, 2, 20)
    for c, name in enumerate(["Sequence", "Levenshtein Distance", "Hamming Distance"]):
        ws.write(0, c, name, f_hdr)

    # ---- sheet 2: where each sequence came from ----
    wl = wb.add_worksheet("Source Location")
    wl.freeze_panes(1, 0)
    for c, (name, width, _) in enumerate(LOC_COLS):
        wl.set_column(c, c, width)
        wl.write(0, c, name, f_hdr)

    # ---- sheet 3: per-position highlight ----
    wa = wb.add_worksheet("Aligned (highlighted)")
    wa.freeze_panes(1, 3)
    wa.set_column(0, 0, 26)
    wa.set_column(1, 2, 20)
    wa.set_column(3, 3 + L - 1, 3.2)
    for c, name in enumerate(["Sequence", "Levenshtein Distance", "Hamming Distance"]):
        wa.write(0, c, name, f_hdr)
    for i in range(L):
        wa.write(0, 3 + i, i + 1, f_hdr)

    ref = ref_core
    for r, (s, dl, dh, lc) in enumerate(zip(seqs, lev, ham, loc), start=1):
        same = [a == b for a, b in zip(s, ref)]

        frags, start = [], 0
        for i in range(1, L + 1):
            if i == L or same[i] != same[start]:
                frags.append((r_match if same[start] else r_diff, s[start:i]))
                start = i
        if len(frags) == 1:                       # uniform status -> plain string
            ws.write_string(r, 0, s, frags[0][0])
        else:
            ws.write_rich_string(r, 0, *[x for f in frags for x in f], f_seq)
        ws.write_number(r, 1, int(dl), f_num)
        ws.write_number(r, 2, int(dh), f_num)

        wl.write_number(r, 0, r + 1, f_num)       # row on the Distances sheet
        wl.write_string(r, 1, s, f_seq)
        wl.write_string(r, 2, lc[0])
        wl.write_number(r, 3, lc[1], f_num)
        wl.write_number(r, 4, lc[2], f_num)
        for c, v in ((5, lc[3]), (6, lc[4])):
            if v == "":
                wl.write_blank(r, c, None)
            elif isinstance(v, str):
                wl.write_string(r, c, v, f_num)
            else:
                wl.write_number(r, c, v, f_num)

        wa.write_string(r, 0, s, f_seq)
        wa.write_number(r, 1, int(dl), f_num)
        wa.write_number(r, 2, int(dh), f_num)
        for i, ch in enumerate(s):
            wa.write_string(r, 3 + i, ch, c_match if same[i] else c_diff)

    # ---- sheet 4: notes ----
    wn = wb.add_worksheet("Notes")
    wn.set_column(0, 0, 32)
    wn.set_column(1, 1, 92)
    wn.write(0, 0, "Field", f_hdr)
    wn.write(0, 1, "Value", f_hdr)
    for i, (k, v) in enumerate(meta, start=1):
        wn.write(i, 0, k)
        wn.write(i, 1, v)

    wb.close()


def main():
    raw = da.oxt_raw.copy()
    raw["row"] = np.arange(1, len(raw) + 1)      # 1-based, header excluded

    # duplicate bookkeeping
    dup_of, seen = {}, {}
    for _, r in raw.iterrows():
        if r["sequence"] in seen:
            dup_of[int(r["row"])] = seen[r["sequence"]]
        else:
            seen[r["sequence"]] = int(r["row"])

    # load each source separately, keeping its physical position
    pools = {}
    for tag, fname, path in SOURCES:
        d = pd.read_csv(path)
        seqs = d["sequence"].tolist()
        rn = (d["row_number"].tolist() if "row_number" in d.columns
              else [""] * len(d))
        arch = (d["architecture"].astype(str).tolist() if "architecture" in d.columns
                else [""] * len(d))
        loc = [(fname, i + 1, i + 2, rn[i], arch[i]) for i in range(len(seqs))]
        pools[tag] = dict(fname=fname, seqs=seqs, loc=loc, Q=da.encode(seqs))
        print(f"{tag:<4} {fname:<40} {len(seqs):>6,} sequences")
    print()

    # distances against each distinct reference core, computed once per source
    cores = raw["core"].tolist()
    uniq = list(dict.fromkeys(cores))
    U = da.encode(uniq)
    idx = {c: i for i, c in enumerate(uniq)}
    for tag in pools:
        pools[tag]["lev"] = da.levenshtein_matrix(pools[tag]["Q"], U)
        pools[tag]["ham"] = da.hamming_matrix(pools[tag]["Q"], U)

    t0 = time.time()
    manifest = []
    for _, row in raw.iterrows():
        rown, core, orig = int(row["row"]), row["core"], row["sequence"]
        k = idx[core]
        dup_note = (f"duplicate of row {dup_of[rown]} in OXT.csv (identical sequence)"
                    if rown in dup_of else "unique sequence")

        for tag in ("r6", "vae"):
            P = pools[tag]
            n = len(P["seqs"])
            seqs = [core] + P["seqs"]
            lev = np.concatenate([[0], P["lev"][:, k]])
            ham = np.concatenate([[0], P["ham"][:, k]])
            loc = [("OXT.csv", rown, rown + 1, "", "")] + P["loc"]

            meta = [
                ("OXT.csv row", rown),
                ("Original sequence (30 nt)", orig),
                ("Reference used (18 nt core)", core),
                ("Trimming", "first 6 and last 6 C removed"),
                ("Duplicate status", dup_note),
                ("Compared source file", P["fname"]),
                ("Sequences compared", n),
                ("Distances sheet row 2", "the reference itself (both distances = 0)"),
                ("Distances sheet rows 3-%d" % (n + 2),
                 "the %d sequences from %s, in file order" % (n, P["fname"])),
                ("Source Location sheet",
                 "physical position of every sequence in its source CSV: "
                 "csv_data_row is 1-based with the header excluded, "
                 "csv_line_number counts the header as line 1"),
                ("row_number_in_csv",
                 "the source CSV's own row_number column - for result_nn_r6.csv it "
                 "runs 11..20000 and is NOT the physical position; for the VAE file "
                 "it is -1 (unset)"),
                ("Green", "character matches the reference at that position"),
                ("Yellow / amber", "character differs from the reference"),
                ("Note on highlighting",
                 "the .xlsx format allows per-character font colour but not "
                 "per-character background fill, so sheet 'Distances' colours the "
                 "letters and sheet 'Aligned (highlighted)' provides the true "
                 "green/yellow cell highlight, one column per position"),
                ("Levenshtein distance", "substitutions + insertions + deletions"),
                ("Hamming distance", "positions that differ (both sequences are 18 nt)"),
                ("Generated by", "make_per_seq_excel.py"),
            ]

            fn = f"oxt_{rown}_{tag}.xlsx"
            path = os.path.join(OUT, fn)
            build(path, seqs, lev, ham, core, loc, meta)
            manifest.append(dict(
                file=fn, oxt_row=rown, source=tag, source_file=P["fname"],
                original_sequence=orig, reference_core=core,
                sequences_compared=n, sheet_rows=len(seqs),
                min_lev=int(lev[1:].min()), min_ham=int(ham[1:].min()),
                mean_lev=round(float(lev[1:].mean()), 3),
                duplicate_of=dup_of.get(rown, ""),
                size_mb=round(os.path.getsize(path) / 1e6, 2)))

        m1, m2 = manifest[-2], manifest[-1]
        print(f"  oxt_{rown:<2} {core}  "
              f"r6: min_lev={m1['min_lev']:>2} {m1['size_mb']:>4.1f}MB   "
              f"vae: min_lev={m2['min_lev']:>2} {m2['size_mb']:>4.1f}MB")

    mf = pd.DataFrame(manifest)
    mf.to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    print()
    print(f"{len(mf)} workbooks ({mf.source.value_counts().to_dict()}), "
          f"{mf.size_mb.sum():.0f} MB total, {time.time() - t0:.0f}s")
    print(f"manifest -> {os.path.join(OUT, 'manifest.csv')}")


if __name__ == "__main__":
    main()
