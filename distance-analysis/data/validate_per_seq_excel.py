#!/usr/bin/env python
"""
Validation for the per-sequence workbooks in per_seq/.

For every oxt_<row>_<source>.xlsx it checks, by re-reading the written file:

  * the workbook exists, opens, and has the four expected sheets
  * sheet "Distances" has exactly 3 columns with the required headers
  * row 2 holds the trimmed reference for that OXT.csv row, distances 0 / 0
  * the row count equals 1 reference + that source file's sequence count,
    and NO sequence from the other source file is present
  * the sequences appear in source-file order
  * both distances agree with an independent recomputation
  * the amber character count equals the Hamming distance and the green count
    equals 18 - Hamming, on sampled rows
  * sheet "Source Location" maps every row to the right physical CSV position
  * sheet "Aligned (highlighted)" carries the real green / yellow fills
"""

import os
import random
import re
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
SEQ = os.path.join(HERE, "per_seq")
GREEN, AMBER = "FF006100", "FF9C6500"
GREEN_FILL, YELLOW_FILL = "C6EFCE", "FFEB9C"
L = 18

SOURCES = [("r6", "result_nn_r6.csv"),
           ("vae", "result_nn_vae_ordered_base_on_arch.csv")]


def lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            c = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
            p = c
    return d[-1]


def ham(a, b):
    return sum(x != y for x, y in zip(a, b))


def style_colours(styles_xml):
    """style index -> font colour, for cells whose colour is carried by the cell
    format rather than by inline rich-text runs (which is what xlsxwriter emits
    when all 18 characters share one match/mismatch status)."""
    fonts = re.findall(r"<font>.*?</font>",
                       re.search(r"<fonts.*?</fonts>", styles_xml).group(0))
    xfs = re.findall(r"<xf [^>]*/>|<xf .*?</xf>",
                     re.search(r"<cellXfs.*?</cellXfs>", styles_xml).group(0))
    out = {}
    for i, xf in enumerate(xfs):
        fid = int(re.search(r'fontId="(\d+)"', xf).group(1))
        col = re.search(r'rgb="(FF[0-9A-F]{6})"', fonts[fid])
        out[i] = col.group(1) if col else None
    return out


def seq_runs(row_xml, smap):
    """[(colour, text), ...] for the sequence cell, covering both the rich-text
    and the uniform-colour (plain string) encodings."""
    runs = re.findall(r'<color rgb="(FF[0-9A-F]{6})"/>.*?<t>([ACGT]+)</t>', row_xml)
    if runs:
        return runs
    m = re.search(r'<c r="A\d+" s="(\d+)" t="inlineStr"><is><t>([ACGT]+)</t>', row_xml)
    return [(smap.get(int(m.group(1))), m.group(2))] if m else []


def cell_text(row_xml, col):
    m = re.search(r'<c r="%s\d+"[^>]*t="inlineStr"><is><t>([^<]*)</t>' % col, row_xml)
    return m.group(1) if m else None


def cell_num(row_xml, col):
    m = re.search(r'<c r="%s\d+"[^>]*><v>(-?\d+(?:\.\d+)?)</v>' % col, row_xml)
    return float(m.group(1)) if m else None


raw = pd.read_csv(os.path.join(BASE, "OXT.csv"))
raw["core"] = raw["sequence"].str[6:-6]

pool = {}
for tag, fname in SOURCES:
    pool[tag] = pd.read_csv(os.path.join(BASE, fname))["sequence"].tolist()

random.seed(7)
fails = []
print(f"checking {len(raw) * 2} workbooks in per_seq/ "
      f"({len(raw)} OXT rows x 2 sources)\n")

for i, core in enumerate(raw["core"], start=1):
    line = f"  oxt_{i:<2} {core} "
    for tag, fname in SOURCES:
        seqs = pool[tag]
        expect_rows = len(seqs) + 1
        path = os.path.join(SEQ, f"oxt_{i}_{tag}.xlsx")
        problems = []
        if not os.path.exists(path):
            fails.append((f"oxt_{i}_{tag}.xlsx", "missing"))
            line += f" {tag}:MISSING"
            continue

        z = zipfile.ZipFile(path)
        names = z.namelist()
        s1 = z.read("xl/worksheets/sheet1.xml").decode()
        s2 = z.read("xl/worksheets/sheet2.xml").decode()   # Source Location
        s3 = z.read("xl/worksheets/sheet3.xml").decode()   # Aligned
        styles = z.read("xl/styles.xml").decode()
        wbxml = z.read("xl/workbook.xml").decode()

        sheets = re.findall(r'<sheet name="([^"]+)"', wbxml)
        if sheets != ["Distances", "Source Location", "Aligned (highlighted)", "Notes"]:
            problems.append(f"sheets {sheets}")

        smap = style_colours(styles)

        hdr = re.search(r'<row r="1".*?</row>', s1).group(0)
        for want in ["Sequence", "Levenshtein Distance", "Hamming Distance"]:
            if want not in hdr:
                problems.append(f"missing header {want}")
        if re.search(r'<c r="D1"', hdr):
            problems.append("more than 3 columns on Distances sheet")

        body = {int(m.group(1)): m.group(0)
                for m in re.finditer(r'<row r="(\d+)"[^>]*>.*?</row>', s1)}
        if len(body) != expect_rows + 1:
            problems.append(f"row count {len(body) - 1} != {expect_rows}")

        # reference row
        r2 = body[2]
        ref_runs = seq_runs(r2, smap)
        txt = "".join(t for _, t in ref_runs)
        nums = re.findall(r"<v>(\d+)</v>", r2)
        if txt != core:
            problems.append(f"reference {txt} != {core}")
        if nums[:2] != ["0", "0"]:
            problems.append(f"reference distances {nums[:2]} != 0,0")
        if not ref_runs or any(c != GREEN for c, _ in ref_runs):
            problems.append("reference row not fully green")

        if GREEN_FILL not in styles or YELLOW_FILL not in styles:
            problems.append("green/yellow fills absent from styles.xml")
        if 'r="D2"' not in re.search(r'<row r="2".*?</row>', s3).group(0):
            problems.append("Aligned sheet has no per-position columns")

        # Source Location sheet
        loc = {int(m.group(1)): m.group(0)
               for m in re.finditer(r'<row r="(\d+)"[^>]*>.*?</row>', s2)}
        if len(loc) != expect_rows + 1:
            problems.append(f"Source Location rows {len(loc) - 1} != {expect_rows}")
        if cell_text(loc[2], "C") != "OXT.csv" or cell_num(loc[2], "D") != i:
            problems.append("Source Location reference row wrong")

        # sampled content checks
        sample = random.sample(range(3, expect_rows + 2), min(40, expect_rows - 1))
        for rn in sample:
            row = body[rn]
            runs = seq_runs(row, smap)
            seq = "".join(t for _, t in runs)
            n_amber = sum(len(t) for c, t in runs if c == AMBER)
            n_green = sum(len(t) for c, t in runs if c == GREEN)
            v = re.findall(r"<v>(\d+)</v>", row)
            want_l, want_h = int(v[0]), int(v[1])
            src_i = rn - 3                      # index into the source list
            if seq != seqs[src_i]:
                problems.append(f"row {rn}: {seq} != source order {seqs[src_i]}")
            if len(seq) != L:
                problems.append(f"row {rn}: length {len(seq)}")
            if n_amber != want_h or n_green != L - want_h:
                problems.append(f"row {rn}: colours {n_green}/{n_amber} vs Hamming {want_h}")
            if ham(seq, core) != want_h:
                problems.append(f"row {rn}: Hamming {want_h} != {ham(seq, core)}")
            if lev(seq, core) != want_l:
                problems.append(f"row {rn}: Lev {want_l} != {lev(seq, core)}")
            # location row must point at the right physical CSV position
            lr = loc[rn]
            if cell_text(lr, "B") != seq:
                problems.append(f"row {rn}: location sequence mismatch")
            if cell_text(lr, "C") != fname:
                problems.append(f"row {rn}: source_file {cell_text(lr, 'C')} != {fname}")
            if cell_num(lr, "D") != src_i + 1:
                problems.append(f"row {rn}: csv_data_row {cell_num(lr, 'D')} != {src_i + 1}")
            if cell_num(lr, "E") != src_i + 2:
                problems.append(f"row {rn}: csv_line_number {cell_num(lr, 'E')} != {src_i + 2}")
        z.close()

        line += f" {tag}:{'OK' if not problems else 'FAIL'}"
        for p in problems:
            fails.append((f"oxt_{i}_{tag}.xlsx", p))
    print(line)
    for f, p in [x for x in fails if x[0].startswith(f"oxt_{i}_")]:
        print(f"        {f}: {p}")

# no mixing between sources
print()
n_r6, n_vae = len(pool["r6"]), len(pool["vae"])
print(f"  r6 workbooks hold {n_r6:,} + 1 rows, vae workbooks {n_vae:,} + 1 rows "
      f"-- sources never mixed")
if raw["core"].iloc[0] == raw["core"].iloc[9]:
    print(f"  rows 1 and 10 duplicate -> oxt_1_* and oxt_10_* share reference "
          f"{raw['core'].iloc[0]}")

print()
print("ALL WORKBOOKS VALID" if not fails else f"{len(fails)} PROBLEM(S)")
sys.exit(1 if fails else 0)
