#!/usr/bin/env python3
"""
Nearest original / round-1 sequence for each round-2 predicted sequence, by Hamming distance.

Query set     table-S4-validation2-JAQUESTA-NEW.xlsx, column `sequence` (20 x 18 nt)
Reference set vae/vaedeep/data/oxt_1.0_2.csv (48 x 18 nt) = 29 originals + 19 round 1,
              which is also the training set for the round-2 models.

Sequence IDs  originals  OXT_01..OXT_29  from distance-analysis/data/oxt_reference_29.csv
              round 1    S1..S19         from table-S3-JAQUESTA-NEW.xlsx, sheet SI-tableS3-final
                                         (aliases SELEC1..8 / VAE1..11 kept alongside)

Outputs (four requested columns, plus an extended table and the reference key):
    round2_nearest_hamming.csv           predicted_sequence, nearest_id, nearest_sequence, hamming_distance
    round2_nearest_hamming_extended.csv  + ties, sub-pool distances, dff, both architecture values
    reference_set_48.csv                 the 48 references with their assigned IDs

Reads every input read-only; writes only the three CSVs above, into this directory.
Standard library only - no pandas, no openpyxl.
"""

import csv
import os
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def p(*parts):
    return os.path.join(ROOT, *parts)


def read_xlsx(path, sheet_index=0):
    """Return one worksheet as a list of row-lists. Handles shared and inline strings."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si"):
                shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
        names = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))
        sheet = ET.fromstring(z.read(names[sheet_index]))

    rows = []
    for row in sheet.iter(NS + "row"):
        cells, width = {}, 0
        for c in row.findall(NS + "c"):
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            idx = 0
            for ch in col:
                idx = idx * 26 + (ord(ch) - 64)
            idx -= 1
            v, inline = c.find(NS + "v"), c.find(NS + "is")
            if c.get("t") == "inlineStr" and inline is not None:
                val = "".join(t.text or "" for t in inline.iter(NS + "t"))
            elif v is None:
                val = ""
            elif c.get("t") == "s":
                val = shared[int(v.text)]
            else:
                val = v.text
            cells[idx] = val
            width = max(width, idx)
        rows.append([cells.get(i, "") for i in range(width + 1)])
    return rows


def rd(path):
    with open(p(path), newline="") as f:
        return list(csv.DictReader(f))


def core(s):
    """Strip the constant CCCCCC clamps, leaving the 18 nt variable core."""
    return s[6:-6] if len(s) == 30 and s.startswith("CCCCCC") and s.endswith("CCCCCC") else s


def hamming(a, b):
    assert len(a) == len(b), "Hamming distance needs equal lengths"
    return sum(x != y for x, y in zip(a, b))


# --------------------------------------------------------------------------- references
originals = rd("vae/oxt/data/oxt.csv")                       # 29, order defines OXT_01..OXT_29
r1_selex = rd("vae/oxt/data/result_r6_80_SELEC_withC.csv")   # 8  round-1 SELEX picks
r1_vae = rd("vae/oxt/data/result_vae_send_withC.csv")        # 11 round-1 VAE picks
train_rows = rd("vae/vaedeep/data/oxt_1.0_2.csv")            # 48, carries dff + category
train = {r["sequence"]: r for r in train_rows}
train_row_no = {r["sequence"]: i for i, r in enumerate(train_rows, start=2)}  # 1-based, header = 1

# Round-1 IDs come from Table S3, which is the published naming. Its order is checked
# against the two source files below so a reordered Table S3 cannot silently mislabel.
s3 = read_xlsx(p("table-S3-JAQUESTA-NEW.xlsx"), sheet_index=1)  # sheet "SI-tableS3-final"
s3_rows = [r for r in s3[1:] if r and r[0].strip() and r[0].strip() != "nanOx"]
s3_ids = [(r[0].strip(), r[1].strip(), core(r[2].strip())) for r in s3_rows]
file_order = [core(r["sequence"]) for r in r1_selex] + [core(r["sequence"]) for r in r1_vae]
assert [q for _, _, q in s3_ids] == file_order, (
    "Table S3 round-1 order no longer matches result_r6_80_SELEC_withC.csv + result_vae_send_withC.csv")

refs = []  # (id, alias, sequence, set, dff, category, row in oxt_1.0_2.csv)
for i, r in enumerate(originals, start=1):
    s = core(r["sequence"])
    refs.append(("OXT_%02d" % i, "", s, "original", float(r["dff"]), train[s]["category"], train_row_no[s]))
for i, (sid, alias, s) in enumerate(s3_ids):
    kind = "round 1 (SELEX)" if i < len(r1_selex) else "round 1 (VAE)"
    refs.append((sid, alias, s, kind, float(train[s]["dff"]), train[s]["category"], train_row_no[s]))

assert len(refs) == 48, "expected 48 references, got %d" % len(refs)
assert len({r[2] for r in refs}) == 48, "reference sequences are not unique"
assert len({r[0] for r in refs}) == 48, "reference ids are not unique"
assert all(len(r[2]) == 18 for r in refs), "every reference must be 18 nt"
assert set(train) == {r[2] for r in refs}, "references do not reconstruct oxt_1.0_2.csv exactly"

# --------------------------------------------------------------------------- queries
sheet = read_xlsx(p("table-S4-validation2-JAQUESTA-NEW.xlsx"))
header, data = sheet[0], sheet[1:]
c_id, c_seq, c_arch = header.index("SeqID (round 2)"), header.index("sequence"), header.index("VAE architecture")
c_prob = header.index("average probability")
c_dff = header.index("ΔF/F (mean at 1195 nm) new method")

arch_repo = {}
for f in ("filter_check_oxt_0.csv", "filter_check_oxt_1.csv"):
    for r in rd(os.path.join("vae", "vaedeep", "result", f)):
        arch_repo[r["sequence"]] = r["architecture"]

queries = []
for row in data:
    seq = row[c_seq]
    assert len(seq) == 18, "query %s is not 18 nt" % seq
    queries.append((row[c_id], seq, " ".join(row[c_arch].split()), float(row[c_prob]), float(row[c_dff])))

# --------------------------------------------------------------------------- nearest
results = []
for sid, seq, xl_arch, prob, dff in queries:
    scored = [(hamming(seq, rs), rid, rs, rset, rdff, rcat, rrow, ral)
              for rid, ral, rs, rset, rdff, rcat, rrow in refs]
    best_d = min(d for d, *_ in scored)
    tied = [s for s in scored if s[0] == best_d]
    best = tied[0]  # tie-break: first in reference order (originals, then round 1)
    results.append(
        dict(
            seqid=sid, seq=seq, d=best_d,
            nid=best[1], nseq=best[2], nset=best[3], ndff=best[4], ncat=best[5], nrow=best[6],
            nalias=best[7],
            all_ids="; ".join(t[1] for t in tied), all_seqs="; ".join(t[2] for t in tied),
            n_tied=len(tied), tied=";".join(t[1] for t in tied),
            d_orig=min(d for d, _, _, st, *_ in scored if st == "original"),
            d_r1=min(d for d, _, _, st, *_ in scored if st.startswith("round 1")),
            d_mean=sum(d for d, *_ in scored) / len(scored),
            d_max=max(d for d, *_ in scored),
            prob=prob, dff=dff, xl_arch=xl_arch, repo_arch=arch_repo[seq],
        )
    )

# --------------------------------------------------------------------------- write
with open(p("round2_nearest_hamming.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["predicted_sequence", "nearest_sequence_id", "original_or_round1_sequence", "hamming_distance"])
    for x in results:
        w.writerow([x["seq"], x["all_ids"], x["all_seqs"], x["d"]])

with open(p("round2_nearest_hamming_extended.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "seqid_round2", "predicted_sequence", "nearest_id", "nearest_sequence", "hamming_distance",
        "nearest_set", "nearest_dff", "nearest_category", "row_in_oxt_1.0_2.csv",
        "nearest_alias", "n_tied_nearest", "tied_ids", "min_dist_to_original29", "min_dist_to_round1_19",
        "mean_dist_all48", "max_dist_all48", "round2_avg_probability", "round2_dff_mean_1195nm",
        "architecture_in_xlsx", "architecture_in_repo",
    ])
    for x in results:
        w.writerow([
            x["seqid"], x["seq"], x["nid"], x["nseq"], x["d"], x["nset"], x["ndff"], x["ncat"], x["nrow"],
            x["nalias"], x["n_tied"], x["tied"], x["d_orig"], x["d_r1"], round(x["d_mean"], 3), x["d_max"],
            x["prob"], round(x["dff"], 6), x["xl_arch"], x["repo_arch"],
        ])

with open(p("reference_set_48.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "alias", "sequence_18nt", "set", "dff", "category", "row_in_oxt_1.0_2.csv"])
    for rid, alias, s, rset, dff, cat, rrow in refs:
        w.writerow([rid, alias, s, rset, dff, cat, rrow])

# --------------------------------------------------------------------------- report
print("%-6s %-19s %-13s %-19s %s" % ("ID", "predicted sequence", "nearest ID", "nearest sequence", "H"))
print("-" * 70)
for x in results:
    print("%-6s %-19s %-13s %-19s %d%s" % (
        x["seqid"], x["seq"], x["nid"], x["nseq"], x["d"], "  (%d tied)" % x["n_tied"] if x["n_tied"] > 1 else ""))
print("-" * 70)
print("min %d  max %d  mean %.2f   |   nearest is an original: %d, round 1: %d" % (
    min(x["d"] for x in results), max(x["d"] for x in results),
    sum(x["d"] for x in results) / len(results),
    sum(1 for x in results if x["nset"] == "original"),
    sum(1 for x in results if x["nset"] != "original")))
print("wrote round2_nearest_hamming.csv, round2_nearest_hamming_extended.csv, reference_set_48.csv")
