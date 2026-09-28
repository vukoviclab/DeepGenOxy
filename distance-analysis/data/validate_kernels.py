#!/usr/bin/env python
"""
Correctness check for the vectorised distance kernels used in
distance_analysis.py, against independent textbook implementations.

Imports distance_analysis (which re-runs the analysis and regenerates the
same outputs -- also a determinism check) and exercises its encode /
levenshtein_matrix / hamming_matrix on random and hand-picked inputs.
"""

import contextlib
import io
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import distance_analysis as da


def ref_lev(a, b):
    """Plain Wagner-Fischer, one row at a time."""
    n = len(b)
    d = list(range(n + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, n + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[n]


def ref_ham(a, b):
    return sum(x != y for x, y in zip(a, b))


fails = []

# --- random cross-check -----------------------------------------------------
random.seed(0)
A = ["".join(random.choice("ACGT") for _ in range(18)) for _ in range(300)]
B = ["".join(random.choice("ACGT") for _ in range(18)) for _ in range(40)]

L = da.levenshtein_matrix(da.encode(A), da.encode(B))
H = da.hamming_matrix(da.encode(A), da.encode(B))
EL = np.array([[ref_lev(a, b) for b in B] for a in A])
EH = np.array([[ref_ham(a, b) for b in B] for a in A])

for name, got, want in [("Levenshtein", L, EL), ("Hamming", H, EH)]:
    ok = np.array_equal(got, want)
    print(f"{name:<12} vs brute force over {L.size:,} pairs : {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(name)

ok = bool((L <= H).all())
print(f"{'invariant':<12} Levenshtein <= Hamming            : {'PASS' if ok else 'FAIL'}")
if not ok:
    fails.append("invariant")

# --- hand-picked edge cases -------------------------------------------------
CASES = [
    ("AAAAAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAAAAA", 0, 0, "identical"),
    ("AAAAAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAAAAT", 1, 1, "single substitution"),
    ("ACGTACGTACGTACGTAC", "CGTACGTACGTACGTACG", 2, 18, "1-nt frameshift"),
    ("AAAAAAAAAAAAAAAAAA", "TTTTTTTTTTTTTTTTTT", 18, 18, "completely different"),
    # OXT_05 core vs its closest VAE sequence: differ only at position 3 (C->G)
    ("GGCGGACAGACTCTAATG", "GGGGGACAGACTCTAATG", 1, 1, "real OXT_05 vs VAE hit"),
    # OXT_14 core vs its closest VAE sequence: differ only at position 3 (G->A)
    ("TTGAGCGTAACGACAGTG", "TTAAGCGTAACGACAGTG", 1, 1, "real OXT_14 vs VAE hit"),
]
print()
for a, b, want_l, want_h, note in CASES:
    gl = int(da.levenshtein_matrix(da.encode([a]), da.encode([b]))[0, 0])
    gh = int(da.hamming_matrix(da.encode([a]), da.encode([b]))[0, 0])
    ok = (gl, gh) == (want_l, want_h)
    print(f"  {note:<24} lev={gl:>2} (want {want_l:>2})  ham={gh:>2} "
          f"(want {want_h:>2})  {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(note)

# --- reference-set integrity ------------------------------------------------
print()
print(f"  unique OXT references loaded            : {len(da.REF_IDS)} (expect 29)")
print(f"  reference core length                   : "
      f"{set(len(c) for c in da.REF_CORES)} (expect {{18}})")
if len(da.REF_IDS) != 29:
    fails.append("reference count")

print()
print("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
