#!/usr/bin/env python
"""
Python port of the dataviz skill's validate_palette.js (no node on this box).

Same math and same thresholds: OKLab/OKLCH conversions, Machado-Oliveira-
Fernandes (2009) severity-1.0 CVD simulation, OKLab dE x100, WCAG contrast.

  usage: python validate_palette.py "#2a78d6,#eb6834" [--mode light|dark]
"""

import math
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}   # OKLCH L
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b))
            for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    ok = True
    print(f"mode={mode}  surface={surface}  palette={palette}\n")

    print("check 2/3/5 — lightness band, chroma floor, contrast vs surface")
    for h in palette:
        L, C = oklch(h)
        cr = contrast(h, surface)
        f_band = "PASS" if lo <= L <= hi else "FAIL"
        f_chr = "PASS" if C >= CHROMA_FLOOR else "FAIL"
        f_con = "PASS" if cr >= CONTRAST_MIN else "WARN"
        ok &= f_band == "PASS" and f_chr == "PASS"
        print(f"  {h}  L={L:.3f} [{f_band}]   C={C:.3f} [{f_chr}]   "
              f"contrast={cr:.2f}:1 [{f_con}]")

    idx = ([(i, i + 1) for i in range(len(palette) - 1)] if pairs == "adjacent"
           else [(i, j) for i in range(len(palette))
                 for j in range(i + 1, len(palette))])
    print(f"\ncheck 4 — CVD separation ({pairs} pairs, target dE>={CVD_TARGET}, "
          f"floor {CVD_FLOOR})")
    for i, j in idx:
        a, b = palette[i], palette[j]
        p, d, t = (delta_e(a, b, "protan"), delta_e(a, b, "deutan"),
                   delta_e(a, b, "tritan"))
        worst = min(p, d)
        flag = ("PASS" if worst >= CVD_TARGET
                else "WARN" if worst >= CVD_FLOOR else "FAIL")
        ok &= flag != "FAIL"
        print(f"  {a} vs {b}  protan={p:.1f} deutan={d:.1f} tritan={t:.1f}  "
              f"min={worst:.1f} [{flag}]")

    print(f"\ncheck 4b — normal-vision floor (worst pair, hard gate {NORMAL_FLOOR})")
    worst_n = min(delta_e(palette[i], palette[j]) for i, j in idx) if idx else 999
    f = "PASS" if worst_n >= NORMAL_FLOOR else "FAIL"
    ok &= f == "PASS"
    print(f"  worst dE={worst_n:.1f} [{f}]")

    print("\n" + ("ALL CHECKS PASS" if ok else "PALETTE HAS FAILURES"))
    return ok


if __name__ == "__main__":
    pal = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    mode = "light"
    surf = None
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
    if "--surface" in sys.argv:
        surf = sys.argv[sys.argv.index("--surface") + 1]
    sys.exit(0 if validate(pal, mode, surf) else 1)
