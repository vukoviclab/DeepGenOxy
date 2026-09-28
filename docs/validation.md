# Packaging checks, 2026-09-28

The distance scripts were run in temporary working copies. The source project and the packaged result snapshots were not used as output directories.

| Check | Result |
| --- | --- |
| Source and packaged file checksums | Matched the source manifest |
| Notebook source | All 14 notebooks retained the original code and Markdown cells; execution outputs were cleared |
| Saved classifiers | All 60 HDF5 files opened successfully and all contained datasets were read |
| Checkpoint references | All 25 model references in each of six archived prediction tables resolved to the packaged checkpoint names |
| Distance functions | Independent checks passed for 12,000 random sequence pairs and the supplied edge cases |
| Regenerated CSVs | All 28 packaged result CSVs checked were byte-identical to their regenerated counterparts |
| New analysis wrapper | Completed both rounds; refused an already existing output directory |

The analysis test environment used NumPy 2.4.1, pandas 2.3.3, Matplotlib 3.10.8, and SciPy 1.17.0.

The checks covered file integrity, notebook structure, distance calculations, and regenerated CSV outputs. The HDF5 checks verified that the saved files and their datasets could be read.

To repeat the package and analysis checks:

```bash
python scripts/check_package.py
python scripts/reproduce_analysis.py --output runs/validation
```

Choose a new output directory for each run. The package checker compares the assets listed in `SOURCE_MANIFEST.csv` against their recorded checksums.
