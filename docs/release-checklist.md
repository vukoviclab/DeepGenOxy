# Release checklist

- Retain the repository's existing MIT `LICENSE` and confirm any additional data-release terms with the authors.
- Confirm that the source workbooks and experimental response values are approved for public release. The copied workbooks retain their original Office metadata.
- Use the documented software environment when running the models. The distance calculations can be run independently of TensorFlow.
- Confirm the final paper title, author names, journal reference, and DOI before adding a formal citation file.
- Keep the contents of `DeepGenOxy` at the repository root; do not add the parent project or its manuscript folder. Retain the MIT license.

## Packaging choices

Original CSVs, workbooks, figures, and checkpoints were copied without changing their contents. Python path references were updated for the renamed `vae/` and `distance-analysis/data/` folders; model definitions and analysis calculations are unchanged. Notebook code and Markdown cells are unchanged; outputs, execution counts, and nonessential notebook metadata were cleared. File-level provenance is in `SOURCE_MANIFEST.csv`, whose original source paths and source hashes are preserved.

The round-2 retained-checkpoint CSV is a new extraction from the original training log. It preserves the logged values and their evidence; it does not recompute model scores. Its source and derivation are recorded in the manifest.

Excluded material includes duplicate project trees and renamed copies of the same checkpoints; serotonin datasets and models; old experiments and unnamed notebooks; training console logs; compressed backups; manuscript drafts and review notes; TOC graphics and visualization data; caches; and large pairwise/Excel exports that can be regenerated.

The original training definitions still contain the sixth classifier family. Prediction remains a five-family setup. The sixth family's checkpoints are included only as a historical training record.
