# DeepGenOxy

Code and data for *Iterative discovery of DNA-carbon nanotube oxytocin sensors using machine learning and generative models*.

The project uses variational autoencoders (VAEs) to generate DNA sequences and neural-network classifiers to rank their predicted oxytocin responses. The files cover two rounds of model development, saved predictions, candidate selection, and sequence-distance analysis. DNA inputs use an 18-nucleotide variable region; experimental constructs include six cytosines at each end.

## Files

| Location | Contents |
| --- | --- |
| `vae/oxt/` | Round-1 data, VAE and classifier code, classifier checkpoints, predictions, and analysis notebooks |
| `vae/vaedeep/` | Round-2 oxytocin data, model code, classifier checkpoints, and predictions |
| `distance-analysis/` | Comparison of VAE and SELEX candidate pools with the original sequences |
| `round2_nearest_hamming.py` | Nearest original or round-1 sequence for each of the 20 round-2 candidates |
| `table-S3-JAQUESTA-NEW.xlsx`, `table-S4-validation2-JAQUESTA-NEW.xlsx` | Source tables used by the round-2 distance calculation |
| `scripts/` | Commands that run analyses in a new working directory |
| `SOURCE_MANIFEST.csv` | Source paths and SHA-256 checksums |

The three VAE architectures are `dense`, `lstm`, and `conv`. Predictions use five classifier architectures: `ConvModel`, `DenseModel`, `LSTMModel`, `ConvLSTMModel`, and `LSTMAttentionModel`. Each contributes five fitted models, giving 25 predictions per sequence in each round. A sixth classifier, `ConvLSTPAMModel`, was trained but was not used for prediction. Its saved weights are retained as part of the training record; both prediction scripts exclude it.

The `architecture` column in a generated-sequence table identifies the VAE generator, not the classifier ensemble.

`vae/vaedeep/NN/retained_checkpoint_records.csv` contains the logged F1 scores and split seeds for all 30 retained round-2 classifiers, with source-line references. These are training-selection records, not newly computed performance estimates.

## Run the distance analysis

From the repository root, use a separate Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_analysis.py --output runs/distance-check
```

`python scripts/check_package.py` checks the copied-file hashes and basic file structure without running the models or analyses.

The command requires a new output directory. It copies the needed inputs there before running the original analysis scripts, so the supplied result tables stay unchanged. It checks the distance functions and writes the round-1 CSVs and plots under `runs/distance-check/distance-analysis/data/`. The three round-2 CSVs are written directly under `runs/distance-check/`.

To run only the round-2 Hamming calculation, no third-party packages are needed:

```bash
python scripts/reproduce_analysis.py --round2-only --output runs/round2-check
```

The round-1 comparison uses 1,378 VAE sequences and 19,983 SELEC round-6 sequences (labeled "SELEX" in the historical analysis exports). `distance-analysis/OXT.csv` has 30 rows with one duplicate, giving 29 unique references. The scripts remove the six-C flanks and compare the 18-base cores. Results are provided both with all 29 references and with the two poly-C controls excluded.

The round-2 calculation compares 20 candidates against 48 references: 29 originals and 19 round-1 sequences. `round2_nearest_hamming.csv` lists every tied nearest reference. `reference_set_48.csv` supplies the sequence IDs. The extended output retains historical workbook annotations in `architecture_in_xlsx`; use `architecture_in_repo`, taken from the saved prediction tables, for generator provenance.

Hamming distance counts differing positions. Levenshtein distance also allows insertions and deletions. The reported normalized Levenshtein similarity is a distance-derived score, not alignment-derived sequence identity or a measure of sensor response.

## Models and notebooks

The recorded model environment used Python 3.10.12, TensorFlow 2.12.0, and Keras 2.12.0. `requirements-ml.txt` contains the main recorded package versions. `vae/freezevae.txt` records the broader environment, including machine-specific GPU packages. Model diagrams also require the Graphviz executable.

Use a separate Python 3.10 environment for the model code:

```bash
python3.10 -m venv .venv-ml
source .venv-ml/bin/activate
python -m pip install -r requirements-ml.txt
mkdir -p runs
cd vae/oxt/NN
python predicting.py ../data/vae_prediction.csv ../../../runs/round1-predictions.csv
```

Choose an unused output filename: the predictor overwrites the destination if it exists. This command uses the saved round-1 classifiers; it does not train a VAE.

The notebooks contain data-preparation, candidate-selection, and plotting code. Their source cells are retained and saved outputs are cleared. Run exploratory notebook cells in a separate working copy to keep the supplied result tables unchanged.

## Repository contents

Model files are in `vae/`; distance-analysis scripts and results are in `distance-analysis/data/`. Manuscript drafts, TOC assets, duplicate project trees, serotonin experiments, caches, and large regenerable analysis exports are excluded. The distance checks and 28 regenerated CSV comparisons passed; see [docs/validation.md](docs/validation.md). Publication and file-organization items are listed in [docs/release-checklist.md](docs/release-checklist.md).

The repository's existing [MIT license](LICENSE) is retained. Publication details and a DOI should be added when confirmed.

Authors: Payam Kelich, Jaquesta Adams, Xavier Velez, Markita P. Landry, and Lela Vuković.
