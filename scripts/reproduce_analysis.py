#!/usr/bin/env python3
"""Run the archived distance scripts in a new, isolated output directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROUND2_INPUTS = (
    'round2_nearest_hamming.py',
    'table-S3-JAQUESTA-NEW.xlsx',
    'table-S4-validation2-JAQUESTA-NEW.xlsx',
    'vae/oxt/data/oxt.csv',
    'vae/oxt/data/result_r6_80_SELEC_withC.csv',
    'vae/oxt/data/result_vae_send_withC.csv',
    'vae/vaedeep/data/oxt_1.0_2.csv',
    'vae/vaedeep/result/filter_check_oxt_0.csv',
    'vae/vaedeep/result/filter_check_oxt_1.csv',
)
ROUND1_INPUTS = (
    'distance-analysis/OXT.csv',
    'distance-analysis/result_nn_r6.csv',
    'distance-analysis/result_nn_vae_ordered_base_on_arch.csv',
    'distance-analysis/data/distance_analysis.py',
    'distance-analysis/data/validate_kernels.py',
    'distance-analysis/data/make_paper_csv.py',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New working directory; existing paths are refused.')
    parser.add_argument('--round2-only', action='store_true',
                        help='Run only the standard-library Hamming calculation.')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f'Output already exists; choose a new directory: {output}')
    if output == ROOT or output in ROOT.parents:
        parser.error('Output cannot be the repository or one of its parents.')
    # Only runs/ is allowed inside the release tree. External scratch paths work.
    if ROOT in output.parents and ROOT / 'runs' not in output.parents:
        parser.error('Inside the repository, choose a directory below runs/.')
    inputs = ROUND2_INPUTS + (() if args.round2_only else ROUND1_INPUTS)
    for rel in inputs:
        if not (ROOT / rel).is_file():
            parser.error(f'Required input is missing: {rel}')
    output.mkdir(parents=True, exist_ok=False)
    for rel in inputs:
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    commands = ['round2_nearest_hamming.py']
    if not args.round2_only:
        commands += ['distance-analysis/data/validate_kernels.py',
                     'distance-analysis/data/make_paper_csv.py']
    # md_to_pdf is intentionally not copied: no browser/PDF renderer is needed.
    for rel in commands:
        print(f'Running {rel}', flush=True)
        subprocess.run([sys.executable, str(output / rel)], cwd=output, check=True)
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
