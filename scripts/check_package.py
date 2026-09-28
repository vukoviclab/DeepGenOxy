#!/usr/bin/env python3
"""Check copied-file hashes and parse Python/notebook files without running them."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    with (ROOT / 'SOURCE_MANIFEST.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    for row in rows:
        relative = row['release_path']
        if relative in seen:
            errors.append(f'Duplicate manifest entry: {relative}')
        seen.add(relative)
        path = ROOT / relative
        if not path.is_file():
            errors.append(f'Missing: {relative}')
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['release_sha256']:
            errors.append(f'Hash mismatch: {relative}')
        if path.stat().st_size != int(row['bytes']):
            errors.append(f'Size mismatch: {relative}')
        if path.suffix == '.ipynb':
            try:
                notebook = json.loads(path.read_text())
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code' and (
                        cell.get('outputs') or cell.get('execution_count') is not None
                    ):
                        errors.append(f'Notebook has execution output: {relative}')
            except (ValueError, KeyError) as exc:
                errors.append(f'Invalid notebook: {relative}: {exc}')
    python_paths = [ROOT / path for path in seen if path.endswith('.py')]
    python_paths += sorted((ROOT / 'scripts').glob('*.py'))
    for path in python_paths:
        try:
            ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
        except SyntaxError as exc:
            errors.append(str(exc))
    for tree in ('oxt', 'vaedeep'):
        paths = list((ROOT / 'vae' / tree / 'NN').glob('*Model/*.h5'))
        if len(paths) != 30:
            errors.append(f'{tree}: expected 30 classifier checkpoints, got {len(paths)}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    size = sum(int(row['bytes']) for row in rows)
    print(f'PASS: {len(rows)} copied/derived assets; {size / 1024**2:.2f} MiB; 60 classifier checkpoints.')
    print('All manifest checksums match; Python files parse; notebooks have no execution outputs.')
    print('Checks cover file integrity, Python syntax, notebook structure, and checkpoint counts.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
