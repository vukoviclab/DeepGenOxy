#!/usr/bin/env python3
"""Extract exact oxytocin checkpoint save/F1 records from a training log.

This is a read-only log parser. It does not load models or recompute scores.
Source line numbers count LF-delimited physical lines, starting at one.
The output is created exclusively; existing output files are never replaced.
"""

import argparse
import csv
import hashlib
from pathlib import Path
import re


SCORE = re.compile(
    rb"Model (\d+) and the random state is (\d+) F1 Score:\s*([0-9.eE+-]+)"
)
SAVE = re.compile(rb"Model (\d+) saved as ([A-Za-z0-9_]+/[^\s]+\.h5)")
CHECKPOINT = re.compile(r"model_(\d+)_rnd_(\d+)_oxt_1\.0_2\.h5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--source-label", help="Provenance path recorded in the CSV")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    actual = {
        str(p.relative_to(args.checkpoint_dir)): p
        for p in args.checkpoint_dir.glob("*/*.h5")
        if CHECKPOINT.fullmatch(p.name)
    }
    if not actual:
        parser.error("No oxytocin oxt_1.0_2 checkpoint files found")

    found = {}
    last_score = None
    digest = hashlib.sha256()
    with args.log.open("rb") as source:
        for number, raw in enumerate(source, 1):
            digest.update(raw)
            score = SCORE.search(raw)
            if score:
                last_score = (number, score)
            save = SAVE.search(raw)
            if not save:
                continue
            checkpoint = save[2].decode("utf-8")
            if checkpoint not in actual:
                continue
            if checkpoint in found:
                raise ValueError(f"Duplicate matching save record: {checkpoint}")
            if last_score is None or last_score[1][1] != save[1]:
                raise ValueError(f"Missing matching score record: {checkpoint}")
            score_line, score = last_score
            if score_line != number - 1:
                raise ValueError(f"Score is not immediately before save: {checkpoint}")
            filename_numbers = CHECKPOINT.fullmatch(actual[checkpoint].name)
            recorded_model = save[1].decode("ascii")
            recorded_seed = score[2].decode("ascii")
            if (recorded_model, recorded_seed) != filename_numbers.groups():
                raise ValueError(f"Log/filename number mismatch: {checkpoint}")
            architecture = actual[checkpoint].parent.name
            found[checkpoint] = {
                "checkpoint_file": checkpoint,
                "checkpoint_sha256": hashlib.sha256(actual[checkpoint].read_bytes()).hexdigest(),
                "architecture": architecture,
                "used_for_prediction": "false" if architecture == "ConvLSTPAMModel" else "true",
                "dataset": "oxt_1.0_2",
                "filename_model_number": filename_numbers[1],
                "filename_random_state": filename_numbers[2],
                "recorded_model_number": recorded_model,
                "recorded_random_state": recorded_seed,
                "recorded_f1": score[3].decode("ascii"),
                "source_path": args.source_label or str(args.log),
                "source_line": number,
                "score_source_line": score_line,
                "source_sha256": "",
                "score_evidence": score[0].decode("utf-8"),
                "save_evidence": save[0].decode("utf-8"),
            }

    missing = set(actual) - set(found)
    if missing:
        raise ValueError(f"Checkpoints without log records: {sorted(missing)}")
    rows = [found[key] for key in sorted(found)]
    for row in rows:
        row["source_sha256"] = digest.hexdigest()
    with args.output.open("x", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Extracted {len(rows)} exact records to {args.output}")
    print(f"Source SHA-256: {digest.hexdigest()}")


if __name__ == "__main__":
    main()
