#!/usr/bin/env python3
"""Stage 1 LOSO training and descriptor export for the fixed cohort; dry-run by default."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subject_nirs.common.cohort import check_partition, load_cohort
from subject_nirs.common.config import load_yaml

COHORT_FILE = ROOT / "configs/subject_cohort.json"


def exports_complete(output: Path) -> bool:
    for path in (output / "subject_features_all_op.npz", output / "test_acquisition_embeddings.npz"):
        if not path.is_file():
            return False
        try:
            with zipfile.ZipFile(path) as archive:
                if not archive.namelist() or archive.testzip() is not None:
                    return False
        except (OSError, zipfile.BadZipFile, EOFError):
            return False
    return True


def stage1_complete(output: Path, cohort, sid: int) -> bool:
    path = output / "best_model.pt"
    if not path.exists():
        return False
    import torch
    from subject_nirs.stage1.data import PREPROCESS_VERSION
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["preprocess_stats"].get("dtof_preprocess") != PREPROCESS_VERSION:
        raise ValueError(f"{path} uses older DTOF preprocessing; move that output directory aside first")
    if checkpoint.get("cohort", {}).get("fingerprint") != cohort.fingerprint:
        raise ValueError(f"Existing Stage 1 checkpoint belongs to another cohort: {path}")
    check_partition(cohort, checkpoint["train_subjects"], checkpoint["val_subjects"], checkpoint["test_subjects"])
    if list(map(int, checkpoint["test_subjects"])) != [sid]:
        raise ValueError("Stage 1 checkpoint test ID differs from its directory")
    if not (output / "evaluation_metrics.json").exists() or not (output / "subject_features_eval_grid.npz").exists():
        raise ValueError(f"Incomplete Stage 1 artifacts; retain and inspect this directory before retrying: {output}")
    return True


def export_command(python: str, output: Path) -> list[str]:
    return [python, str(ROOT / "scripts/export_subject_features.py"),
            "--checkpoint", str(output / "best_model.pt"),
            "--data_dir", str(ROOT / "data/raw/stage1"),
            "--save_path", str(output / "subject_features_all_op.npz"),
            "--test_acquisition_save_path", str(output / "test_acquisition_embeddings.npz"), "--amp"]


def commands(config: Path, phase: str, cohort, subject_ids: list[int]) -> list[list[str]]:
    experiment = load_yaml(config)["experiment_name"]
    python = sys.executable
    result = []
    for sid in subject_ids:
        output = ROOT / "artifacts/stage1" / str(sid) / experiment
        train = phase == "stage1" and not stage1_complete(output, cohort, sid)
        if train:
            result.append([python, str(ROOT / "scripts/train_stage1.py"), "--config", str(config),
                           "--test_id", str(sid), "--output_dir", str(output), "--amp", "--skip_tsne"])
        elif phase == "stage1":
            print(f"Stage 1 test_id={sid} {experiment}: complete, retaining checkpoint")
        if not train and exports_complete(output):
            print(f"Stage 1 test_id={sid} {experiment}: exports complete, skipping")
        else:
            result.append(export_command(python, output))
    return result


def child_environment() -> dict[str, str]:
    # Set native-library thread limits before child processes import them.
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MPLBACKEND="Agg",
                       PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    return environment


def run_logged(command: list[str], environment: dict[str, str], log_path: Path) -> None:
    print(f"Log: {log_path}", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        if returncode:
            print(f"Child exited with code {returncode}; output retained in {log_path}", file=sys.stderr)
            raise subprocess.CalledProcessError(returncode, command)


def is_export(command: list[str]) -> bool:
    # Every queued fold ends with exactly one export, so a finished export is a finished fold.
    return Path(command[1]).name == "export_subject_features.py"


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m"


def progress_line(done: int, total: int, elapsed: float, now: datetime) -> str:
    """Estimate the remaining time from the mean duration of the folds finished in this run."""
    average = elapsed / done
    remaining = average * (total - done)
    finish = now + timedelta(seconds=remaining)
    return (f"[Progress] {done}/{total} folds | elapsed {format_duration(elapsed)} | "
            f"{average / 60:.1f} min/fold | remaining {format_duration(remaining)} | "
            f"finish {finish:%m-%d %H:%M}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage1_32.yaml",
                        help="Stage 1 YAML; its experiment_name names the per-fold output directory")
    parser.add_argument("--phase", choices=("stage1", "exports"), default="stage1",
                        help="stage1 trains missing folds and exports; exports only exports")
    parser.add_argument("--test-id", type=int, action="append", help="Original zero-based Stage 1 ID; can repeat")
    parser.add_argument("--execute", action="store_true", help="Explicitly start training/export")
    args = parser.parse_args()
    config = args.config.resolve()
    cohort = load_cohort(COHORT_FILE, require_confirmed=True)
    subject_ids = args.test_id or cohort.subject_ids
    if not set(subject_ids).issubset(cohort.subject_ids):
        parser.error("Requested ID is excluded or unknown")
    queue = commands(config, args.phase, cohort, subject_ids)
    print(f"{config.name}: {len(subject_ids)} folds; {len(queue)} commands; execute={args.execute}")
    environment = child_environment()
    session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    total_folds = sum(map(is_export, queue))
    finished_folds = 0
    started = time.monotonic()
    for index, command in enumerate(queue, start=1):
        print(shlex.join(command), flush=True)
        if args.execute:
            log_dir = ROOT / "results/logs/training"
            log_dir.mkdir(parents=True, exist_ok=True)
            run_logged(command, environment, log_dir / f"{session}_{config.stem}_{args.phase}_{index:04d}.log")
            if is_export(command):
                finished_folds += 1
                print(progress_line(finished_folds, total_folds, time.monotonic() - started, datetime.now()),
                      flush=True)


if __name__ == "__main__":
    main()
