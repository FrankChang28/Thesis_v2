#!/usr/bin/env python3
"""Print or execute a prepared cohort's experiment sequence; dry-run by default."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
import shlex
import subprocess
import sys
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subject_nirs.common.cohort import load_cohort
from subject_nirs.common.config import load_yaml

def exports_complete(output: Path, acquisitions: bool) -> bool:
    paths = [output / "subject_features_all_op.npz"]
    if acquisitions:
        paths.append(output / "test_acquisition_embeddings.npz")

    for path in paths:
        if not path.is_file():
            return False
        try:
            with zipfile.ZipFile(path) as archive:
                if not archive.namelist() or archive.testzip() is not None:
                    return False
        except (OSError, zipfile.BadZipFile, EOFError):
            return False
    return True

def commands(run: Path, phase: str, subject_ids: list[int], dims: list[int]) -> list[list[str]]:
    artifacts = ROOT / "artifacts" if run == ROOT else ROOT / "artifacts" / run.name
    configs = run / "configs"
    python = sys.executable
    result = []
    if phase in ("stage1", "all"):
        for dim in dims:
            for sid in subject_ids:
                output = artifacts / "stage1" / str(sid) / f"relat_cons_{dim}"
                if stage1_complete(output, run, sid):
                    print(f"Stage 1 test_id={sid} dim={dim}: complete, retaining checkpoint")
                else:
                    result.append([python, str(ROOT / "scripts/train_stage1.py"), "--config",
                                   str(configs / f"stage1_{dim}.yaml"), "--test_id", str(sid),
                                   "--output_dir", str(output), "--amp", "--skip_tsne"])
                if exports_complete(output, acquisitions=(dim == 32)):
                    print(f"Stage 1 test_id={sid} dim={dim}: exports complete, skipping")
                else:
                    result.append(export_command(python, output, acquisitions=(dim == 32)))
                
    if phase == "exports":
        for sid in subject_ids:
            output = artifacts / "stage1" / str(sid) / "relat_cons_32"
            if exports_complete(output, acquisitions=True):
                print(f"Stage 1 test_id={sid}: exports complete, skipping")
            else:
                result.append(export_command(python, output, acquisitions=True))
    if phase in ("stage2", "all"):
        paths = list((configs / "stage2").rglob("*.yaml"))
        def priority(path):
            values = load_yaml(path)
            return (0 if values.get("experiment_mode") == "baseline" else
                    2 if values.get("execution_mode") == "descriptor_control" else 1, str(path))
        for path in sorted(paths, key=priority):
            result.append([python, str(ROOT / "scripts/train_stage2.py"), "--config", str(path)])
    if phase in ("analysis", "all"):
        # Notebooks are frozen, outputs cleared, and paths retargeted at preparation.
        for notebook in sorted((run / "notebooks").glob("*.ipynb")):
            result.append([python, "-m", "jupyter", "nbconvert", "--execute", "--to", "notebook",
                           "--inplace", "--ExecutePreprocessor.timeout=-1", str(notebook)])
    return result


def stage1_complete(output: Path, run: Path, sid: int) -> bool:
    path = output / "best_model.pt"
    if not path.exists():
        return False
    import torch
    from subject_nirs.common.cohort import check_partition
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cohort = load_cohort(ROOT / "configs/subject_cohort.json" if run == ROOT else run / "cohort.json",
                         expected_fingerprint=None if run == ROOT else run.name)
    if checkpoint.get("cohort", {}).get("fingerprint") != cohort.fingerprint:
        raise ValueError(f"Existing Stage 1 checkpoint belongs to another cohort: {path}")
    check_partition(cohort, checkpoint["train_subjects"], checkpoint["val_subjects"], checkpoint["test_subjects"])
    if list(map(int, checkpoint["test_subjects"])) != [sid]:
        raise ValueError("Stage 1 checkpoint test ID differs from its directory")
    if not (output / "evaluation_metrics.json").exists() or not (output / "subject_features_eval_grid.npz").exists():
        raise ValueError(f"Incomplete Stage 1 artifacts; retain and inspect this directory before retrying: {output}")
    return True


def export_command(python: str, output: Path, acquisitions: bool) -> list[str]:
    command = [python, str(ROOT / "scripts/export_subject_features.py"),
               "--checkpoint", str(output / "best_model.pt"),
               "--data_dir", str(ROOT / "data/raw/stage1"),
               "--save_path", str(output / "subject_features_all_op.npz"), "--amp"]
    if acquisitions:
        command.extend(["--test_acquisition_save_path", str(output / "test_acquisition_embeddings.npz")])
    return command



def child_environment() -> dict[str, str]:
    # Match the original shell launchers before native libraries are imported.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, help="Prepared runs/<fingerprint> directory")
    parser.add_argument("--phase", choices=("stage1", "exports", "stage2", "analysis", "all"), default="all")
    parser.add_argument("--stage1-dim", choices=("32", "16", "both"), default="32")
    parser.add_argument("--test-id", type=int, action="append", help="Original zero-based Stage 1 ID; can repeat")
    parser.add_argument("--execute", action="store_true", help="Explicitly start training/analysis")
    args = parser.parse_args()
    if args.run:
        run = args.run.resolve()
    else:
        run = ROOT
    cohort = load_cohort(ROOT / "configs/subject_cohort.json" if run == ROOT else run / "cohort.json",
                         require_confirmed=True, expected_fingerprint=None if run == ROOT else run.name)
    if run != ROOT and not run.is_relative_to(ROOT / "runs"):
        parser.error("--run must refer to this project's prepared runs directory")
    if args.test_id and args.phase not in ("stage1", "exports"):
        parser.error("--test-id is supported for Stage 1/exports only; Stage 2 subsets use a prepared YAML copy")
    subject_ids = args.test_id or cohort.subject_ids
    if not set(subject_ids).issubset(cohort.subject_ids):
        parser.error("Requested ID is excluded or unknown")
    dims = [32, 16] if args.stage1_dim == "both" else [int(args.stage1_dim)]
    queue = commands(run, args.phase, subject_ids, dims)
    print(f"Cohort {run.name}: {len(cohort.subject_ids)} subjects; {len(queue)} commands; execute={args.execute}")
    environment = child_environment()
    session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for index, command in enumerate(queue, start=1):
        print(shlex.join(command), flush=True)
        if args.execute:
            log_dir = ROOT / "results/logs/training" if run == ROOT else run / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{session}_{args.phase}_{index:04d}.log"
            run_logged(command, environment, log_path)


if __name__ == "__main__":
    main()
