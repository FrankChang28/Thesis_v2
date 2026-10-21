#!/usr/bin/env python3
"""Run only Stage 1 analyses. Print commands unless --execute is supplied."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from subject_nirs.common.cohort import load_cohort
from subject_nirs.stage1.analysis_support import analysis_environment, validate_checkpoints, validate_features
from subject_nirs.stage1.evaluation import discover_checkpoints
from run_subject_cohort import run_logged


def commands(run, phase, old_root):
    python = sys.executable
    notebooks = run / "notebooks"
    execute_nb = lambda name: [python, str(ROOT / "scripts/run_stage1_analysis.py"),
                               "--run", str(run), "--notebook", str(notebooks / name)]
    if phase == "full":
        return [execute_nb(name) for name in ["00_dtof_subject_variation.ipynb", "01_stage1_retrieval.ipynb",
            "02_stage1_op_robustness_cka.ipynb", "03_stage1_structure_probe.ipynb",
            "09_stage1_cohort_comparison.ipynb"]]
    if phase == "comparison":
        return [[python, str(ROOT / "scripts/compare_stage1_cohorts.py"), "--run", str(run), "--old-root", str(old_root)]]
    tables = ROOT / "results" / run.name / "tables/stage1"
    metadata = run / "metadata_aligned.csv"
    probe = [python, "-m", "subject_nirs.stage1.probe", "--feature_key", "latent_raw",
             "--metadata_csv", str(metadata), "--plot_format", "png", "--thickness_unit", "mm"]
    return [execute_nb("00_dtof_subject_variation.ipynb"),
        probe + ["--feature_glob", str(ROOT / "artifacts" / run.name / "stage1/*/relat_cons_32/subject_features_all_op.npz"),
                 "--cohort_file", str(run / "cohort.json"), "--output_dir", str(tables / "structure_probe_32")],
        probe + ["--feature_glob", str(old_root / "artifacts/stage1/*/relat_cons_32/subject_features_all_op.npz"),
                 "--output_dir", str(tables / "structure_probe_old_aligned_154")],
        *commands(run, "comparison", old_root)]


def execute_notebook(path):
    # These analysis notebooks contain plain Python cells, without shell/magic syntax.
    # Run their source directly so the CLI does not require nbconvert/nbclient.
    scope = {"__name__": "__main__"}
    for index, cell in enumerate(json.loads(path.read_text())["cells"]):
        if cell["cell_type"] == "code":
            print(f"Notebook {path.name}, cell {index}", flush=True)
            exec(compile("".join(cell["source"]), f"{path}#cell-{index}", "exec"), scope)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--phase", choices=["quick", "full", "comparison"], default="quick")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--old-root", type=Path, default=ROOT.parent / "final_refactored")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--notebook", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / "runs"):
        p.error("Use a prepared run in this project's runs directory")
    if args.notebook:
        notebook = args.notebook.resolve()
        if notebook.parent != run / "notebooks":
            p.error("Notebook must belong to the selected run")
        execute_notebook(notebook)
        return
    cohort = load_cohort(run / "cohort.json", require_confirmed=True, expected_fingerprint=run.name)
    checkpoints = discover_checkpoints(ROOT / "artifacts" / run.name / "stage1", "relat_cons_32", None)
    validate_checkpoints(checkpoints, run / "cohort.json")
    features = [path.parent / "subject_features_all_op.npz" for _, path in checkpoints]
    validate_features(features, run / "cohort.json")
    if not (run / "metadata_aligned.csv").is_file():
        raise FileNotFoundError("Aligned run metadata is required")
    environment = analysis_environment(ROOT)
    environment["STAGE1_ANALYSIS_DEVICE"] = args.device
    queue = commands(run, args.phase, args.old_root.resolve())
    print(f"Validated {len(cohort.subject_ids)} complete folds; phase={args.phase}; execute={args.execute}")
    session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for index, command in enumerate(queue, 1):
        print(shlex.join(command), flush=True)
        if args.execute:
            logs = run / "logs"
            logs.mkdir(exist_ok=True)
            run_logged(command, environment, logs / f"{session}_stage1_analysis_{args.phase}_{index:02d}.log")


if __name__ == "__main__":
    main()
