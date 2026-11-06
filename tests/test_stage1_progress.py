"""CPU checks for the remaining-time line printed by the Stage 1 cohort runner."""
from __future__ import annotations

from datetime import datetime
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_subject_cohort", ROOT / "scripts/run_subject_cohort.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_remaining_time_uses_mean_fold_duration():
    line = runner.progress_line(2, 6, 1200.0, datetime(2026, 11, 5, 22, 30))
    assert line == ("[Progress] 2/6 folds | elapsed 0h20m | 10.0 min/fold | "
                    "remaining 0h40m | finish 11-05 23:10")


def test_last_fold_has_no_remaining_time():
    line = runner.progress_line(3, 3, 5400.0, datetime(2026, 11, 5, 22, 30))
    assert "3/3 folds" in line and "remaining 0h00m" in line and "finish 11-05 22:30" in line


def test_export_detection_counts_one_command_per_fold():
    train = ["python", "/repo/scripts/train_stage1.py"]
    export = ["python", "/repo/scripts/export_subject_features.py"]
    assert sum(map(runner.is_export, [train, export, train, export])) == 2
