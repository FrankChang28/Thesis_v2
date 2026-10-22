from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from subject_nirs.stage2.loso import (
    completed_test_ids,
    global_fold_pairs,
    retain_completed_epoch_rows,
)


def test_subrange_keeps_global_fold_numbers() -> None:
    all_subjects = list(range(154))
    assert global_fold_pairs(all_subjects, [19, 20, 153]) == [
        (20, 19),
        (21, 20),
        (154, 153),
    ]


def test_resume_keeps_only_complete_correctly_numbered_rows() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "complete.pth"
        model.touch()
        results = root / "loso_results.csv"
        fieldnames = ["fold", "test_id", "test_loss", "model_path"]
        rows = [
            {"fold": 1, "test_id": 0, "test_loss": 0.1, "model_path": model},
            # A selected subrange must retain the global fold numbering.
            {"fold": 1, "test_id": 19, "test_loss": 0.2, "model_path": model},
            # Missing model means the fold was not safely completed.
            {"fold": 3, "test_id": 2, "test_loss": 0.3, "model_path": root / "missing.pth"},
        ]
        with results.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        fold_by_test_id = {subject_id: subject_id + 1 for subject_id in range(154)}
        assert completed_test_ids(str(results), fold_by_test_id) == {0}
        with results.open("r", newline="", encoding="utf-8") as handle:
            normalized = list(csv.DictReader(handle))
        assert len(normalized) == 1
        assert int(normalized[0]["test_id"]) == 0
        assert results.with_name("loso_results.pre_resume_backup.csv").exists()


def test_resume_accepts_checkpoint_relocated_with_experiment_directory() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        relocated_model = root / "best_model_fold_001_test_id_0.pth"
        relocated_model.touch()
        results = root / "loso_results.csv"
        historical_model = (
            Path("artifacts/stage2/legacy_long_experiment_name")
            / relocated_model.name
        )
        with results.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["fold", "test_id", "test_loss", "model_path"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "fold": 1,
                    "test_id": 0,
                    "test_loss": 0.1,
                    "model_path": historical_model,
                }
            )

        assert completed_test_ids(str(results), {0: 1}) == {0}


def test_epoch_resume_discards_nul_tail_and_interrupted_fold() -> None:
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "epochs.csv"
        original = b"fold,test_id,epoch,train_loss\n1,0,1,0.5\n2,1,1,0.7\n" + b"\x00" * 626
        path.write_bytes(original)
        retain_completed_epoch_rows(str(path), {0})
        assert list(csv.DictReader(path.open())) == [
            {"fold": "1", "test_id": "0", "epoch": "1", "train_loss": "0.5"}
        ]
        backups = list(path.parent.glob("epochs.nul_backup_*.csv"))
        assert len(backups) == 1 and backups[0].read_bytes() == original
        clean = path.read_bytes()
        retain_completed_epoch_rows(str(path), {0})
        assert path.read_bytes() == clean
        assert len(list(path.parent.glob("epochs.nul_backup_*.csv"))) == 1


def test_epoch_resume_drops_corrupt_record_without_joining_values() -> None:
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "epochs.csv"
        path.write_bytes(b"fold,test_id,epoch,train_loss\n1,0,1,0.5\n1,0,2,1\x002\n1,0,3,0.4\n")
        retain_completed_epoch_rows(str(path), {0})
        rows = list(csv.DictReader(path.open()))
        assert [row["epoch"] for row in rows] == ["1", "3"]
        assert [row["train_loss"] for row in rows] == ["0.5", "0.4"]


def test_epoch_resume_cleans_nul_tail_when_all_folds_complete() -> None:
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "epochs.csv"
        clean = b"fold,test_id,epoch\r\n1,0,1\r\n"
        path.write_bytes(clean + b"\x00" * 10)
        retain_completed_epoch_rows(str(path), {0})
        assert path.read_bytes() == clean


def test_epoch_resume_rejects_corrupt_header_without_modification() -> None:
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "epochs.csv"
        original = b"fold,test_\x00id,epoch\n1,0,1\n"
        path.write_bytes(original)
        try:
            retain_completed_epoch_rows(str(path), {0})
        except ValueError as error:
            assert "header contains NUL" in str(error)
        else:
            raise AssertionError("A damaged header must not be silently repaired")
        assert path.read_bytes() == original


def test_epoch_resume_removes_clean_interrupted_fold() -> None:
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "epochs.csv"
        path.write_bytes(b"fold,test_id,epoch\n1,0,1\n2,1,1\n")
        retain_completed_epoch_rows(str(path), {0})
        assert [row["test_id"] for row in csv.DictReader(path.open())] == ["0"]


if __name__ == "__main__":
    test_subrange_keeps_global_fold_numbers()
    test_resume_keeps_only_complete_correctly_numbered_rows()
    test_resume_accepts_checkpoint_relocated_with_experiment_directory()
    print("Stage 2 resume tests: OK")
