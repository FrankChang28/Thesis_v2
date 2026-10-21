from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


def prepare_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("cohort_prepare_alignment", root / "scripts/prepare_subject_cohort.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alignment_uses_names_and_preserves_original(tmp_path):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    source = root / "data/temp_metadata_v2.csv"
    text = "subject,sbj,scalp\nAlpha,0,1.0\nChang,1,2.0\nChang2,2,3.0\nLast,3,4.0\n"
    source.write_text(text)
    (root / "data/subject_names.txt").write_text("Y51_F_Alpha_EEG\nY56_M_Chang2_EEG\nY56_M_Chang_EEG\nn_Y71_F_Last_EEG\n")
    run = root / "runs/test"
    path = prepare_module().align_metadata(root, run)
    rows = list(csv.DictReader(path.open()))
    assert [row["subject"] for row in rows] == ["Alpha", "Chang2", "Chang", "Last"]
    assert [row["sbj"] for row in rows] == ["0", "1", "2", "3"]
    assert rows[1]["scalp"] == "3.0" and rows[2]["scalp"] == "2.0"
    assert source.read_text() == text
    assert prepare_module().align_metadata(root, run) == path


def test_alignment_rejects_ambiguous_names(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/temp_metadata_v2.csv").write_text("subject,sbj\nAlpha,0\nAlpha,1\n")
    (tmp_path / "data/subject_names.txt").write_text("Y51_F_Alpha_EEG\nY52_F_Beta_EEG\n")
    with pytest.raises(ValueError, match="uniquely"):
        prepare_module().align_metadata(tmp_path, tmp_path / "runs/test")
