from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from subject_nirs.common.cohort import Cohort, load_cohort, check_partition
from subject_nirs.common.config import apply_dataclass_config, load_yaml
from subject_nirs.stage1.config import build_parser
from subject_nirs.stage1.data import _make_split, load_data_bundle
from subject_nirs.stage1.evaluation import query_metrics
from subject_nirs.stage2.config import CFG, Config
from subject_nirs.stage2.loso import global_fold_pairs, split_subjects
from subject_nirs.stage2.latent import _apply_control, load_dtof_bank

ROOT = Path(__file__).resolve().parents[1]


def cohort_file(tmp_path, removed=(2, 7, 13), raw=24, confirmed=True):
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps({"raw_num_subjects": raw, "id_base": 0,
        "remove_subject_ids": list(removed), "confirmed": confirmed}), encoding="utf-8")
    return path


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_fold_excludes_removed_and_stage_splits_match(tmp_path, monkeypatch):
    path = cohort_file(tmp_path)
    cohort = load_cohort(path, 24)
    monkeypatch.setattr(CFG, "num_subjects", 24)
    monkeypatch.setattr(CFG, "cohort_file", str(path))
    for sid in cohort.subject_ids:
        stage1 = _make_split(24, sid, 100, cohort)
        train, val = split_subjects(cohort.subject_ids, sid)
        assert train == stage1.train_subjects.tolist()
        assert val == stage1.val_subjects.tolist()
        assert len(train) == len(val) == 10
        assert set(train + val + [sid]) == set(cohort.subject_ids)
        assert not set(train + val + [sid]).intersection(cohort.excluded_subject_ids)
    assert global_fold_pairs(cohort.subject_ids, [1, 3, 23]) == [(2, 1), (4, 3), (24, 23)]
    with pytest.raises(ValueError, match="Excluded"):
        cohort.split(2)


@pytest.mark.parametrize("removed", [[2, 2], [-1], [24], [True], [1.5]])
def test_invalid_removal_list_fails(tmp_path, removed):
    with pytest.raises(ValueError):
        load_cohort(cohort_file(tmp_path, removed=removed), 24)


def test_one_based_list_has_same_identity(tmp_path):
    path = cohort_file(tmp_path)
    original = load_cohort(path, 24)
    values = json.loads(path.read_text())
    values.update(id_base=1, remove_subject_ids=[3, 8, 14])
    path.write_text(json.dumps(values))
    assert load_cohort(path, 24).fingerprint == original.fingerprint


def test_pending_and_changed_cohort_are_blocked(tmp_path):
    path = cohort_file(tmp_path, confirmed=False)
    cohort = load_cohort(path, 24)
    with pytest.raises(ValueError, match="unconfirmed"):
        load_cohort(path, 24, require_confirmed=True)
    path = cohort_file(tmp_path, removed=(2, 8, 13))
    with pytest.raises(ValueError, match="changed"):
        load_cohort(path, 24, expected_fingerprint=cohort.fingerprint)


def test_original_mat_row_layout_is_preserved(tmp_path):
    path = cohort_file(tmp_path)
    args = build_parser().parse_args([])
    args.num_subjects = 24
    args.cohort_file = str(path)
    args.data_dir = str(tmp_path)
    args.test_id = 23
    args.samples_per_subject = 4
    args.subjects_per_batch = 4
    args.val_group_rounds = 1
    rng = np.random.default_rng(41)
    dtof = rng.uniform(.1, 1, (24 * 6, 5, 8)).astype(np.float32)
    op = np.tile(np.arange(6, dtype=np.float32)[:, None], (24, 3))
    with h5py.File(tmp_path / "dataset_dtof.mat", "w") as handle:
        handle.create_dataset("dataset_dtof", data=dtof.transpose(2, 1, 0))
    with h5py.File(tmp_path / "dataset_OP.mat", "w") as handle:
        handle.create_dataset("dataset_OP", data=op.T)
    bundle = load_data_bundle(args)
    assert bundle.op_per_subject == 6
    assert bundle.num_subjects == 24
    assert bundle.temporal.shape[0] == 144
    observed = []
    for batch in bundle.make_aligned_loader("all", [0, 1], 64):
        observed.extend(batch[4].numpy().tolist())
    assert set(observed) == set(load_cohort(path, 24).subject_ids)
    assert 23 in observed and 2 not in observed


def test_shuffle_and_normalization_use_active_training_subjects(tmp_path, monkeypatch):
    path = cohort_file(tmp_path)
    cohort = load_cohort(path, 24)
    train, val = cohort.split(23)
    for name, value in {"num_subjects": 24, "cohort_file": str(path),
                         "structure_control": "shuffle", "structure_feature_dim": 2}.items():
        monkeypatch.setattr(CFG, name, value)
    raw = np.arange(48, dtype=np.float32).reshape(24, 2)
    raw[list(cohort.excluded_subject_ids)] = 100000
    shuffled, description = _apply_control(raw, 23, np.asarray(train), np.asarray(val))
    assert all(not np.array_equal(shuffled[sid], raw[sid]) for sid in train + val)
    assert any(np.array_equal(shuffled[23], raw[sid]) for sid in train)
    feature = tmp_path / "descriptor.npz"
    np.savez(feature, latent_raw=raw, subject_ids=np.arange(24),
             cohort_fingerprint=np.asarray(cohort.fingerprint),
             train_subject_ids=train, val_subject_ids=val, test_subject_ids=[23])
    monkeypatch.setattr(CFG, "structure_feature_path", str(feature))
    bank, _ = load_dtof_bank(23, np.asarray(train))
    np.testing.assert_allclose(bank[train].mean(axis=0), 0, atol=1e-6)
    assert np.max(np.abs(bank[train])) < 3
    with pytest.raises(ValueError, match="another held-out"):
        load_dtof_bank(22, np.asarray(train))
    np.savez(feature, latent_raw=raw, cohort_fingerprint=np.asarray("other"))
    with pytest.raises(ValueError, match="another cohort"):
        load_dtof_bank(23, np.asarray(train))
    with pytest.raises(ValueError, match="partition"):
        check_partition(cohort, train + [2], val, [23])


def test_gallery_maps_noncontiguous_original_ids():
    centroids = np.asarray([[1., 0.], [0., 1.], [-1., 0.]])
    metrics = query_metrics(np.asarray([[0., 1.]]), 5, centroids, subject_ids=[1, 5, 9])
    assert metrics["top1"] == 1
    assert metrics["gallery_subjects"] == 3
    assert metrics["most_confused_subject_id"] == 1


def test_preparation_retargets_every_config_and_notebook(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    shutil.copytree(ROOT / "configs", root / "configs")
    shutil.copytree(ROOT / "notebooks", root / "notebooks")
    path = cohort_file(tmp_path, raw=154, removed=(2, 7, 13, 18, 24, 31, 46, 59, 73, 88, 101, 117, 133, 148))
    module = load_script("prepare_subject_cohort")
    run = module.prepare(root, path)
    cohort = load_cohort(run / "cohort.json")
    assert len(cohort.subject_ids) == 140
    splits = json.loads((run / "split_manifest.json").read_text())
    assert len(splits) == 140
    assert all(len(row["train_subjects"]) == 129 for row in splits)
    configs = list((run / "configs/stage2").rglob("*.yaml"))
    expected_configs = {
        path.relative_to(root / "configs/stage2")
        for path in (root / "configs/stage2").rglob("*.yaml")
    }
    assert {path.relative_to(run / "configs/stage2") for path in configs} == expected_configs
    for config in configs:
        cfg = apply_dataclass_config(Config(), load_yaml(config))
        assert cfg.active_subject_ids == cohort.subject_ids
        assert run.name in Path(cfg.output_dir).parts
        assert cfg.num_subjects == 154
    for notebook in (run / "notebooks").glob("*.ipynb"):
        data = json.loads(notebook.read_text())
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == [] and cell["execution_count"] is None
                source = "".join(cell["source"])
                compile(source, str(notebook), "exec")
                assert "set(range(EXPECTED_SUBJECTS))" not in source
                assert "EXPECTED_SUBJECTS = 154" not in source
                assert "Path('artifacts/stage" not in source
    assert (run / "notebooks/07_unseen_dataset_validation_sto2.ipynb").is_file()
    runner = load_script("run_subject_cohort")
    monkeypatch.setattr(runner, "ROOT", root)
    queue = runner.commands(run, "all", cohort.subject_ids, [32])
    assert len(queue) == 140 * 2 + len(configs) + len(list((run / "notebooks").glob("*.ipynb")))
    assert all(str(root) in " ".join(command) for command in queue)
    assert module.prepare(root, path) == run


def test_resume_refuses_missing_or_different_cohort_metadata(tmp_path):
    from subject_nirs.common.cohort import check_existing_run
    cohort = Cohort(24, (2, 7, 13))
    (tmp_path / "loso_results.csv").write_text("fold,test_id\n")
    with pytest.raises(ValueError, match="no cohort"):
        check_existing_run(tmp_path, cohort)
    (tmp_path / "experiment_config.json").write_text(json.dumps({"cohort_fingerprint": "old"}))
    with pytest.raises(ValueError, match="another cohort"):
        check_existing_run(tmp_path, cohort)


def test_pending_preparation_creates_nothing(tmp_path):
    path = cohort_file(tmp_path, raw=154, confirmed=False)
    module = load_script("prepare_subject_cohort")
    with pytest.raises(ValueError, match="unconfirmed"):
        module.prepare(tmp_path / "repo", path)
    assert not (tmp_path / "repo/runs").exists()
