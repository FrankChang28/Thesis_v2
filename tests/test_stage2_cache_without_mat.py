"""Transferred caches must support real LOSO training without a MAT dataset."""

import hashlib
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from subject_nirs.stage2 import cache, compact_cache, loso
from subject_nirs.stage2.config import CFG, Config


@pytest.fixture
def cached_data(tmp_path, monkeypatch):
    cfg = Config(
        experiment_mode="baseline", baseline_init="scratch",
        num_subjects=5, val_subject_count=2, run_all_loso=False,
        test_subject_id=0, test_subject_start=None, test_subject_end=None,
        resume=False, epochs=1, batch_size=4,
        train_sims_per_subject=1, train_rows_per_sim=16,
        val_sims_per_subject=1, val_rows_per_sim=4,
        max_train_rows_per_subject=8, sim_windows_per_sim=2,
        cache_chunk_rows=8, compact_cache_chunk_rows=8,
        mat_file=str(tmp_path / "source.mat"),
        cache_root=str(tmp_path / "cache"),
        output_dir_override=str(tmp_path / "baseline"),
        save_predictions=True,
    )
    for name, value in asdict(cfg).items():
        monkeypatch.setattr(CFG, name, value)
    monkeypatch.setattr(loso, "DEVICE", torch.device("cpu"))
    rng = np.random.default_rng(12)
    with h5py.File(CFG.mat_file, "w") as source:
        for split, rows_per_subject in (("Train", 16), ("Val", 4)):
            n = 5 * rows_per_subject
            source[f"{split}_Dataset_X"] = rng.uniform(.2, 1.2, (n, 30))
            source[f"{split}_Dataset_Y"] = rng.uniform(20, 80, (n, 2))
            source[f"{split}_Dataset_SubjID"] = np.repeat(
                np.arange(1, 6), rows_per_subject
            )[:, None]
        source["num_val_rows"] = 20
    cache.build_preprocessed_cache()
    with h5py.File(CFG.mat_file, "r") as source:
        infos = [compact_cache.build_or_load_subject_compact_cache_from_mat(
            source, subject, 0
        ) for subject in range(5)]
    bank = tmp_path / "dtof.npz"
    np.savez(bank, latent_raw=rng.normal(size=(5, 32)).astype(np.float32),
             subject_ids=np.arange(5), holdout_subject_ids=np.array([0]))
    monkeypatch.setattr(CFG, "structure_feature_path", str(bank))
    return infos


def forbid_mat(*args, **kwargs):
    raise AssertionError("Complete caches must never open the MAT file")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint():
    return next(Path(CFG.output_dir).glob("best_model*.pth"))


@pytest.mark.parametrize("target", ["hc", "sto2"])
def test_baseline_and_finetune_train_and_resume_without_mat(
    cached_data, monkeypatch, tmp_path, target
):
    monkeypatch.setattr(CFG, "target_mode", target)
    Path(CFG.mat_file).unlink()
    monkeypatch.setattr(h5py, "File", forbid_mat)
    loso.run_loso()
    baseline_path = checkpoint()
    before = sha(baseline_path)
    assert len(list((Path(CFG.output_dir) / "predictions").glob("*_test_predictions.npz"))) == 1
    monkeypatch.setattr(CFG, "resume", True)
    loso.run_loso()
    assert sha(baseline_path) == before
    monkeypatch.setattr(CFG, "experiment_mode", "fusion")
    monkeypatch.setattr(CFG, "baseline_init", "pretrained")
    monkeypatch.setattr(CFG, "fusion_method", "concatenation")
    monkeypatch.setattr(CFG, "baseline_checkpoint_path", str(baseline_path))
    monkeypatch.setattr(CFG, "output_dir_override", str(tmp_path / "finetune"))
    loso.run_loso()
    assert checkpoint().is_file()
    assert len(list((Path(CFG.output_dir) / "predictions").glob("*_test_predictions.npz"))) == 1
    assert sha(baseline_path) == before
    # A completed DTOF fold is reused only while its descriptor file is unchanged.
    monkeypatch.setattr(CFG, "resume", True)
    loso.run_loso()
    bank = Path(CFG.structure_feature_path)
    with np.load(bank) as data:
        arrays = dict(data)
    arrays["latent_raw"] = arrays["latent_raw"] + 1
    np.savez(bank, **arrays)
    with pytest.raises(ValueError, match="other Stage 1 descriptors"):
        loso.run_loso()


@pytest.mark.parametrize("method", ["concatenation", "film", "residual"])
@pytest.mark.parametrize("feature", ["dtof", "metadata"])
def test_progressive_scratch_without_mat(cached_data, monkeypatch, tmp_path, method, feature):
    Path(CFG.mat_file).unlink()
    monkeypatch.setattr(h5py, "File", forbid_mat)
    monkeypatch.setattr(CFG, "experiment_mode", "fusion")
    monkeypatch.setattr(CFG, "fusion_method", method)
    monkeypatch.setattr(CFG, "subject_feature_source", feature)
    if feature == "metadata":
        metadata = tmp_path / "metadata.csv"
        np.savetxt(metadata, np.column_stack([np.zeros((5, 5)), np.arange(20).reshape(5, 4)]),
                   delimiter=",", header=",".join(f"f{i}" for i in range(9)), comments="")
        monkeypatch.setattr(CFG, "metadata_feature_path", str(metadata))
        monkeypatch.setattr(CFG, "structure_feature_dim", 4)
    monkeypatch.setattr(CFG, "fusion_ramp_epochs", 1)
    loso.run_loso()
    assert checkpoint().is_file()


@pytest.mark.parametrize("force_rebuild", [False, True])
def test_missing_raw_fails_before_overwriting_cache(
    cached_data, monkeypatch, force_rebuild
):
    info = cached_data[1]
    if not force_rebuild:
        Path(info["meta"]).unlink()
    monkeypatch.setattr(CFG, "rebuild_compact_cache", force_rebuild)
    Path(CFG.mat_file).unlink()
    before = {k: sha(info[k]) for k in ("x", "y", "subj")}
    with pytest.raises(FileNotFoundError, match="subject=1, stage=1"):
        compact_cache.build_or_load_subject_compact_cache_from_mat(
            lambda: h5py.File(CFG.mat_file, "r"), 1, 0
        )
    assert {k: sha(info[k]) for k in before} == before


def test_lazy_and_direct_cache_builds_have_identical_rows(cached_data):
    info = cached_data[1]
    expected = {k: np.load(info[k]).copy() for k in ("x", "y", "subj")}
    Path(info["meta"]).unlink()
    with h5py.File(CFG.mat_file, "r") as source:
        calls = []
        def lazy_source():
            calls.append(1)
            return source
        rebuilt = compact_cache.build_or_load_subject_compact_cache_from_mat(
            lazy_source, 1, 0
        )
        assert calls == [1]
        compact_cache.build_or_load_subject_compact_cache_from_mat(lazy_source, 1, 0)
        assert calls == [1]
    for k, expected_rows in expected.items():
        np.testing.assert_array_equal(np.load(rebuilt[k]), expected_rows)


def test_loso_opens_raw_once_only_when_needed(cached_data, monkeypatch, tmp_path):
    # Save the original run's learned weights before rebuilding its inputs.
    loso.run_loso()
    expected = torch.load(checkpoint(), map_location="cpu", weights_only=False)
    training, _ = loso.split_subjects(CFG.active_subject_ids, CFG.test_subject_id)
    for subject in training:
        Path(cached_data[subject]["meta"]).unlink()
    monkeypatch.setattr(CFG, "output_dir_override", str(tmp_path / "rebuilt"))
    real_file = h5py.File
    opened = []
    def tracked_file(*args, **kwargs):
        source = real_file(*args, **kwargs)
        opened.append(source)
        return source
    monkeypatch.setattr(h5py, "File", tracked_file)
    loso.run_loso()
    assert len(opened) == 1
    assert not opened[0].id.valid
    actual = torch.load(checkpoint(), map_location="cpu", weights_only=False)
    for key, value in expected["model_state_dict"].items():
        torch.testing.assert_close(actual["model_state_dict"][key], value, rtol=0, atol=0)


def test_corrupt_training_cache_is_rejected(cached_data, monkeypatch):
    training, _ = loso.split_subjects(CFG.active_subject_ids, 0)
    info = cached_data[training[0]]
    x = np.load(info["x"]).copy()
    x[0, 0] = np.nan
    np.save(info["x"], x)
    before = sha(info["x"])
    Path(CFG.mat_file).unlink()
    monkeypatch.setattr(h5py, "File", forbid_mat)
    with pytest.raises(ValueError, match="Non-finite training cache"):
        loso.run_loso()
    assert sha(info["x"]) == before


def test_missing_validation_cache_still_needs_raw(cached_data):
    Path(cache.cache_metadata_path()).unlink()
    Path(CFG.mat_file).unlink()
    with pytest.raises(FileNotFoundError):
        cache.build_preprocessed_cache()
