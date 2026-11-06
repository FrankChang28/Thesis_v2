from __future__ import annotations

import csv
import importlib.util
import sys

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_remaining_stage2.py"
SPEC = importlib.util.spec_from_file_location("run_remaining_stage2", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_selected_test_ids_respects_range_and_limit() -> None:
    config = MODULE.Config()
    config.num_subjects = 5
    config.test_subject_start = 2
    config.test_subject_end = 4
    config.max_folds = 2
    assert MODULE.selected_test_ids(config) == frozenset({1, 2})


def test_audit_accepts_checkpoint_relocated_with_output_directory() -> None:
    with TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        checkpoint = output_dir / "best_model_fold_001_test_id_0.pth"
        checkpoint.touch()
        results = output_dir / "loso_results.csv"
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
                    "model_path": (
                        "artifacts/stage2/legacy_long_name/"
                        "best_model_fold_001_test_id_0.pth"
                    ),
                }
            )
        assert MODULE.completed_test_ids(results, output_dir) == frozenset({0})


if __name__ == "__main__":
    test_selected_test_ids_respects_range_and_limit()
    test_audit_accepts_checkpoint_relocated_with_output_directory()
    print("Remaining Stage 2 runner tests: OK")


def test_pending_queue_prioritizes_dtof_before_metadata(tmp_path, monkeypatch):
    cases = {
        "a_metadata.yaml": MODULE.Config(subject_feature_source="metadata"),
        "z_dtof.yaml": MODULE.Config(subject_feature_source="dtof"),
        "m_baseline.yaml": MODULE.Config(experiment_mode="baseline", baseline_init="scratch"),
        "b_sto2.yaml": MODULE.Config(target_mode="sto2"),
        "c_control.yaml": MODULE.Config(structure_control="zero"),
    }
    for name in cases:
        (tmp_path / name).touch()
    def fake_status(path):
        config = cases[path.name]
        return config, MODULE.ExperimentStatus(
            path, tmp_path / path.stem, frozenset(), frozenset({0})
        )
    monkeypatch.setattr(MODULE, "load_status", fake_status)
    queue = MODULE.discover_statuses(tmp_path, {"hc"})
    assert [s.config_path.name for s in queue] == [
        "m_baseline.yaml", "z_dtof.yaml", "a_metadata.yaml"
    ]


@pytest.mark.parametrize("target", ["hc", "sto2"])
def test_canonical_configs_use_new_finetune_settings(target):
    from subject_nirs.common.config import apply_dataclass_config, load_yaml
    root = SCRIPT_PATH.parents[1]
    outputs = set()
    for feature in ["dtof", "metadata"]:
        for method in ["concatenation", "film", "residual"]:
            path = root / f"configs/stage2/{method}/finetune/{feature}_{target}.yaml"
            cfg = MODULE.Config()
            apply_dataclass_config(cfg, load_yaml(path))
            assert cfg.learning_rate == 3e-5 and cfg.baseline_optimizer_lr == 1e-6
            assert cfg.output_variant == "" and cfg.output_dir.endswith(f"/{target}")
            assert cfg.baseline_init == "pretrained" and cfg.baseline_train == "finetune"
            assert cfg.target_mode == target and cfg.resume and cfg.fusion_ramp_epochs == 0
            assert cfg.subject_feature_source == feature and cfg.fusion_method == method
            assert cfg.baseline_checkpoint_path.endswith(
                f"baseline/{target}/best_model_fold_{{fold:03d}}_test_id_{{test_id}}.pth"
            )
            assert cfg.output_dir not in outputs
            outputs.add(cfg.output_dir)
            assert not path.with_name(f"{feature}_{target}_lr3e5.yaml").exists()


@pytest.mark.parametrize("target", ["hc", "sto2"])
def test_all_canonical_scratch_configs_are_progressive(target):
    from subject_nirs.common.config import apply_dataclass_config, load_yaml
    root = SCRIPT_PATH.parents[1]
    for feature in ["dtof", "metadata"]:
        for method in ["concatenation", "film", "residual"]:
            cfg = MODULE.Config()
            apply_dataclass_config(cfg, load_yaml(root / f"configs/stage2/{method}/scratch/{feature}_{target}.yaml"))
            assert cfg.baseline_init == "scratch" and cfg.baseline_train == "finetune"
            assert cfg.learning_rate == cfg.baseline_optimizer_lr == 1e-4
            assert cfg.fusion_ramp_epochs == 1 and cfg.output_variant == ""
            assert cfg.subject_feature_source == feature and cfg.fusion_method == method
            assert cfg.target_mode == target and cfg.resume


def test_audit_rejects_folds_trained_on_other_descriptors(tmp_path):
    descriptor = tmp_path / "stage1/0/dtof.npz"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_bytes(b"current descriptors")
    config = MODULE.Config(subject_feature_source="dtof",
                           structure_feature_path=str(tmp_path / "stage1/{test_id}/dtof.npz"))
    output = tmp_path / "out"
    (output / "configs").mkdir(parents=True)
    fold_config = output / "configs/config_fold_001_test_id_0.json"
    status = MODULE.ExperimentStatus(tmp_path / "x.yaml", output, frozenset({0}), frozenset({0}))
    fold_config.write_text('{"structure_sha256": "%s"}' % MODULE.file_sha256(descriptor))
    MODULE.check_descriptors(config, status)
    for recorded in ("{}", '{"structure_sha256": "old"}'):
        fold_config.write_text(recorded)
        with pytest.raises(ValueError, match="other Stage 1 descriptors"):
            MODULE.check_descriptors(config, status)
