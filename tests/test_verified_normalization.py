import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from subject_nirs.stage2 import normalization as norm


class VerifiedNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = SimpleNamespace(input_dim=3, target_index=0, cohort_fingerprint="test",
                                   cache_chunk_rows=100, cache_dir=str(self.root / "cache"))
        self.infos = []
        rng = np.random.default_rng(91)
        for sid in [2, 7]:
            folder = self.root / str(sid)
            folder.mkdir()
            info = dict(subject_id=sid, n_rows=80)
            for key, arr in dict(x=rng.normal(size=(80, 3)).astype(np.float32),
                                 y=rng.uniform(10, 30, (80, 2)).astype(np.float32),
                                 subj=np.full(80, sid, dtype=np.int64)).items():
                p = folder / (key + ".npy")
                np.save(p, arr)
                info[key] = str(p)
            self.infos.append(info)

    def tearDown(self):
        self.tmp.cleanup()

    def test_values_match_training_only_reference(self):
        values = norm.load_or_fit(self.infos, self.cfg)
        x = np.concatenate([np.load(i["x"]) for i in self.infos])
        y = np.concatenate([np.load(i["y"])[:, :1] for i in self.infos]).astype(np.float64)
        np.testing.assert_array_equal(values[0], x.min(axis=0))
        np.testing.assert_array_equal(values[1], x.max(axis=0))
        np.testing.assert_allclose(values[2], y.mean(axis=0), rtol=1e-7)
        np.testing.assert_allclose(values[3], y.std(axis=0), rtol=1e-7)
        with self.assertRaises(ValueError):
            values[0][0] = 10

    def test_fusion_loads_same_record_without_refit(self):
        first = norm.load_or_fit(self.infos, self.cfg)
        with patch.object(norm, "_scan_once", side_effect=AssertionError("must not refit")):
            second = norm.load_or_fit(self.infos, self.cfg)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_target_and_train_partitions_have_distinct_records(self):
        first, _ = norm.record_path(self.infos, self.cfg)
        self.cfg.target_index = 1
        other, _ = norm.record_path(self.infos, self.cfg)
        single, _ = norm.record_path(self.infos[:1], self.cfg)
        self.assertNotEqual(first, other)
        self.assertNotEqual(other, single)

    def test_changed_source_is_rejected_even_if_statistics_unchanged(self):
        norm.load_or_fit(self.infos, self.cfg)
        p = self.infos[0]["x"]
        data = np.load(p)
        data[10, 0] += .001
        np.save(p, data)
        with self.assertRaisesRegex(ValueError, "cache changed"):
            norm.load_or_fit(self.infos, self.cfg)

    def test_record_corruption_is_rejected(self):
        norm.load_or_fit(self.infos, self.cfg)
        path, _ = norm.record_path(self.infos, self.cfg)
        value = json.loads(path.read_text())
        value["values"]["x_max"][0] *= 2
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "corrupt"):
            norm.load_or_fit(self.infos, self.cfg)

    def test_repeated_fit_disagreement_does_not_create_record(self):
        original = norm._scan_once
        calls = 0
        def intermittent(*args):
            nonlocal calls
            value = original(*args)
            calls += 1
            if calls == 2:
                value["x_max"][0] *= 2
            return value
        with patch.object(norm, "_scan_once", side_effect=intermittent):
            with self.assertRaisesRegex(ValueError, "Repeated"):
                norm.load_or_fit(self.infos, self.cfg)
        path, _ = norm.record_path(self.infos, self.cfg)
        self.assertFalse(path.exists())

    def test_foreign_baseline_scale_is_rejected(self):
        values = norm.load_or_fit(self.infos, self.cfg)
        actual = {k: a.copy() for k, a in zip(norm.KEYS, values)}
        norm.assert_matching(actual, values)
        actual["x_max"][0] += 1
        with self.assertRaisesRegex(ValueError, "retraining"):
            norm.assert_matching(actual, values)

    def test_float_conversion_corruption_is_rejected_before_persistence(self):
        original = np.asarray
        def bad_conversion(value, *args, **kwargs):
            result = original(value, *args, **kwargs)
            if kwargs.get("dtype") == np.float64 and result.shape == (80, 3):
                result = result.copy()
                result[0, 0] *= 2
            return result
        with patch.object(norm.np, "asarray", side_effect=bad_conversion):
            with self.assertRaisesRegex(ValueError, "conversion"):
                norm.load_or_fit(self.infos, self.cfg)

    def test_real_pretrained_builder_checks_normalization_before_loading_weights(self):
        import torch
        from subject_nirs.stage2.config import CFG
        from subject_nirs.stage2.loso import build_model_and_optimizer
        path = self.root / "bad_model.pth"
        values = norm.load_or_fit(self.infos, self.cfg)
        actual = {k: a.copy() for k, a in zip(norm.KEYS, values)}
        actual["test_id"] = 0
        actual["x_max"][0] *= 2
        torch.save(dict(config=actual, model_state_dict={}), path)
        with patch.multiple(CFG, cohort_file=None, experiment_mode="fusion", baseline_init="pretrained",
                            baseline_train="finetune", baseline_checkpoint_path=str(path)):
            with self.assertRaisesRegex(ValueError, "Pretrained baseline.*x_max"):
                build_model_and_optimizer(32, 1, 0, normalization=values)

    def test_bad_subject_id_and_nonfinite_cache_rejected(self):
        bad = dict(self.infos[0], subject_id=99)
        with self.assertRaisesRegex(ValueError, "subject"):
            norm.scan_verified(bad, 3)
        data = np.load(self.infos[0]["x"])
        data[0, 0] = np.nan
        np.save(self.infos[0]["x"], data)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            norm.scan_verified(self.infos[0], 3)



if __name__ == "__main__":
    unittest.main()
