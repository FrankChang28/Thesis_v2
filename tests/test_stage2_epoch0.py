"""CPU regression checks for baseline candidates and selection semantics."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from subject_nirs.stage2.config import Config
from subject_nirs.stage2.model import CNN1D, build_baseline_fusion_model
from subject_nirs.stage2.selection import (
    ValidationSelection, check_selection_namespace, make_baseline_candidate,
)


class EpochZeroTests(unittest.TestCase):
    def test_candidates_equal_baseline_and_roundtrip_without_changing_training(self):
        torch.manual_seed(7)
        x, z = torch.randn(16, 30), torch.randn(16, 32)
        for method in ["concatenation", "film", "residual"]:
            with self.subTest(method=method):
                model = build_baseline_fusion_model(
                    method, baseline=CNN1D(), latent_dim=32, output_dim=1,
                    freeze_baseline=False, residual_use_gate=True,
                    residual_gate_init_logit=0.0,
                ).train()
                before = copy.deepcopy(model.state_dict())
                rng = torch.get_rng_state().clone()
                candidate = make_baseline_candidate(model)
                self.assertTrue(model.training)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(all(torch.equal(before[k], v) for k,v in model.state_dict().items()))
                with torch.no_grad():
                    expected = candidate.baseline(x)
                    torch.testing.assert_close(candidate(x,z), expected, rtol=1e-6, atol=1e-7)
                    torch.testing.assert_close(candidate(x,z*5), expected, rtol=1e-6, atol=1e-7)
                with tempfile.TemporaryDirectory() as tmp:
                    p=Path(tmp)/"candidate.pth"
                    torch.save({"model_state_dict":candidate.state_dict()},p)
                    restored=copy.deepcopy(model).eval()
                    restored.load_state_dict(torch.load(p,weights_only=False)["model_state_dict"],strict=True)
                    with torch.no_grad():
                        torch.testing.assert_close(restored(x,z),expected,rtol=1e-6,atol=1e-7)

    def test_selection_keeps_epoch_zero_when_training_never_beats_it(self):
        state=ValidationSelection(best_loss=0.5)
        self.assertEqual([state.observe(i,v) for i,v in enumerate([1.,.8,.9],1)], [False]*3)
        self.assertEqual((state.best_epoch,state.best_loss,state.patience),(0,.5,1))

    def test_selection_promotes_better_epoch_and_keeps_tie(self):
        state=ValidationSelection(best_loss=.5)
        self.assertFalse(state.observe(1,.5))
        self.assertTrue(state.observe(2,.4))
        self.assertFalse(state.observe(3,.4))
        self.assertEqual((state.best_epoch,state.best_loss),(2,.4))

    def test_epoch_zero_does_not_change_early_stopping(self):
        guarded,legacy=ValidationSelection(best_loss=.1),ValidationSelection()
        for i,v in enumerate([1.,.8,.9,.7,.8,.8],1):
            guarded.observe(i,v); legacy.observe(i,v)
            self.assertEqual(guarded.patience,legacy.patience)
        self.assertEqual(guarded.best_epoch,0)
        self.assertEqual(legacy.best_epoch,4)

    def test_selection_namespace_rejects_mixed_completed_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"loso_results.csv").touch()
            check_selection_namespace(tmp,False)
            with self.assertRaisesRegex(ValueError,"separate output"):
                check_selection_namespace(tmp,True)
            (root/"experiment_config.json").write_text(json.dumps({"include_baseline_epoch0":True}))
            check_selection_namespace(tmp,True)
            with self.assertRaises(ValueError):check_selection_namespace(tmp,False)

    def test_config_requires_pretrained_fusion(self):
        for cfg in [Config(experiment_mode="baseline",include_baseline_epoch0=True),
                    Config(baseline_init="scratch",include_baseline_epoch0=True)]:
            with self.assertRaisesRegex(ValueError,"pretrained fusion"):
                cfg._validate_experiment()
        Config(include_baseline_epoch0=True)._validate_experiment()
        self.assertFalse(Config().include_baseline_epoch0)

    def test_extra_validation_can_preserve_training_rng(self):
        loader=DataLoader(TensorDataset(torch.ones(8,1)),batch_size=4)
        torch.manual_seed(23); before=torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):list(loader)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
