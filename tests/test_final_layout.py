import copy
import json
from types import MethodType, SimpleNamespace

import pytest
import torch

from subject_nirs.common.cohort import Cohort, check_output_namespace
from subject_nirs.stage2.config import Config
from subject_nirs.stage2.curriculum import configure_progressive_fusion
from subject_nirs.stage2.model import CNN1D, build_baseline_fusion_model


def test_canonical_output_requires_correct_cohort_marker(tmp_path):
    cohort=Cohort(24,(2,7,13))
    output=tmp_path/'artifacts/stage2/model'
    with pytest.raises(ValueError):check_output_namespace(output,cohort)
    marker=tmp_path/'artifacts/.cohort.json';marker.parent.mkdir()
    marker.write_text(json.dumps({'fingerprint':cohort.fingerprint}))
    check_output_namespace(output,cohort)
    marker.write_text(json.dumps({'fingerprint':'different'}))
    with pytest.raises(ValueError,match='another cohort'):check_output_namespace(output,cohort)


def test_versions_have_distinct_paths_and_ramp_scope():
    c=Config(fusion_method='concatenation',baseline_init='scratch')
    old=c.output_dir;c.output_variant='progressive_r1'
    assert c.output_dir != old and c.output_dir.endswith('hc_progressive_r1')
    c.fusion_ramp_epochs=1;c.validate()
    c.baseline_init='pretrained'
    with pytest.raises(ValueError,match='scratch fusion'):c.validate()


def test_streaming_metadata_steps_without_iterating_loader():
    class NoIteration:
        infos=[{'n_rows':50000}]*125
        drop_last=True
        def __iter__(self):raise AssertionError('Do not consume training RNG')
    model=SimpleNamespace()
    loader=SimpleNamespace(dataset=NoIteration(),batch_size=None,num_workers=0)
    configure_progressive_fusion(model,[loader],SimpleNamespace(fusion_ramp_epochs=1,batch_size=4096))
    assert model.fusion_ramp_total_steps==1525 and model.fusion_ramp_step==0


def test_native_ramp_matches_frozen_pilot_values_and_gradients():
    torch.manual_seed(42)
    native=build_baseline_fusion_model('concatenation',baseline=CNN1D(output_dim=1,latent_dim=0),
                                     latent_dim=32,output_dim=1,freeze_baseline=False,
                                     residual_use_gate=True,residual_gate_init_logit=0.)
    pilot=copy.deepcopy(native)
    original=pilot.adapt_embedding
    def adapt(self,h,z):
        fused,parts=original(h,z)
        if not self.training:return fused,parts
        alpha=min(1.,self.pilot_step/3);self.pilot_step+=1
        return (fused if alpha==1. else (1.-alpha)*h+alpha*fused),parts
    pilot.adapt_embedding=MethodType(adapt,pilot);pilot.pilot_step=0
    native.fusion_ramp_total_steps=4
    x=torch.randn(8,30);z=torch.randn(8,32)
    for step in range(6):
        native.train();pilot.train();native.zero_grad();pilot.zero_grad()
        state=torch.random.get_rng_state();a=native(x,z);a.sum().backward()
        torch.random.set_rng_state(state);b=pilot(x,z);b.sum().backward()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for pa,pb in zip(native.parameters(),pilot.parameters()):
            if pa.grad is None:assert pb.grad is None
            else:torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
    before=native.fusion_ramp_step
    native.eval();pilot.eval();torch.testing.assert_close(native(x,z),pilot(x,z),rtol=0,atol=0)
    assert native.fusion_ramp_step==before
    assert native.state_dict().keys()==pilot.state_dict().keys()


def test_notebooks_use_canonical_relative_paths():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for path in (root / 'notebooks').glob('*.ipynb'):
        book = json.loads(path.read_text())
        for cell in book['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            for prefix in ('artifacts', 'cache', 'results'):
                assert prefix + '/01deae79e5b1d272/' not in source, str(path)
            compile(source, str(path), 'exec')


@pytest.mark.parametrize("method", ["concatenation", "film", "residual"])
@pytest.mark.parametrize("latent_dim", [4, 32])
def test_progressive_fusion_values_gradients_and_full_evaluation(method, latent_dim, monkeypatch):
    from subject_nirs.stage2.config import CFG
    monkeypatch.setattr(CFG, "dropout", 0.)
    torch.manual_seed(52)
    model = build_baseline_fusion_model(
        method, baseline=CNN1D(output_dim=1), latent_dim=latent_dim, output_dim=1,
        freeze_baseline=False, residual_use_gate=True, residual_gate_init_logit=0.
    )
    # FiLM/residual start at identity; make a nonzero learned adapter so a
    # broken ramp cannot pass merely because both branches initially agree.
    with torch.no_grad():
        adapter = (model.film_conditioner[-1] if method == "film" else model.latent_proj[-1])
        adapter.weight.normal_(0., .05)
        adapter.bias.fill_(.1)
    reference = copy.deepcopy(model)
    keys = tuple(model.state_dict())
    model.fusion_ramp_total_steps = 3
    x, z = torch.randn(8, 30), torch.randn(8, latent_dim)
    for step, alpha in enumerate([0., .5, 1., 1.]):
        model.train(); reference.train(); model.zero_grad(); reference.zero_grad()
        h = reference._encode_baseline(x)
        fused, _ = reference.adapt_embedding(h, z)
        expected_h = fused if alpha == 1. else (1.-alpha)*h + alpha*fused
        expected = reference.baseline.regression_head(expected_h)
        actual = model(x, z, return_parts=True)
        torch.testing.assert_close(actual["adapted_drs_embedding"], expected_h, rtol=0, atol=0)
        torch.testing.assert_close(actual["pred"], expected, rtol=0, atol=0)
        actual["pred"].square().mean().backward()
        expected.square().mean().backward()
        for (name, p), (other_name, q) in zip(model.named_parameters(), reference.named_parameters()):
            assert name == other_name
            if p.grad is None: assert q.grad is None
            else: torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        if step == 0:
            assert all(p.grad is None or not p.grad.any()
                       for name, p in model.named_parameters() if not name.startswith("baseline."))
            assert any(p.grad is not None and p.grad.any() for p in model.baseline.parameters())
        if step == 2:
            assert any(p.grad is not None and p.grad.any()
                       for name, p in model.named_parameters() if not name.startswith("baseline."))
    model.fusion_ramp_step = 0
    model.eval(); reference.eval()
    torch.testing.assert_close(model(x, z), reference(x, z), rtol=0, atol=0)
    assert model.fusion_ramp_step == 0 and tuple(model.state_dict()) == keys
