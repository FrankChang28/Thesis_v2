"""CPU checks for the fixed-range DTOF preprocessing."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from subject_nirs.stage1 import data as stage1_data
from subject_nirs.stage1.evaluation import _stats_arrays, preprocess_batch

OP_PER_SUBJECT, NUM_SDS, NUM_BINS = 6, 5, 40


def _raw_dtof(subjects=4, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.gamma(2.0, 1e-9, size=(subjects * OP_PER_SUBJECT, NUM_SDS, NUM_BINS))
    raw[:, :, :3] = 0.0  # before photon arrival
    raw[rng.random(raw.shape) < 0.05] = 0.0  # sparse zeros on the rising edge
    return raw.astype(np.float32)


def _preprocess(raw):
    split = stage1_data.SplitInfo(np.array([0, 1]), np.array([2]), np.array([3]))
    conditions = torch.rand(raw.shape[0], 3)
    return stage1_data._preprocess(torch.from_numpy(raw.copy()), conditions, split, OP_PER_SUBJECT)


def test_fixed_range_formula():
    raw = _raw_dtof()
    temporal, _, _, stats = _preprocess(raw)
    cleaned = stage1_data.zero_before_onset(torch.from_numpy(raw.copy())).numpy()
    fraction = cleaned / cleaned.sum(2, keepdims=True)
    expected = (np.clip(np.log10(np.maximum(fraction, 1e-10)), -10, 0) + 10) / 10
    np.testing.assert_allclose(temporal.numpy(), expected, atol=1e-5)
    assert float(temporal.min()) == 0.0 and float(temporal.max()) < 1.0
    assert stats.to_dict()["dtof_preprocess"] == stage1_data.PREPROCESS_VERSION


def test_zero_area_curve_is_rejected():
    raw = _raw_dtof()
    raw[5, 2] = 0.0
    with pytest.raises(ValueError, match="zero area"):
        _preprocess(raw)


def _curve(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1)


def test_onset_zeroes_only_bins_before_first_run():
    values = [0, 1, 0, 2, 2, 0, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4]
    cleaned = stage1_data.zero_before_onset(_curve(values)).flatten().tolist()
    assert cleaned == [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4]


def test_onset_leaves_a_curve_without_isolated_bins_unchanged():
    values = [0, 0, 1, 2, 3, 4, 5, 6]
    assert stage1_data.zero_before_onset(_curve(values)).flatten().tolist() == values


def test_onset_ignores_absorption_scaling():
    raw = torch.from_numpy(_raw_dtof())
    weights = torch.rand(raw.shape, generator=torch.Generator().manual_seed(1)) * 0.9 + 0.1
    dimmer = stage1_data.zero_before_onset(raw * weights)
    original = stage1_data.zero_before_onset(raw.clone())
    assert torch.equal(dimmer > 0, original > 0)


def test_curve_without_continuous_segment_is_rejected():
    with pytest.raises(ValueError, match="consecutive nonzero"):
        stage1_data.zero_before_onset(_curve([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]))


def test_areas_exclude_removed_bins():
    values = [9.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    temporal, relative_area = stage1_data.dtof_inputs(torch.tensor([[values, values]], dtype=torch.float32))
    assert float(temporal[0, 0, 0]) == 0.0
    assert float(temporal[0, 0, 2]) == pytest.approx(1 + np.log10(1 / 6) / 10)
    assert relative_area[0].tolist() == pytest.approx([np.log10(0.5)] * 2)


def test_evaluation_matches_training():
    raw = _raw_dtof()
    temporal, relative_area, _, stats = _preprocess(raw)
    eval_temporal, eval_area = preprocess_batch(raw, _stats_arrays({"preprocess_stats": stats.to_dict()}))
    torch.testing.assert_close(eval_temporal, temporal, rtol=0, atol=1e-6)
    torch.testing.assert_close(eval_area, relative_area, rtol=0, atol=1e-5)


def test_evaluation_rejects_older_preprocessing():
    old = {"temporal_low": -10.0, "temporal_high": -1.2, "area_mean": [0.0], "area_std": [1.0]}
    with pytest.raises(ValueError, match="retrain Stage 1"):
        _stats_arrays({"preprocess_stats": old})
