from pathlib import Path

import h5py
import numpy as np
import pytest

from subject_nirs.stage1.evaluation import LazyDTOFArray


def test_preload_is_exact_for_source_precision_original_ids_and_query_order(tmp_path):
    path = tmp_path / "dataset_dtof.mat"
    source = np.random.default_rng(61).normal(size=(7, 3, 31)).astype(np.float64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("dataset_dtof", data=source, chunks=(7, 3, 2), compression="gzip")
    lazy, preloaded = LazyDTOFArray(path), LazyDTOFArray(path, preload=True)
    try:
        for ids in [[], [0], [30, 0, 17, 4], list(range(31))]:
            np.testing.assert_array_equal(preloaded[ids], lazy[ids])
            assert preloaded[ids].dtype == np.float32
        for reader in [lazy, preloaded]:
            with pytest.raises(ValueError, match="Duplicate"):
                reader[[2,2]]
            with pytest.raises(IndexError):
                reader[[31]]
    finally:
        lazy.close()
        preloaded.close()
    assert preloaded._preloaded is None
