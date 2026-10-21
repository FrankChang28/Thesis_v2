from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

from subject_nirs.common.cohort import Cohort
from subject_nirs.stage1.analysis_support import validate_features
from subject_nirs.stage1.evaluation import load_existing_rows

ROOT = Path(__file__).resolve().parents[1]


def test_probe_validation_rejects_excluded_subjects_and_mixed_cohorts(tmp_path):
    cohort = Cohort(15, (2,))
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps({"raw_num_subjects": 154, "remove_subject_ids": [2], "confirmed": True}))
    # Use the production-sized raw layout, with all active folds represented.
    cohort = Cohort(154, (2,))
    files = []
    for sid in cohort.subject_ids:
        train, val = cohort.split(sid)
        ids = np.arange(154)
        splits = np.full(154, "excluded", dtype="U8")
        splits[train], splits[val], splits[sid] = "train", "val", "test"
        dest = tmp_path / f"{sid}.npz"
        np.savez(dest, cohort_fingerprint=cohort.fingerprint, subject_ids=ids, splits=splits)
        files.append(dest)
    assert validate_features(files, path).fingerprint == cohort.fingerprint
    with pytest.raises(ValueError, match="exactly one"):
        validate_features(files[:-1], path)
    with np.load(files[0]) as data:
        splits = data['splits'].copy()
    splits[2] = 'train'
    np.savez(files[0], cohort_fingerprint=cohort.fingerprint, subject_ids=ids, splits=splits)
    with pytest.raises(ValueError, match="active IDs"):
        validate_features(files, path)
    np.savez(files[0], cohort_fingerprint='other', subject_ids=ids, splits=splits)
    with pytest.raises(ValueError, match="another cohort"):
        validate_features(files, path)


def test_retrieval_cache_requires_checkpoint_contents_and_op_seed(tmp_path):
    from subject_nirs.stage1.analysis_support import file_sha256
    checkpoint = tmp_path / 'best_model.pt'
    checkpoint.write_bytes(b'first model')
    cache = tmp_path / 'full_grid_evaluation.json'
    payload = {'experiment':'relat_cons_32','op_split_seed':2026,
               'checkpoint_sha256':file_sha256(checkpoint)}
    cache.write_text(json.dumps(payload))
    assert 0 in load_existing_rows([(0,checkpoint)],'relat_cons_32')
    assert not load_existing_rows([(0,checkpoint)],'relat_cons_32',2027)
    checkpoint.write_bytes(b'changed model, same path')
    assert not load_existing_rows([(0,checkpoint)],'relat_cons_32')


def test_frozen_stage1_templates_have_aligned_probe_and_new_tsne(tmp_path):
    spec = importlib.util.spec_from_file_location('prepare_analysis',ROOT/'scripts/prepare_subject_cohort.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cohort = Cohort(154,(2,))
    frozen = tmp_path / 'runs' / cohort.fingerprint / 'cohort.json'
    for path in (ROOT/'notebooks').glob('0[0-3]_*.ipynb'):
        nb = module.cohort_notebook(json.loads(path.read_text()),tmp_path,cohort,frozen)
        text = '\n'.join(''.join(cell['source']) for cell in nb['cells'])
        assert str(frozen) in text
        assert str(tmp_path) in text
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),str(path),'exec')
        if path.name.startswith('03'):
            assert "RUN_DIR / 'metadata_aligned.csv'" in text
            assert 'data/raw/stage1/temp_metadata_v2.csv' not in text


def test_cka_cache_rejects_changed_checkpoint_at_same_path(tmp_path, monkeypatch):
    from subject_nirs.stage1 import cka
    from subject_nirs.stage1.analysis_support import file_sha256
    monkeypatch.setattr(cka, 'EXPECTED_OPS', 4)
    checkpoint = tmp_path / 'best_model.pt'
    checkpoint.write_bytes(b'first checkpoint')
    cache = tmp_path / 'cache.npz'
    vectors = np.zeros((4,121),dtype=np.float32)
    vectors[:,0] = 1
    np.savez(cache,cka_vectors=vectors,source_checkpoint=str(checkpoint),
             experiment='relat_cons_32',checkpoint_sha256=file_sha256(checkpoint))
    result = cka.load_valid_cache(cache,checkpoint,'relat_cons_32')
    assert result is not None
    result.close()
    checkpoint.write_bytes(b'changed checkpoint')
    assert cka.load_valid_cache(cache,checkpoint,'relat_cons_32') is None


def test_cli_notebook_execution_shares_variables_without_jupyter(tmp_path):
    spec = importlib.util.spec_from_file_location('analysis_runner',ROOT/'scripts/run_stage1_analysis.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path/'result.txt'
    path = tmp_path/'analysis.ipynb'
    path.write_text(json.dumps({'cells':[
        {'cell_type':'markdown','source':['Description']},
        {'cell_type':'code','source':['value = 41']},
        {'cell_type':'code','source':[f'from pathlib import Path; Path({str(output)!r}).write_text(str(value + 1))']},
    ]}))
    module.execute_notebook(path)
    assert output.read_text() == '42'
