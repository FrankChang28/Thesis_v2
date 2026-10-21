"""Cohort validation and fresh selected-fold plots for Stage 1 analyses."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..common.cohort import check_partition, load_cohort


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_checkpoints(checkpoints, cohort_file, *, complete=True):
    from .evaluation import load_checkpoint
    cohort = load_cohort(cohort_file, require_confirmed=True)
    ids = [fold for fold, _ in checkpoints]
    if len(ids) != len(set(ids)) or not set(ids).issubset(cohort.subject_ids):
        raise ValueError("Duplicate, excluded or unknown analysis folds")
    if complete and set(ids) != set(cohort.subject_ids):
        raise ValueError("Analysis requires every active cohort fold")
    for fold, path in checkpoints:
        checkpoint = load_checkpoint(path)
        if checkpoint.get("cohort") != cohort.metadata():
            raise ValueError(f"Checkpoint belongs to another cohort: {path}")
        if list(map(int, checkpoint["test_subjects"])) != [fold]:
            raise ValueError(f"Checkpoint test ID differs from its fold: {path}")
        check_partition(cohort, checkpoint["train_subjects"], checkpoint["val_subjects"], [fold])
    return cohort


def validate_features(paths, cohort_file):
    cohort = load_cohort(cohort_file, require_confirmed=True)
    seen = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["cohort_fingerprint"].item()) != cohort.fingerprint:
                raise ValueError(f"Feature belongs to another cohort: {path}")
            ids, splits = data["subject_ids"], data["splits"].astype(str)
            active = ids[np.isin(splits, ["train", "val", "test"])].astype(int).tolist()
            if sorted(active) != cohort.subject_ids:
                raise ValueError(f"Feature active IDs differ from cohort: {path}")
            test = ids[splits == "test"].astype(int).tolist()
            check_partition(cohort, ids[splits == "train"], ids[splits == "val"], test)
            seen.extend(test)
    if len(seen) != len(set(seen)) or set(seen) != set(cohort.subject_ids):
        raise ValueError("Probe requires exactly one feature file per active test ID")
    return cohort


def analysis_environment(root):
    env = dict(os.environ, PYTHONPATH=str(Path(root) / "src"), MPLBACKEND="Agg",
               PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def selected_fold_tsne(result_root, experiment, data_dir, cohort_file, metrics, output_dir,
                       device_name="cpu", samples=50, seed=42):
    """Fit new coordinates for data-selected folds; never replay legacy geometry."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.manifold import TSNE
    from .data import choose_eval_op_positions
    from .evaluation import (build_model, infer_embeddings, load_checkpoint, load_raw_dtof,
                             resolve_device, subject_position_indices)
    from .tsne_plotting import plot_fold_tsne
    cohort = load_cohort(cohort_file, require_confirmed=True)
    device = resolve_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, selected in metrics.drop_duplicates("fold").iterrows():
        sid = int(selected["fold"])
        path = Path(result_root) / str(sid) / experiment / "best_model.pt"
        validate_checkpoints([(sid, path)], cohort_file, complete=False)
        checkpoint = load_checkpoint(path)
        raw = load_raw_dtof(Path(data_dir))
        try:
            op_count = raw.shape[0] // cohort.raw_num_subjects
            positions = choose_eval_op_positions(op_count, samples, seed + 77)
            signature = {"checkpoint_sha256": file_sha256(path), "cohort": cohort.fingerprint,
                         "positions": positions.tolist(), "seed": seed, "device": str(device)}
            csv_path = output_dir / f"global_tsne_fold_{sid}.csv"
            audit_path = csv_path.with_suffix(".json")
            if csv_path.exists() and audit_path.exists() and json.loads(audit_path.read_text()) == signature:
                coordinates = pd.read_csv(csv_path)
            else:
                model = build_model(checkpoint, device)
                indices = subject_position_indices(cohort.subject_ids, positions, op_count)
                z = infer_embeddings(model, raw, indices, checkpoint, device, 512, False)
                xy = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=40,
                          random_state=seed, max_iter=1000).fit_transform(z)
                ids = np.repeat(cohort.subject_ids, len(positions))
                train, val = cohort.split(sid)
                split = np.where(ids == sid, "test", np.where(np.isin(ids, val), "val", "train"))
                coordinates = pd.DataFrame({"tsne_1": xy[:, 0], "tsne_2": xy[:, 1],
                                            "subject_id": ids, "split": split})
                coordinates.to_csv(csv_path, index=False)
                audit_path.write_text(json.dumps(signature, indent=2))
                del model
            display_ids = {int(subject): index + 1 for index, subject in enumerate(sorted(cohort.subject_ids))}
            fig, _ = plot_fold_tsne(coordinates, selected, subject_id_labels=display_ids)
            fig.savefig(output_dir / f"global_tsne_fold_{sid}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        finally:
            raw.close()
