"""Memory-efficient sampled training-cache construction."""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, Sequence

import h5py
import numpy as np

from .cache import (
    preprocess_drs_x,
    read_subject_selected_rows,
    read_x_rows,
    read_y_rows,
    select_target_columns,
    target_cache_tag,
)
from .config import CFG
from .sampling import ranges_to_batches, sample_train_ranges_for_subject
from .utils import count_rows, ensure_dir, write_json


def compact_stage_seed(stage_index: int) -> int:
    return int(CFG.subset_seed + stage_index * CFG.compact_stage_seed_stride)


def subject_compact_paths(
    subject_id: int,
    stage_index: int,
    n_rows: int,
) -> Dict[str, str]:
    seed = compact_stage_seed(stage_index) + int(subject_id)
    tag = (
        f"subject_{subject_id:03d}_stage_{stage_index + 1:02d}_"
        f"target_{target_cache_tag()}_"
        f"rows_{CFG.max_train_rows_per_subject}_"
        f"win_{CFG.sim_windows_per_sim}_seed_{seed}_n_{n_rows}"
    )
    directory = os.path.join(CFG.compact_cache_dir, "per_subject", tag)
    return {
        "dir": directory,
        "x": os.path.join(directory, "train_x.npy"),
        "y": os.path.join(directory, "train_y.npy"),
        "subj": os.path.join(directory, "train_subj.npy"),
        "meta": os.path.join(directory, "metadata.json"),
    }


def _cache_is_complete(paths: Dict[str, str]) -> bool:
    return all(os.path.exists(paths[key]) for key in ("x", "y", "subj", "meta"))


def build_or_load_subject_compact_cache_from_mat(
    file: h5py.File | Callable[[], h5py.File],
    subject_id: int,
    stage_index: int,
) -> dict:
    """Load cached rows first; resolve a lazy MAT source only for a rebuild."""
    seed = compact_stage_seed(stage_index) + int(subject_id)
    ranges = sample_train_ranges_for_subject(subject_id, seed)
    ranges = sorted((int(start), int(end)) for start, end in ranges if end > start)
    n_rows = count_rows(ranges)

    paths = subject_compact_paths(subject_id, stage_index, n_rows)
    ensure_dir(paths["dir"])

    info = {
        "subject_id": int(subject_id),
        "stage_idx": int(stage_index),
        "n_rows": int(n_rows),
        **paths,
    }
    if _cache_is_complete(paths) and not CFG.rebuild_compact_cache:
        return info

    if callable(file):
        try:
            file = file()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Compact training cache is incomplete or a rebuild was requested: "
                f"subject={subject_id}, stage={stage_index + 1}, "
                f"directory={paths['dir']}. Copy the complete cache or provide "
                f"the original MAT file: {CFG.mat_file}"
            ) from exc

    print(
        "Building subject compact cache: "
        f"subject={subject_id}, "
        f"stage={stage_index + 1}/{CFG.compact_stage_count}, "
        f"rows={n_rows}"
    )
    print(f"  {paths['dir']}")

    x_out = np.lib.format.open_memmap(
        paths["x"],
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, CFG.input_dim),
    )
    y_out = np.lib.format.open_memmap(
        paths["y"],
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, CFG.cache_target_dim),
    )
    subject_out = np.lib.format.open_memmap(
        paths["subj"],
        mode="w+",
        dtype=np.int64,
        shape=(n_rows,),
    )

    write_position = 0
    started = time.time()
    for rows in ranges_to_batches(
        ranges,
        CFG.compact_cache_chunk_rows,
        shuffle=False,
        drop_last=False,
    ):
        batch_rows = len(rows)
        target_slice = slice(write_position, write_position + batch_rows)
        x_out[target_slice] = preprocess_drs_x(
            read_x_rows(file["Train_Dataset_X"], rows)
        )
        y_out[target_slice] = read_y_rows(file["Train_Dataset_Y"], rows)
        subject_out[target_slice] = read_subject_selected_rows(
            file["Train_Dataset_SubjID"],
            rows,
        )
        write_position += batch_rows

    if write_position != n_rows:
        raise RuntimeError(
            f"Compact cache row mismatch: expected {n_rows}, wrote {write_position}"
        )

    x_out.flush()
    y_out.flush()
    subject_out.flush()

    elapsed = time.time() - started
    write_json(
        paths["meta"],
        {
            "subject_id": int(subject_id),
            "original_matlab_subj_id": int(subject_id) + 1,
            "stage_idx": int(stage_index),
            "n_rows": int(n_rows),
            "cache_target_names": list(CFG.cache_target_names),
            "max_train_rows_per_subject": CFG.max_train_rows_per_subject,
            "sim_windows_per_sim": CFG.sim_windows_per_sim,
            "seed": seed,
            "input_dim": CFG.input_dim,
            "copy_seconds": elapsed,
        },
    )
    print(f"Subject compact cache finished in {elapsed:.1f}s")
    return info


def compute_stats_from_compact(
    infos: Sequence[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one immutable, verified training-fold normalization shared by models."""
    from .normalization import load_or_fit
    return load_or_fit(infos, CFG)
