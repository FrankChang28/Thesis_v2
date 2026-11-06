"""Verified, immutable training-fold normalization shared by all Stage 2 models."""
from __future__ import annotations

import fcntl
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

SCHEMA = "verified-stage2-normalization-v1"
KEYS = ("x_min", "x_max", "y_mean", "y_std")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def info_identity(info):
    return dict(n_rows=int(info["n_rows"]),
                paths={k: str(Path(info[k]).resolve()) for k in ("x", "y", "subj")})


def source_hashes(info):
    return {k: sha256(info[k]) for k in ("x", "y", "subj")}


def _scan_once(info, input_dim, chunk_rows):
    """Check float64 conversion and reductions against original float32 extrema."""
    x = np.load(info["x"], mmap_mode="r")
    y = np.load(info["y"], mmap_mode="r")
    subjects = np.load(info["subj"], mmap_mode="r")
    n = int(info["n_rows"])
    if n < 1 or x.shape[0] < n or y.shape[0] < n or subjects.shape[0] < n:
        raise ValueError("Incomplete training cache")
    if x.ndim != 2 or x.shape[1] != input_dim or y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("Unexpected training cache shape")
    if "subject_id" in info and not np.all(subjects[:n] == info["subject_id"]):
        raise ValueError("Training cache subject ID mismatch")
    lo, hi = np.full(input_dim, np.inf), np.full(input_dim, -np.inf)
    sums, squares = np.zeros(2), np.zeros(2)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        raw = x[start:stop]
        values = np.asarray(raw, dtype=np.float64)
        targets = np.asarray(y[start:stop], dtype=np.float64)
        if not np.isfinite(values).all() or not np.isfinite(targets).all():
            raise ValueError("Non-finite training cache values")
        if not np.array_equal(values.astype(raw.dtype), raw):
            raise ValueError("Training float conversion changed source values")
        low, high = values.min(axis=0), values.max(axis=0)
        if not np.array_equal(low, raw.min(axis=0)) or not np.array_equal(high, raw.max(axis=0)):
            raise ValueError("Training extrema disagree across float32/float64 paths")
        lo, hi = np.minimum(lo, low), np.maximum(hi, high)
        # Match the original single-target summation order for each target.
        for column in range(2):
            selected = targets[:, column:column + 1]
            sums[column] += selected.sum(axis=0)[0]
            squares[column] += np.square(selected).sum(axis=0)[0]
    return dict(n=n, x_min=lo.tolist(), x_max=hi.tolist(),
                y_sum=sums.tolist(), y_sum_squared=squares.tolist())


def scan_verified(info, input_dim=30, chunk_rows=1048576):
    before = source_hashes(info)
    first = _scan_once(info, input_dim, chunk_rows)
    second = _scan_once(info, input_dim, chunk_rows)
    if first != second:
        raise ValueError("Repeated training statistics disagree; refusing to persist normalization")
    if before != source_hashes(info):
        raise ValueError("Training cache changed during normalization")
    return dict(identity=info_identity(info), hashes=before, moments=first)


def combine(summaries, target_index):
    if not summaries or target_index not in (0, 1):
        raise ValueError("Expected training summaries and one valid target")
    lo, hi = None, None
    total = 0
    sums = np.zeros(1, dtype=np.float64)
    squares = np.zeros(1, dtype=np.float64)
    for summary in summaries:
        m = summary["moments"]
        lower, upper = np.asarray(m["x_min"]), np.asarray(m["x_max"])
        lo = lower.copy() if lo is None else np.minimum(lo, lower)
        hi = upper.copy() if hi is None else np.maximum(hi, upper)
        sums += m["y_sum"][target_index]
        squares += m["y_sum_squared"][target_index]
        total += m["n"]
    mean = sums / total
    std = np.sqrt(np.maximum(squares / total - np.square(mean), 1e-12))
    values = tuple(a.astype(np.float32) for a in (lo, hi, mean, std))
    validate_values(values)
    return values


def validate_values(values):
    if len(values) != 4 or any(not np.isfinite(a).all() for a in values):
        raise ValueError("Invalid normalization values")
    if values[0].shape != values[1].shape or values[2].shape != (1,) or values[3].shape != (1,):
        raise ValueError("Invalid normalization dimensions")
    if np.any(values[0] > values[1]) or np.any(values[3] <= 0):
        raise ValueError("Invalid normalization range or standard deviation")


def assert_matching(actual, expected, context="checkpoint"):
    """Require matching statistics, allowing bounded float32 input-range rounding."""
    for key, wanted in zip(KEYS, expected):
        if key in actual:
            observed = np.asarray(actual[key], dtype=np.float32)
            wanted = np.asarray(wanted, dtype=np.float32)
            compatible = (observed.shape == wanted.shape
                          and np.isfinite(observed).all() and np.isfinite(wanted).all())
            if compatible and np.array_equal(observed, wanted):
                continue
            if compatible and key in ("x_min", "x_max"):
                # Bound differences by both float32 spacing and channel range.
                spacing = np.maximum(np.abs(np.spacing(observed)), np.abs(np.spacing(wanted)))
                span = (np.asarray(expected[1], dtype=np.float64)
                        - np.asarray(expected[0], dtype=np.float64))
                tolerance = np.minimum(4.0 * spacing.astype(np.float64), 1e-6 * span)
                difference = np.abs(observed.astype(np.float64) - wanted.astype(np.float64))
                if np.all(difference <= tolerance):
                    warnings.warn(
                        f"{context}: {key} has float32 rounding differences "
                        f"(max_abs={difference.max():.9g}); using verified training normalization",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
        raise ValueError(f"{context}: {key} differs from verified training normalization; retraining required")


def record_path(infos, cfg):
    identity = dict(schema=SCHEMA, training=[info_identity(i) for i in infos],
                    input_dim=cfg.input_dim, target_index=cfg.target_index,
                    cohort=cfg.cohort_fingerprint, chunk_rows=cfg.cache_chunk_rows)
    return Path(cfg.cache_dir) / "verified_normalization_v1" / (digest(identity) + ".json"), identity


def load_or_fit(infos, cfg):
    """Verify source bytes every call; fit and publish one record under a file lock."""
    path, identity = record_path(infos, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hashes = [source_hashes(i) for i in infos]
        if path.exists():
            record = json.loads(path.read_text())
            checksum = record.pop("sha256")
            if digest(record) != checksum or record["identity"] != identity:
                raise ValueError(f"Normalization record is corrupt or foreign: {path}")
            if record["source_hashes"] != hashes:
                raise ValueError(f"Training cache changed since normalization was frozen: {path}")
            values = tuple(np.asarray(record["values"][k], dtype=np.float32) for k in KEYS)
            validate_values(values)
        else:
            summaries = [scan_verified(i, cfg.input_dim, cfg.cache_chunk_rows) for i in infos]
            if hashes != [s["hashes"] for s in summaries]:
                raise ValueError("Training cache changed before statistics were fitted")
            values = combine(summaries, cfg.target_index)
            record = dict(identity=identity, source_hashes=hashes,
                          values={k: v.tolist() for k, v in zip(KEYS, values)})
            record["sha256"] = digest(record)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
        for value in values:
            value.setflags(write=False)
        return values
