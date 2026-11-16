"""Write measured CSF and skull thickness as a 2-D subject descriptor, one NPZ per LOSO fold.

Stage 2 reads these files through the DTOF descriptor path (subject_feature_source: dtof,
structure_feature_dim: 2). The metadata path is not used because it always reads the four
columns scalp, skull, CSF and sinus. Each fold is standardised with its own training subjects
inside Stage 2, exactly as for the DTOF descriptors.

Columns of latent_raw: [csf, skull] from data/metadata_aligned.csv, unchanged values.
Output: <output-dir>/<test_id>/descriptor.npz. Run from the repository root.

    python scripts/build_thickness_descriptor.py
    bash scripts/run_stage2_loso.sh configs/stage2_csf_skull/concatenation/finetune/hc.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subject_nirs.common.cohort import load_cohort  # noqa: E402

COLUMNS = ("csf", "skull")


def main(output_dir: Path, metadata_path: Path, cohort_path: Path) -> None:
    cohort = load_cohort(cohort_path, require_confirmed=True)
    meta = pd.read_csv(metadata_path, encoding="utf-8-sig")
    if len(meta) != cohort.raw_num_subjects:
        raise ValueError(f"{metadata_path}: {len(meta)} rows, expected {cohort.raw_num_subjects}")
    values = np.stack([meta[c].to_numpy(float) for c in COLUMNS], axis=1).astype(np.float32)
    ids = [int(s) for s in cohort.subject_ids]
    if not np.isfinite(values[ids]).all():
        raise ValueError("CSF or skull thickness has NaN or Inf for a cohort subject")

    for sid in ids:
        train, val = cohort.split(sid)
        folder = output_dir / str(sid)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "descriptor.npz", latent_raw=values,
                 cohort_fingerprint=cohort.fingerprint, test_subject_ids=np.array([sid]),
                 train_subject_ids=np.asarray(train), val_subject_ids=np.asarray(val))
    print(f"wrote {len(ids)} folds, columns {list(COLUMNS)}, cohort {cohort.fingerprint} -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/thickness_descriptor/csf_skull")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/metadata_aligned.csv")
    parser.add_argument("--cohort", type=Path, default=ROOT / "configs/subject_cohort.json")
    args = parser.parse_args()
    main(args.output_dir, args.metadata, args.cohort)
