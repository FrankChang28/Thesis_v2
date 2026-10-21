#!/usr/bin/env python3
"""Freeze a confirmed subject list and prepare isolated experiment inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subject_nirs.common.cohort import load_cohort
from subject_nirs.common.config import load_yaml
import yaml


def write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Prepared file changed; refusing to overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def align_metadata(root: Path, run: Path) -> Path:
    """Align a derived feature table to simulation order, preserving raw files."""
    source = root / "data/temp_metadata_v2.csv"
    names_file = root / "data/subject_names.txt"
    if not names_file.exists():
        return source
    names = [re.sub(r"^(?:N_)?Y\d+_[FM]_", "", line.strip(), flags=re.I).removesuffix("_EEG")
             for line in names_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    lookup = {row["subject"].strip().casefold(): (index, row) for index, row in enumerate(rows)}
    keys = [name.casefold() for name in names]
    if (len(set(keys)) != len(names) or len(lookup) != len(rows)
            or len(names) != len(rows) or set(keys) != set(lookup)):
        raise ValueError("Simulation subject names and metadata must match uniquely")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    reordered = []
    for subject_id, key in enumerate(keys):
        index, row = lookup[key]
        aligned = dict(row)
        aligned["sbj"] = subject_id
        writer.writerow(aligned)
        if index != subject_id:
            reordered.append({"subject_id": subject_id, "source_metadata_row": index})
    destination = run / "metadata_aligned.csv"
    write_once(destination, output.getvalue())
    audit = {"source_metadata": str(source), "subject_names_file": str(names_file),
             "metadata_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "subject_names_sha256": hashlib.sha256(names_file.read_bytes()).hexdigest(),
             "reordered_rows": reordered, "raw_num_subjects": len(names)}
    write_once(run / "metadata_alignment.json", json.dumps(audit, indent=2) + "\n")
    return destination


def cohort_notebook(notebook: dict, root: Path, cohort, frozen: Path) -> dict:
    """Retarget source cells, drop saved output, and retain original-ID semantics."""
    notebook = json.loads(json.dumps(notebook))
    fingerprint = cohort.fingerprint
    if notebook.get("metadata", {}).get("subject_cohort_analysis_version") == 2:
        return freeze_stage1_notebook(notebook, root, frozen, fingerprint)
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "code":
            source = re.sub(r"ROOT = Path.cwd\(\)(?:\.resolve\(\))?\n.*?os.chdir\(ROOT\)",
                            f"ROOT = Path({str(root)!r})\nos.chdir(ROOT)", source, flags=re.S)
            if "os.chdir(ROOT)" in source:
                source = source.replace("os.chdir(ROOT)", "os.chdir(ROOT)\n"
                    "import sys\nfrom pathlib import Path\n"
                    "sys.path.insert(0, str(ROOT / 'src'))\n"
                    "from subject_nirs.common.cohort import load_cohort\n"
                    f"COHORT = load_cohort({str(frozen)!r}, require_confirmed=True, "
                    f"expected_fingerprint={fingerprint!r})\n"
                    "ACTIVE_SUBJECT_IDS = COHORT.subject_ids\n")
            for prefix in ("artifacts", "cache", "results"):
                source = source.replace(prefix + "/", prefix + "/" + fingerprint + "/")
            aligned_metadata = frozen.parent / "metadata_aligned.csv"
            if aligned_metadata.exists():
                source = source.replace("data/raw/stage1/temp_metadata_v2.csv", str(aligned_metadata)).replace("data/temp_metadata_v2.csv", str(aligned_metadata))
            source = source.replace("'results' /", f"'results/{fingerprint}' /")
            source = source.replace("EXPECTED_SUBJECTS = 154", "EXPECTED_SUBJECTS = len(ACTIVE_SUBJECT_IDS)")
            source = source.replace("len(df) != 154", "set(df['fold'].astype(int)) != set(ACTIVE_SUBJECT_IDS)")
            source = source.replace("set(range(EXPECTED_SUBJECTS))", "set(ACTIVE_SUBJECT_IDS)")
            source = source.replace("np.arange(NUM_SUBJECTS, dtype=np.int64) * OP_PER_SUBJECT",
                                    "np.asarray(ACTIVE_SUBJECT_IDS, dtype=np.int64) * OP_PER_SUBJECT")
            source = source.replace("{NUM_SUBJECTS} subjects at fixed OP", "{len(ACTIVE_SUBJECT_IDS)} subjects at fixed OP")
            if "if len(thickness_metadata) != EXPECTED_SUBJECTS:" in source:
                source = source.replace("if len(thickness_metadata) != EXPECTED_SUBJECTS:",
                    "thickness_metadata = thickness_metadata.loc["
                    "thickness_metadata['subject_id'].isin(ACTIVE_SUBJECT_IDS)].copy()\n"
                    "if len(thickness_metadata) != EXPECTED_SUBJECTS:")
            if "FIG30_INPUT_DIR" in source:
                source = "# Historical Fig. 30 replay is intentionally replaced by this cohort's fresh probe.\n"
            if "from subject_nirs.stage1.tsne_plotting import" in source:
                source = fresh_tsne_cell()
            if "subject_nirs.stage2.unseen_scattering'" in source:
                source = source.replace("'subject_nirs.stage2.unseen_scattering',",
                    f"'subject_nirs.stage2.unseen_scattering', '--cohort_file', {str(frozen)!r},")
            source = source.replace("Expected 154 LOSO folds", "Expected all active cohort LOSO folds")
            source = source.replace("requires 154 aligned subjects", "requires all active cohort subjects")
            source = source.replace("all 154 DTOF descriptor subjects", "all active DTOF descriptor subjects")
            cell["outputs"] = []
            cell["execution_count"] = None
        cell["source"] = source.splitlines(keepends=True)
    notebook["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": [
        f"# Subject cohort {fingerprint}\n",
        f"本實驗納入 {len(cohort.subject_ids)} 位，保留原始 ID；所有舊輸出已清除，須以新模型重新執行。\n"]})
    return notebook


def freeze_stage1_notebook(nb, root, frozen, fingerprint):
    nb = json.loads(json.dumps(nb))
    for c in nb['cells']:
        s = ''.join(c['source'])
        if c['cell_type'] == 'code':
            s = s.replace("ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').is_file())", f'ROOT = Path({str(root)!r})')
            s = s.replace("RUN_DIR = ROOT\n", f"RUN_DIR = ROOT / 'runs' / {fingerprint!r}\n")
            s = s.replace("ROOT / 'data/metadata_aligned.csv'", "RUN_DIR / 'metadata_aligned.csv'")
            for prefix in ('artifacts', 'cache', 'results'):
                s = s.replace(f"ROOT / '{prefix}' /", f"ROOT / '{prefix}' / {fingerprint!r} /")
            s = s.replace("ROOT = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]\n            if (p / 'pyproject.toml').is_file() and (p / 'src/subject_nirs').is_dir())", f'ROOT = Path({str(root)!r})')
            s = s.replace("COHORT_FILE = ROOT / 'configs/subject_cohort.json'", f'COHORT_FILE = Path({str(frozen)!r})')
            s = s.replace('load_cohort(COHORT_FILE, require_confirmed=True)', f'load_cohort(COHORT_FILE, require_confirmed=True, expected_fingerprint={fingerprint!r})')
            c.update(outputs=[], execution_count=None)
        c['source'] = s.splitlines(keepends=True)
    return nb



def fresh_tsne_cell() -> str:
    return '''# Recompute the examples from the new cohort's checkpoints.
import argparse
import torch
from subject_nirs.stage1.evaluation import load_checkpoint, build_model
from subject_nirs.stage1.data import load_data_bundle, choose_eval_op_positions
from subject_nirs.stage1.metrics import collect_embeddings
from subject_nirs.stage1.visualization import plot_global_tsne
for sid in sorted({int(representative['fold']), int(most_dispersed['fold'])}):
    checkpoint = load_checkpoint(RESULT_ROOT / str(sid) / EXPERIMENT / 'best_model.pt')
    train_args = argparse.Namespace(**checkpoint['args'])
    bundle = load_data_bundle(train_args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(checkpoint, device)
    positions = choose_eval_op_positions(bundle.op_per_subject, 50, train_args.seed + 77)
    payloads = {name: collect_embeddings(model, bundle.make_aligned_loader(
        name, positions, 512), device, use_amp=train_args.amp)
        for name in ('train', 'val', 'test')}
    destination = FIGURE_DIR / f'tsne_fold_{sid}'
    destination.mkdir(parents=True, exist_ok=True)
    plot_global_tsne(payloads, destination, samples_per_subject=50, seed=train_args.seed)
    del model, bundle, payloads
'''


def prepare(root: Path, cohort_file: Path, *, preview: bool = False) -> Path:
    root = root.resolve()
    cohort = load_cohort(cohort_file, require_confirmed=not preview)
    run = root / "runs" / cohort.fingerprint
    print(json.dumps({**cohort.metadata(), "run_dir": str(run),
                      "train_per_fold": len(cohort.subject_ids) - cohort.val_subject_count - 1,
                      "confirmed": cohort.confirmed}, indent=2))
    if preview:
        return run
    frozen = run / "cohort.json"
    frozen_values = {"raw_num_subjects": cohort.raw_num_subjects, "id_base": 0,
                     "remove_subject_ids": list(cohort.excluded_subject_ids),
                     "val_subject_count": cohort.val_subject_count,
                     "split_seed": cohort.split_seed, "confirmed": True}
    write_once(frozen, json.dumps(frozen_values, indent=2) + "\n")
    metadata_path = align_metadata(root, run)
    splits = []
    for sid in cohort.subject_ids:
        train, val = cohort.split(sid)
        splits.append({"test_id": sid, "stage1_fold": sid, "stage2_fold": sid + 1,
                       "train_subjects": train, "val_subjects": val, "test_subjects": [sid]})
    write_once(run / "split_manifest.json", json.dumps(splits, indent=2) + "\n")
    artifact = root / "artifacts" / cohort.fingerprint
    cache = root / "cache" / cohort.fingerprint
    common = {"num_subjects": cohort.raw_num_subjects, "cohort_file": str(frozen),
              "cohort_fingerprint": cohort.fingerprint}
    for source in sorted((root / "configs").rglob("*.yaml")):
        values = load_yaml(source)
        if source.name.startswith("stage1_"):
            values.update(common, data_dir=str(root / "data/raw/stage1"), split_seed=cohort.split_seed)
        else:
            values = {key: (value.replace("./artifacts/", str(artifact) + "/")
                            if isinstance(value, str) else value)
                      for key, value in values.items()}
            values.update(common, mat_file=str(root / "data/raw/stage2/NIRS_Absolute_Dataset_grid.mat"),
                          output_root=str(artifact / "stage2"), cache_root=str(cache / "stage2"),
                          val_subject_count=cohort.val_subject_count,
                          split_random_state_base=cohort.split_seed,
                          metadata_feature_path=str(metadata_path),
                          test_subject_start=None, test_subject_end=None, run_all_loso=True,
                          structure_feature_path=str(artifact / "stage1/{test_id}/relat_cons_32/subject_features_all_op.npz"))
        write_once(run / "configs" / source.relative_to(root / "configs"), yaml.safe_dump(values, sort_keys=False))
    for source in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = cohort_notebook(json.loads(source.read_text(encoding="utf-8")), root, cohort, frozen)
        target = run / "notebooks" / source.name
        if not target.exists():
            write_once(target, json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        if source.name.startswith("07_"):
            second = json.loads(json.dumps(notebook))
            for cell in second["cells"]:
                cell["source"] = [line.replace("TARGET = 'hc'", "TARGET = 'sto2'") for line in cell["source"]]
            target = run / "notebooks/07_unseen_dataset_validation_sto2.ipynb"
            if not target.exists():
                write_once(target, json.dumps(second, ensure_ascii=False, indent=1) + "\n")
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=ROOT / "configs/subject_cohort.json")
    parser.add_argument("--preview", action="store_true", help="Validate and print only; no files or training")
    args = parser.parse_args()
    prepare(ROOT, args.cohort, preview=args.preview)


if __name__ == "__main__":
    main()
