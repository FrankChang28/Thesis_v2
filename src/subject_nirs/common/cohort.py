"""Original subject IDs, inclusion rules, and shared LOSO partitions."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cohort:
    raw_num_subjects: int
    excluded_subject_ids: tuple[int, ...] = ()
    val_subject_count: int = 10
    split_seed: int = 100
    confirmed: bool = True

    @property
    def subject_ids(self) -> list[int]:
        excluded = set(self.excluded_subject_ids)
        return [sid for sid in range(self.raw_num_subjects) if sid not in excluded]

    @property
    def fingerprint(self) -> str:
        identity = [self.raw_num_subjects, self.excluded_subject_ids,
                    self.val_subject_count, self.split_seed]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]

    def split(self, test_id: int) -> tuple[list[int], list[int]]:
        from sklearn.model_selection import train_test_split
        if test_id not in self.subject_ids:
            raise ValueError(f"Excluded or unknown test subject: {test_id}")
        train, val = train_test_split(
            [sid for sid in self.subject_ids if sid != test_id],
            test_size=self.val_subject_count,
            random_state=self.split_seed + test_id, shuffle=True,
        )
        return sorted(train), sorted(val)

    def metadata(self) -> dict:
        return {"fingerprint": self.fingerprint,
                "raw_num_subjects": self.raw_num_subjects,
                "active_subject_ids": self.subject_ids,
                "excluded_subject_ids": list(self.excluded_subject_ids),
                "val_subject_count": self.val_subject_count,
                "split_seed": self.split_seed}


def load_cohort(path: str | Path | None, raw_num_subjects: int = 154,
                val_subject_count: int = 10, split_seed: int = 100,
                *, require_confirmed: bool = False,
                expected_fingerprint: str | None = None) -> Cohort:
    if path is None:
        return Cohort(raw_num_subjects, (), val_subject_count, split_seed)
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_count = values["raw_num_subjects"]
    base = values.get("id_base", 0)
    removed = values["remove_subject_ids"]
    if type(raw_count) is not int or raw_count != raw_num_subjects:
        raise ValueError("raw_num_subjects must match the unchanged source layout")
    if type(base) is not int or base not in (0, 1):
        raise ValueError("id_base must be 0 or 1")
    if not isinstance(removed, list) or any(type(sid) is not int for sid in removed):
        raise ValueError("remove_subject_ids must be a list of integer IDs")
    removed = [sid - base for sid in removed]
    if len(set(removed)) != len(removed):
        raise ValueError("Duplicate removal IDs")
    if any(sid < 0 or sid >= raw_count for sid in removed):
        raise ValueError("Removal ID outside the original subject range")
    val_count = values.get("val_subject_count", 10)
    seed = values.get("split_seed", 100)
    if type(val_count) is not int or type(seed) is not int:
        raise ValueError("val_subject_count and split_seed must be integers")
    if val_count != val_subject_count or seed != split_seed:
        raise ValueError("Cohort and experiment split settings differ")
    if val_count < 2 or raw_count - len(removed) <= val_count + 1:
        raise ValueError("Cohort must leave validation and at least two training subjects")
    confirmed = values.get("confirmed", False)
    if type(confirmed) is not bool:
        raise ValueError("confirmed must be a JSON boolean")
    cohort = Cohort(raw_count, tuple(sorted(removed)), val_count, seed, confirmed)
    if require_confirmed and not cohort.confirmed:
        raise ValueError("Subject removal list is unconfirmed; set confirmed=true after review")
    if expected_fingerprint and cohort.fingerprint != expected_fingerprint:
        raise ValueError("Cohort changed after preparation; prepare a new run")
    return cohort


def cohort_from_args(args, *, require_confirmed: bool = False) -> Cohort:
    return load_cohort(getattr(args, "cohort_file", None), int(args.num_subjects),
                       int(getattr(args, "val_subject_count", 10)),
                       int(getattr(args, "split_seed", 100)),
                       require_confirmed=require_confirmed,
                       expected_fingerprint=getattr(args, "cohort_fingerprint", None))


def check_partition(cohort: Cohort, train, val, test) -> None:
    test = list(map(int, test))
    if len(test) != 1:
        raise ValueError("Expected exactly one held-out subject")
    expected_train, expected_val = cohort.split(test[0])
    if sorted(map(int, train)) != expected_train or sorted(map(int, val)) != expected_val:
        raise ValueError("Checkpoint subject partition differs from this cohort/split")


def check_output_namespace(path: str | Path, cohort: Cohort) -> None:
    resolved = Path(path).resolve()
    for directory in (resolved, *resolved.parents):
        marker = directory / ".cohort.json"
        if marker.is_file():
            if json.loads(marker.read_text()).get("fingerprint") != cohort.fingerprint:
                raise ValueError("Output root belongs to another cohort")
            return
    if cohort.fingerprint not in resolved.parts:
        raise ValueError("Output root requires a matching .cohort.json marker")


def check_existing_run(directory: str | Path, cohort: Cohort) -> None:
    directory = Path(directory)
    metadata = directory / "experiment_config.json"
    if metadata.exists():
        if json.loads(metadata.read_text()).get("cohort_fingerprint") != cohort.fingerprint:
            raise ValueError("Existing experiment metadata belongs to another cohort")
    elif (directory / "loso_results.csv").exists():
        raise ValueError("Existing results have no cohort metadata; refusing to reuse them")
