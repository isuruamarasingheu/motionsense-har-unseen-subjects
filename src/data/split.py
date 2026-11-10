"""
Module: src.data.split
Purpose: Subject-wise, grouped dataset splitting without data leakage.

Responsibilities:
- Partition the 24 subjects into train (16 subjects), validation (4 subjects),
  and test (4 subjects) sets using the seed defined in configs/data.yaml.
- Ensure gender balance across splits:
    * Validation: 2 male + 2 female (4 subjects)
    * Test:       2 male + 2 female (4 subjects)
    * Train:      10 male + 6 female (16 subjects)
- Persist the split definition to `outputs/splits/split.json` with metadata.
- Verify via strict assertions that train, val, and test subject sets are disjoint
  (intersection is empty), preventing identity leakage.

Design constraints:
- Data splitting is BY SUBJECT ONLY. Never split windows randomly.
  A subject's data must appear in exactly one of train / val / test.
- Splitting happens BEFORE windowing.
- Seed is loaded from config for exact reproducibility.
- Docstrings explain why subject-wise splitting is essential.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import load_config, save_json

logger = logging.getLogger(__name__)


def create_subject_split(
    subjects_info_path: str | Path = "MotionSense_dataset/data_subjects_info.csv",
    seed: int = 42,
    val_male: int = 2,
    val_female: int = 2,
    test_male: int = 2,
    test_female: int = 2,
    output_path: str | Path = "outputs/splits/split.json",
) -> dict[str, Any]:
    """
    Partition subjects into train, val, and test sets with stratified gender balance.

    Why subject-wise splitting:
        Sensor recordings contain distinctive biometric signatures (gait dynamics,
        stride frequency, posture tilt). If overlapping windows from the same subject
        were randomly assigned to both training and test sets, the model could easily
        memorise participant identities rather than learning generalisable activity patterns,
        yielding artificially inflated performance that collapses on unseen users.

    Why gender balance:
        Biomechanical differences (height, mass distribution, cadence) vary by sex.
        Stratifying validation and test sets to contain an equal number of male and female
        participants (2M + 2F) ensures evaluation is unbiased and representative.

    Args:
        subjects_info_path: Path to `data_subjects_info.csv`.
        seed: Random seed for deterministic subject shuffling.
        val_male: Number of male subjects in validation set.
        val_female: Number of female subjects in validation set.
        test_male: Number of male subjects in test set.
        test_female: Number of female subjects in test set.
        output_path: Destination path for `split.json`.

    Returns:
        Dictionary containing split lists, seed, and method description.
    """
    info_path = Path(subjects_info_path)
    if not info_path.is_file():
        raise FileNotFoundError(f"Demographics file not found: {info_path.resolve()}")

    df_subjects = pd.read_csv(info_path).rename(columns={"code": "subject"})

    # Validate demographic expectations: MotionSense has 24 subjects (14 male: 1, 10 female: 0)
    all_subjects = sorted(df_subjects["subject"].unique().tolist())
    male_subjects = sorted(df_subjects[df_subjects["gender"] == 1]["subject"].tolist())
    female_subjects = sorted(df_subjects[df_subjects["gender"] == 0]["subject"].tolist())

    if len(all_subjects) != 24:
        raise ValueError(f"Expected 24 subjects, found {len(all_subjects)}")
    if len(male_subjects) != 14 or len(female_subjects) != 10:
        raise ValueError(
            f"Expected 14 male and 10 female subjects, found {len(male_subjects)}M / {len(female_subjects)}F"
        )

    # Deterministic permutation using RandomState with seed
    rng = np.random.RandomState(seed)
    male_perm = rng.permutation(male_subjects).tolist()
    female_perm = rng.permutation(female_subjects).tolist()

    # Assign validation subjects (2M + 2F)
    val_m = male_perm[:val_male]
    val_f = female_perm[:val_female]
    val_subjects = sorted(val_m + val_f)

    # Assign test subjects (2M + 2F)
    test_m = male_perm[val_male : val_male + test_male]
    test_f = female_perm[val_female : val_female + test_female]
    test_subjects = sorted(test_m + test_f)

    # Assign remaining subjects to train (10M + 6F = 16 subjects)
    train_m = male_perm[val_male + test_male :]
    train_f = female_perm[val_female + test_female :]
    train_subjects = sorted(train_m + train_f)

    split_dict: dict[str, Any] = {
        "train": [int(s) for s in train_subjects],
        "val": [int(s) for s in val_subjects],
        "test": [int(s) for s in test_subjects],
        "seed": int(seed),
        "method": (
            f"gender-balanced subject-wise split ({len(train_subjects)} train / "
            f"{len(val_subjects)} val / {len(test_subjects)} test; "
            f"val: {val_male}M+{val_female}F, test: {test_male}M+{test_female}F)"
        ),
        "demographics": {
            "train": {"male": len(train_m), "female": len(train_f), "total": len(train_subjects)},
            "val": {"male": len(val_m), "female": len(val_f), "total": len(val_subjects)},
            "test": {"male": len(test_m), "female": len(test_f), "total": len(test_subjects)},
        },
    }

    # Verify split safety and integrity
    assert_disjoint_splits(split_dict, all_subjects)

    # Save to outputs/splits/split.json
    out_file = Path(output_path)
    save_json(split_dict, out_file)
    logger.info("Subject split saved to %s", out_file)

    return split_dict


def assert_disjoint_splits(split_dict: dict[str, Any], all_subjects: list[int] | None = None) -> None:
    """
    Assert that train, val, and test splits have zero subject overlap.

    Why:
        A subject appearing in multiple splits causes
        identity leakage, which invalidates the coursework results.

    Args:
        split_dict: Dictionary containing 'train', 'val', and 'test' subject lists.
        all_subjects: Optional list of all expected subject IDs.
    """
    train_set = set(split_dict["train"])
    val_set = set(split_dict["val"])
    test_set = set(split_dict["test"])

    # Disjointness checks
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    three_way_overlap = train_set & val_set & test_set

    assert len(train_val_overlap) == 0, f"Subject overlap between train and val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Subject overlap between train and test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Subject overlap between val and test: {val_test_overlap}"
    assert len(three_way_overlap) == 0, f"Subject overlap train ∩ val ∩ test: {three_way_overlap}"

    # Partition completeness check
    combined = train_set | val_set | test_set
    if all_subjects is not None:
        expected_set = set(all_subjects)
        assert combined == expected_set, f"Split subjects {combined} do not match all subjects {expected_set}"

    # Verify target partition sizes (16 / 4 / 4)
    assert len(train_set) == 16, f"Expected 16 train subjects, got {len(train_set)}"
    assert len(val_set) == 4, f"Expected 4 val subjects, got {len(val_set)}"
    assert len(test_set) == 4, f"Expected 4 test subjects, got {len(test_set)}"


def load_subject_split(split_path: str | Path = "outputs/splits/split.json") -> dict[str, Any]:
    """
    Load an existing subject split from JSON and assert validity.

    Args:
        split_path: Path to split.json.

    Returns:
        Split dictionary with train, val, and test subject lists.
    """
    from src.utils import load_json

    data = load_json(split_path)
    assert_disjoint_splits(data)
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config("configs/data.yaml")
    split_cfg = cfg.get("split", {})

    split = create_subject_split(
        subjects_info_path=cfg.get("dataset", {}).get("subjects_info", "MotionSense_dataset/data_subjects_info.csv"),
        seed=split_cfg.get("seed", 42),
        val_male=split_cfg.get("val_composition", {}).get("male", 2),
        val_female=split_cfg.get("val_composition", {}).get("female", 2),
        test_male=split_cfg.get("test_composition", {}).get("male", 2),
        test_female=split_cfg.get("test_composition", {}).get("female", 2),
        output_path="outputs/splits/split.json",
    )

    print("\n--- Subject Split Generated Successfully ---")
    print(f"Seed:   {split['seed']}")
    print(f"Train ({len(split['train'])} subjects): {split['train']}")
    print(f"Val   ({len(split['val'])} subjects): {split['val']}")
    print(f"Test  ({len(split['test'])} subjects): {split['test']}")
    print("\nDemographics:")
    for k, v in split["demographics"].items():
        print(f"  {k:5s}: {v['male']}M + {v['female']}F = {v['total']} total")
