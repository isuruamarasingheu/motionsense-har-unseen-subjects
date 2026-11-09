"""
Module: src.data.split
Purpose: Subject-wise, grouped dataset splitting without data leakage.

Responsibilities:
- Partition the 24 subjects into train (16 subjects), validation (4 subjects),
  and test (4 subjects) sets using the seed defined in configs/data.yaml.
- Ensure gender balance across splits (e.g., validation: 2M + 2F; test: 2M + 2F;
  training: remaining 10M + 6F).
- Persist the split definition to `outputs/splits/split.json` with metadata.
- Verify via strict assertions that train, val, and test subject sets are disjoint
  (intersection is empty), preventing identity leakage.

Rule compliance:
- Rule 4: Data splitting is BY SUBJECT ONLY. Never split windows randomly.
  A subject's data must appear in exactly one of train / val / test.
"""

