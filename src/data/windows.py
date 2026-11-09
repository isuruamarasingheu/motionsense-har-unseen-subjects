"""
Module: src.data.windows
Purpose: Trial-bounded fixed-length sliding window segmenter.

Responsibilities:
- Segment continuous sensor streams into windows of length 128 samples (2.56 s at 50 Hz)
  with a stride of 64 samples (50% overlap).
- Constrain window generation strictly WITHIN each individual (subject, trial) recording;
  windows must never cross trial boundaries or subject boundaries.
- Generate raw 3D tensor arrays X_raw of shape (N, 128, 12), activity targets y of shape (N,),
  and per-window tracking metadata (subject, trial, start_index).
- Save window metadata records per split to `outputs/splits/windows_<split>.csv`.

Rule compliance:
- Rule 4: Windows must never cross trial boundaries.
- Rule 5: Splitting happens BEFORE windowing.
"""

