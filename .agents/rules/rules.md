PROJECT RULES — follow in every task:

1. Language: Python 3.11. Frameworks: NumPy, pandas, scikit-learn, PyTorch (CPU), matplotlib.
2. Raw data is READ-ONLY. Never modify or copy files inside MotionSense_dataset/.
3. Ignore any __MACOSX folders. Drop the unnamed index column in each CSV.
4. Data splitting is BY SUBJECT ONLY. Never split windows randomly. A subject's data must
   appear in exactly one of train / val / test. Windows must never cross trial boundaries.
5. Splitting happens BEFORE windowing. Any scaler/normaliser is fitted on TRAIN data only,
   then saved and reused for val/test/inference.
6. Data augmentation (if any) is applied only to training batches.
7. The test set is used ONCE, only by src/evaluate.py, only after configs are frozen.
   Never use test results to choose models, seeds, epochs or hyperparameters.
8. Every run reads a config file from configs/ and writes: checkpoint, config copy,
   metrics JSON, training log, run_info.json (seed, versions, CPU, timestamp).
9. Set all seeds (python random, numpy, torch) from the config.
10. Never invent, estimate or hard-code results. All numbers in reports come from saved
    metrics files. Keep failed/unsuccessful runs in the logs.
11. Separate code into: preparation (src/data/), models (src/models/), training
    (src/train.py), evaluation (src/evaluate.py), inference (src/predict.py).
12. Every plot has a title, labelled axes with units, and a legend where needed.
    Save figures as PNG (dpi 150) in outputs/figures/ with a numbered name, e.g. fig01_class_distribution.png.
13. Add docstrings and comments explaining WHY, not just what. Mark any borrowed code
    with a comment giving its source.
14. Keep code simple and readable — this is coursework that the student must explain.
15. After finishing a task, print a short summary of files created and key numbers.