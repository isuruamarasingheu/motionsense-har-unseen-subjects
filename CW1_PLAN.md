# CW1 Implementation Plan — Human Activity Recognition with MotionSense

**Module:** NB6057CEM Artificial Neural Networks — CW1 Practical (100 marks)
**Dataset:** MotionSense (A_DeviceMotion_data) — Malekzadeh et al., https://github.com/mmalekzadeh/motion-sense
**Coding tool:** Google Antigravity (agent) — **Supervisor:** Claude (review after every phase)

---

## 0. How to use this file

1. Put this file in the project root (`C:\Users\Isuru\Desktop\ANN\`).
2. Give Antigravity **Section 2 (Agent Rules)** first, either as a workspace rule or pasted at the start of every task.
3. Work **one phase at a time**. Each phase has a **Prompt for Antigravity** block. Copy it into the agent.
4. When a phase is finished, run its **Acceptance checks** yourself.
5. Then send the **Supervision checkpoint** items to Claude for review **before** starting the next phase.
6. Tick the boxes in this file as you go.

> You must understand and be able to explain every line in the demo (T6b). After each phase, ask the agent to explain anything you don't understand.
> Declare your use of generative AI tools according to your institution's policy.

---

## 1. Project summary (for T1a and the README)

| Item | Decision |
|---|---|
| Intended user | Developers of a fitness/health-monitoring phone app that logs daily activity |
| Prediction target | Activity class (6): downstairs `dws`, upstairs `ups`, walking `wlk`, jogging `jog`, sitting `sit`, standing `std` |
| Unit of observation | One window of 128 samples (2.56 s at 50 Hz) × 12 sensor channels |
| Practical value | Automatic activity logging without manual input; basis for step/exercise tracking |
| Data | 24 subjects (14 male, 10 female), iPhone 6s in front trouser pocket, 15 trials, 12 channels: attitude (roll, pitch, yaw), gravity (x, y, z), rotationRate (x, y, z), userAcceleration (x, y, z) |
| Licence | **Check and quote the licence on the GitHub repo yourself**. Cite the dataset papers (Malekzadeh et al., 2018/2019). |
| Ethics | Motion data can reveal gender, weight, identity (the dataset was built to study this). Small, non-diverse sample (age 18–46). Only one device/placement. |
| Primary metric | **Macro-F1** (classes are imbalanced, all activities matter equally) |
| Complementary metrics | Accuracy, per-class precision/recall/F1, confusion matrix |

### Trial → label map

| Label | Code | Trials (folder suffix) |
|---|---|---|
| 0 | dws | 1, 2, 11 |
| 1 | ups | 3, 4, 12 |
| 2 | wlk | 7, 8, 15 |
| 3 | jog | 9, 16 |
| 4 | std | 6, 14 |
| 5 | sit | 5, 13 |

### Model line-up

| Role | Model | Input | Task |
|---|---|---|---|
| From-scratch | NumPy 1-hidden-layer MLP | Hand-crafted window features (subset) | T2 |
| Baseline (non-neural) | Random Forest | Hand-crafted window features | T3a |
| Neural model A | PyTorch MLP (2+ hidden layers) | Hand-crafted window features | T3b |
| Neural model B | PyTorch 1D-CNN | Raw window 12 × 128 | T3b |

The MLP (fixed feature vector, no notion of time order) and the 1D-CNN (learns local temporal filters from raw sequences) are **meaningfully different architectures**.

---

## 2. Agent Rules (give these to Antigravity — non-negotiable)

```
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
```

---

## 3. Project structure

```
ANN/
├── CW1_PLAN.md                  ← this file
├── README.md
├── requirements.txt             ← pinned versions
├── .gitignore                   ← excludes raw data, __pycache__
├── MotionSense_dataset/         ← raw data (NOT included in submission ZIP)
│   ├── data_subjects_info.csv
│   └── A_DeviceMotion_data/A_DeviceMotion_data/<act>_<trial>/sub_<n>.csv
├── configs/
│   ├── data.yaml                ← paths, window size, stride, split seed
│   ├── tuning_budget.yaml       ← declared BEFORE tuning
│   ├── numpy_mlp.yaml
│   ├── baseline_rf.yaml
│   ├── mlp.yaml
│   └── cnn.yaml
├── src/
│   ├── data/
│   │   ├── load.py              ← read CSVs → one DataFrame with subject, activity, trial
│   │   ├── split.py             ← subject-wise split, saves outputs/splits/split.json
│   │   ├── windows.py           ← windowing per trial
│   │   ├── features.py          ← hand-crafted features per window
│   │   └── preprocess.py        ← fit/save/load scalers
│   ├── models/
│   │   ├── numpy_mlp.py         ← T2 from-scratch MLP
│   │   ├── mlp.py               ← PyTorch MLP
│   │   └── cnn.py               ← PyTorch 1D-CNN
│   ├── utils.py                 ← seeding, run_info, JSON helpers
│   ├── train.py                 ← trains any framework model from a config
│   ├── train_baseline.py        ← Random Forest
│   ├── tune.py                  ← runs the declared tuning budget on VAL only
│   ├── evaluate.py              ← final test evaluation (run once)
│   ├── experiments.py           ← ablation, noise stress test, subgroup test
│   ├── benchmark.py             ← parameter counts + inference timing
│   └── predict.py               ← CLI inference
├── notebooks/
│   ├── 01_eda.ipynb             ← T1b data audit + plots
│   └── 02_numpy_mlp.ipynb       ← T2 demonstration
├── outputs/
│   ├── splits/                  ← split.json, window index files
│   ├── preprocessing/           ← scaler files
│   ├── checkpoints/             ← <model>_seed<k>.pt / .pkl + config copy
│   ├── metrics/                 ← *.json, results_table.csv
│   ├── figures/
│   └── logs/
├── samples/                     ← example inputs for predict.py (valid + invalid)
└── report/
    └── technical_record.md      ← ≤1500 words, T1–T6
```

---

## 4. Fixed design decisions (keep the same across all models — fair comparison)

| Setting | Value | Reason |
|---|---|---|
| Window length | 128 samples (2.56 s) | Standard in HAR literature; covers ≥2 gait cycles |
| Stride | 64 (50% overlap) for all splits | More training windows; overlap stays inside one split so no leakage |
| Split | 16 train / 4 val / 4 test subjects, gender-balanced (val 2M+2F, test 2M+2F), seed 42 | Grouped split prevents identity leakage |
| Feature set (RF, MLP, NumPy MLP) | 12 channels + `|userAcc|` + `|rotRate|` = 14 signals × 6 stats (mean, std, min, max, median, IQR) = **84 features** | Same information for baseline and MLP |
| Raw input (CNN) | 12 × 128, standardised per channel with train mean/std | |
| Loss | Cross-entropy (softmax output) | Multiclass |
| Optimiser | Adam, lr 1e-3, weight decay 1e-4 (tuned within budget) | |
| Batch size | 64 | |
| Max epochs / early stopping | 60 / patience 8 on **val macro-F1**, keep best checkpoint | |
| Seeds | 0, 1, 2 | Required 3 seeds |
| Class imbalance | Report it; try class-weighted loss **within** the tuning budget | |

---

## 5. Phases

### Phase 0 — Environment setup  ☑

**Prompt for Antigravity:**
```
Set up the project skeleton described in CW1_PLAN.md Section 3 (folders and empty
module files with docstrings). Create requirements.txt with pinned versions for:
numpy, pandas, scikit-learn, matplotlib, seaborn, pyyaml, joblib, jupyter, torch (CPU).
Install PyTorch CPU with:
  pip install torch --index-url https://download.pytorch.org/whl/cpu
Create src/utils.py with: set_seed(seed), save_json, load_json, load_config(yaml path),
and write_run_info(path) that records Python, numpy, torch, sklearn versions, OS,
CPU name and timestamp. Create .gitignore that excludes MotionSense_dataset/,
__pycache__/, .ipynb_checkpoints/. Do not write any modelling code yet.
```

**Acceptance checks**
- [x] `python -c "import torch; print(torch.__version__)"` works
- [x] `pip freeze` matches `requirements.txt` versions
- [x] Folders exist; raw data untouched

**Supervision checkpoint → send Claude:** `requirements.txt`, folder tree.

---

### Phase 1 — T1 Problem definition & data preparation (8 marks)  ☑

#### 1a. Loading + audit (T1a, T1b)

**Prompt for Antigravity:**
```
Implement src/data/load.py:
- Walk MotionSense_dataset/A_DeviceMotion_data/A_DeviceMotion_data/, skip __MACOSX.
- Folder name "<act>_<trial>" gives activity code and trial number; file "sub_<n>.csv" gives subject id.
- Drop the unnamed index column. Add columns: subject, activity, label (map in CW1_PLAN.md Section 1), trial.
- Merge subject info from data_subjects_info.csv (code→subject: weight, height, age, gender).
- Return one DataFrame; cache it to outputs/cache/motionsense.parquet (or pickle).

Then create notebooks/01_eda.ipynb that performs a data audit and prints/saves:
1. Total rows, rows per activity, rows per subject, rows per trial.
2. Missing values per column; number of exact duplicate rows; any subject missing any trial.
3. Basic stats per channel (min/max/mean/std) and any extreme outliers.
4. Subject demographics table (gender counts, age/weight/height range).
Figures (save to outputs/figures/):
 fig01_class_distribution.png  — bar chart of rows (and minutes) per activity
 fig02_signals_per_activity.png — 6 small panels: userAcceleration x/y/z over 5 s for one subject per activity
 fig03_rows_per_subject.png    — stacked bars per subject by activity
 fig04_channel_boxplots.png    — userAcceleration magnitude by activity (boxplot)
Write a markdown cell under each figure: what it shows and how it affects design
(e.g. imbalance → macro-F1; sit vs std look similar → expect confusion).
```

#### 1c. Leakage-safe pipeline (T1c — 4 marks, very important)

**Prompt for Antigravity:**
```
Implement:
- src/data/split.py: subject-wise split 16/4/4 using seed from configs/data.yaml,
  gender-balanced: val = 2 male + 2 female, test = 2 male + 2 female, rest train.
  Save outputs/splits/split.json {train: [...], val: [...], test: [...], seed, method}.
  Assert no subject appears in more than one split.
- src/data/windows.py: create windows of length 128, stride 64, INSIDE each
  (subject, trial) recording only — never crossing trials. Label = trial activity.
  Return X_raw (N,128,12), y (N,), meta (subject, trial, start_index).
  Save window meta per split to outputs/splits/windows_<split>.csv.
- src/data/features.py: 84 features per window (14 signals × mean,std,min,max,median,IQR;
  signals = 12 channels + magnitude of userAcceleration + magnitude of rotationRate).
- src/data/preprocess.py: StandardScaler for features and per-channel mean/std for raw
  windows, FITTED ON TRAIN ONLY. Save to outputs/preprocessing/ (joblib + json).
- A script `python -m src.data.prepare` that runs all of this and prints window counts
  per split and per class, and saves arrays to outputs/cache/.
Add assertions/tests:
  * subjects in train ∩ val ∩ test = empty
  * scaler mean equals mean of TRAIN features only
  * no window spans two trials
```

**Acceptance checks**
- [x] `split.json` exists; no subject overlap (assertion passes)
- [x] Window counts printed per split and per class (expect roughly 20k+ windows total)
- [x] Scaler fitted only on train
- [x] 4 figures saved with labelled axes

**Supervision checkpoint → send Claude:** audit printout (missing/duplicates/imbalance numbers), `split.json`, window counts table, the 4 figures.

**Write in technical record (T1):** user/target/unit, licence + citation, ethics, audit findings, split justification (why grouped by subject — overlapping windows from the same person would leak identity and inflate scores), preprocessing fitted on train only.

---

### Phase 2 — T2 NumPy MLP from first principles (12 marks)  ☑

**Prompt for Antigravity:**
```
Implement src/models/numpy_mlp.py using ONLY NumPy (no autograd, no torch), float64:
- Class NumpyMLP(n_in=84, n_hidden=32, n_out=6, seed):
  W1 (84,32), b1 (32,), W2 (32,6), b2 (6,) — Xavier/Glorot init for tanh.
  forward(X): Z1 = X@W1 + b1 ; H = tanh(Z1) ; Z2 = H@W2 + b2 ; P = softmax(Z2) (numerically stable).
  loss(P, y): mean cross-entropy.
  backward(): analytical gradients, vectorised:
    dZ2 = (P - Y_onehot)/N ; dW2 = H.T@dZ2 ; db2 = sum(dZ2) ;
    dH = dZ2@W2.T ; dZ1 = dH*(1-H**2) ; dW1 = X.T@dZ1 ; db1 = sum(dZ1)
  step(lr): mini-batch gradient descent update.
  Document array shapes in docstrings.
- Training loop: mini-batches (size 64), lr 0.05, max 200 epochs, early stop when val
  loss has not improved for 15 epochs. Use a SUBSET: up to 3000 train windows
  (stratified sample), full val set. Inputs = scaled 84 features.
  Record initial loss, final loss, actual epochs completed.
- Gradient check function: on a batch of 8 samples, central differences with eps=1e-5
  for at least 2 entries each of W1, b1, W2, b2 (8 entries total). Report analytical,
  numerical, absolute diff, relative diff = |a-n| / max(|a|+|n|, 1e-12). Tolerance 1e-6.
  (tanh is smooth, so there are no ReLU kink points.)
- Controlled check: overfit a tiny batch of 32 samples (no early stopping, 2000 epochs)
  and show loss approaching 0 / accuracy 100%.
Create notebooks/02_numpy_mlp.ipynb that runs all of this and saves:
 fig05_numpy_mlp_loss.png (train & val loss vs epoch)
 fig06_numpy_mlp_overfit.png (tiny-batch loss vs epoch, log y-axis)
 outputs/metrics/numpy_mlp.json (initial/final loss, epochs, val macro-F1, grad-check table)
Include a markdown cell on the biological neuron analogy (dendrites≈weighted inputs,
soma≈summation+bias, firing≈activation) and its limits.
If training does not converge, show it and diagnose (lr too high/low, scaling) — do not hide it.
```

**Acceptance checks**
- [x] All grad-check relative errors < 1e-6 (8 entries, all 4 parameter arrays)
- [x] Loss curve decreases; initial loss ≈ ln(6) ≈ 1.79 (sanity check for 6 classes)
- [x] Tiny batch overfits to near-zero loss
- [x] Learning rate, stopping rule, epochs completed reported

**Supervision checkpoint → send Claude:** grad-check table, initial/final loss, both figures, val macro-F1.

---

### Phase 3 — T3 Baseline and framework models (14 marks)  ☑

#### 3a. Declare tuning budget FIRST

**Prompt for Antigravity:**
```
Create configs/tuning_budget.yaml BEFORE any tuning, declaring:
- Random Forest: 6 trials over n_estimators {100, 300} × max_depth {None, 20, 10}
- MLP: 6 trials over hidden sizes {[128,64], [256,128]} × dropout {0.2, 0.4} × lr {1e-3} + class_weight {false,true} on best → keep total = 6
- CNN: 6 trials over filters {[32,64,64], [64,128,128]} × dropout {0.3, 0.5} × kernel {5, 7} (choose 6 combos, list them explicitly)
Selection rule: highest mean VAL macro-F1, using seed 0 only for tuning.
Commit this file before running anything.
```

#### 3b. Models + training

**Prompt for Antigravity:**
```
Implement:
1. src/train_baseline.py — RandomForestClassifier on the 84 scaled features
   (class_weight="balanced"). Evaluate VAL macro-F1. Save model with joblib and its config.
2. src/models/mlp.py — PyTorch MLP: Linear(84→h1) → BatchNorm → ReLU → Dropout →
   Linear(h1→h2) → BatchNorm → ReLU → Dropout → Linear(h2→6). Kaiming (He) init for ReLU layers.
3. src/models/cnn.py — PyTorch 1D-CNN, input (batch, 12, 128):
   [Conv1d(k) → BatchNorm1d → ReLU → MaxPool1d(2)] × 3 blocks → GlobalAveragePooling →
   Dropout → Linear(→6). Kaiming init.
4. src/train.py — generic trainer: `python -m src.train --config configs/cnn.yaml --seed 0`
   - DataLoader (batch 64, shuffle train only), Adam (lr, weight_decay from config),
     CrossEntropyLoss (optional class weights computed from TRAIN labels),
   - each epoch log train loss, val loss, val macro-F1, val accuracy,
   - early stopping on val macro-F1 (patience 8), save BEST checkpoint to
     outputs/checkpoints/<name>_seed<k>.pt plus config copy and run_info.json,
   - save history CSV and loss curve figure.
   - `--smoke-test` flag: 2 train subjects, 1 val subject, 1 epoch — must run in < 1 minute.
5. src/tune.py — runs EXACTLY the trials in configs/tuning_budget.yaml on VAL only,
   writes outputs/metrics/tuning_log.csv (all trials, including bad ones), and writes
   the selected final configs to configs/mlp_final.yaml and configs/cnn_final.yaml.
6. A function that prints a layer table (layer, output shape, activation, trainable params)
   using torch — save to outputs/metrics/<model>_layers.md. Total params must match
   sum(p.numel() for p in model.parameters() if p.requires_grad).
Figures: fig08_mlp_loss_curves.png, fig09_cnn_loss_curves.png (train & val loss vs epoch,
marking the selected epoch).
Do NOT load or touch the test split in any of these scripts.
```

**Acceptance checks**
- [x] `tuning_budget.yaml` created before `tuning_log.csv` (check file timestamps)
- [x] `tuning_log.csv` has exactly the declared trials
- [x] Smoke test runs: `python -m src.train --config configs/cnn.yaml --smoke-test`
- [x] Layer tables with parameter counts for MLP and CNN
- [x] Loss curves for both, best epoch marked
- [x] grep confirms no script except `evaluate.py` reads the test split

**Supervision checkpoint → send Claude:** `tuning_log.csv`, final configs, both layer tables, both loss curves, RF val macro-F1.

---

### Phase 4 — T4 Experiments & evaluation (14 marks)  ☐

> ⚠️ **Freeze point:** before this phase, `mlp_final.yaml` and `cnn_final.yaml` are final. Do not change them after you see test results.

#### 4a. Repeated runs + test evaluation

**Prompt for Antigravity:**
```
1. Train mlp_final and cnn_final with seeds 0, 1, 2 (same split). Also confirm RF
   (RF is deterministic given random_state; run seeds 0,1,2 too for comparison).
2. Implement src/evaluate.py — the ONLY script that loads the test split:
   - loads each saved checkpoint + saved scaler,
   - computes per seed: test macro-F1, accuracy, per-class precision/recall/F1,
   - writes outputs/metrics/results_table.csv with columns:
     model, config_file, seed, val_macro_f1, test_macro_f1, test_accuracy, n_params, checkpoint_path
   - writes summary with mean ± std per model,
   - confusion matrices (normalised by row) for each model (seed 0 or the median seed —
     choose by VAL score, never test): fig09_cm_rf.png, fig10_cm_mlp.png, fig11_cm_cnn.png
   - per-class F1 bar chart comparing the 3 models: fig12_per_class_f1.png
```

#### 4c. Ablation (validation only)

**Prompt for Antigravity:**
```
In src/experiments.py add `ablation`: take cnn_final.yaml and change ONE factor only:
dropout = 0 (everything else identical). Train with seeds 0,1,2. Compare VAL macro-F1
and val loss curves vs the original CNN (same seeds). Save
outputs/metrics/ablation.json and fig13_ablation.png (bar with error bars + curves).
```

#### 4d. Errors & robustness

**Prompt for Antigravity:**
```
In src/experiments.py add:
1. `errors`: for the selected CNN (seed chosen on VAL), list misclassified test windows;
   pick at least 5 representative ones (different true→predicted pairs, include sit↔std
   and ups↔dws if present). For each save a plot of the 12 channels with true label,
   predicted label, confidence, subject id: fig14_error_examples.png. Save list to errors.csv.
2. `noise_stress`: PROTOCOL (write it in the output JSON before running):
   add Gaussian noise N(0, σ²) to standardised test inputs, σ ∈ {0, 0.1, 0.25, 0.5, 1.0},
   5 repeats per σ, for RF, MLP and CNN (all 3 seeds). Note: for RF/MLP, noise is added to
   raw windows THEN features are recomputed, so all models see the same corrupted signals.
   Report macro-F1 mean±std vs σ: fig15_noise_robustness.png.
3. `subgroup`: test macro-F1 separately for male vs female test subjects, with number of
   windows and number of subjects in each group. Save subgroup.json.
```

**Acceptance checks**
- [ ] `results_table.csv` has 9 rows (3 models × 3 seeds) + mean ± std
- [ ] Confusion matrices and per-class F1 saved, axes labelled
- [ ] Ablation compared on **validation**, same seeds
- [ ] ≥5 errors inspected; noise protocol written before results; sample sizes reported

**Supervision checkpoint → send Claude:** `results_table.csv`, summary, confusion matrices, ablation, noise and subgroup results, error examples.

---

### Phase 5 — T5 Recommendation & inference (8 marks)  ☑

**Prompt for Antigravity:**
```
1. src/benchmark.py: for RF, MLP, CNN report trainable params (RF: number of nodes),
   model file size, and inference time on CPU for batch size 1 and 256
   (warm-up 10 runs, then mean ± std over 100 runs, torch.set_num_threads fixed and
   recorded). Save outputs/metrics/benchmark.json and a combined table with mean test
   macro-F1 ± std.
2. src/predict.py CLI:
   python -m src.predict --model cnn --input samples/walking_example.csv
   - loads saved checkpoint + saved scaler (no refitting),
   - input = CSV with the 12 MotionSense columns (any order, exact names), ≥128 rows,
   - windows it (128/64), predicts each window, returns majority activity +
     mean probability per class as a readable table, e.g.
       Predicted activity: WALKING (confidence 0.91)
   - validation with clear error messages and non-zero exit code for:
     missing/wrong columns, fewer than 128 rows, NaN/inf values, non-numeric data,
     values far outside training range (flag as out-of-scope warning).
3. Create samples/: one valid example cut from a TEST subject recording
   (state which subject/trial), plus samples/invalid_short.csv and samples/invalid_columns.csv.
```

**Acceptance checks**
- [x] Valid sample → sensible prediction; both invalid samples → clean error message (no traceback)
- [x] Timing table states batch size, thread count, CPU model

**Supervision checkpoint → send Claude:** benchmark table, predict.py output for all 3 samples. We will write the recommendation (T5a) and limitations (T5c) together from the evidence.

---

### Phase 6 — T6 Reproducibility & demo (4 marks)  ☐

**Prompt for Antigravity:**
```
Write README.md with:
- Project overview (1 paragraph), dataset citation + download instructions
  (clone https://github.com/mmalekzadeh/motion-sense, unzip A_DeviceMotion_data.zip and
  data_subjects_info.csv into MotionSense_dataset/). State the raw data is not redistributed.
- Setup: python version, `pip install -r requirements.txt`, PyTorch CPU command.
- Run order with exact commands:
    python -m src.data.prepare
    python -m src.train --config configs/cnn.yaml --smoke-test
    python -m src.train_baseline --config configs/baseline_rf.yaml
    python -m src.tune
    python -m src.train --config configs/mlp_final.yaml --seed {0,1,2}
    python -m src.train --config configs/cnn_final.yaml --seed {0,1,2}
    python -m src.evaluate
    python -m src.experiments --all
    python -m src.benchmark
    python -m src.predict --model cnn --input samples/walking_example.csv
- Where splits, seeds, configs, checkpoints, metrics live.
- Expected runtime on CPU.
- Borrowed code / AI-assistance declaration section.
```

**Demo plan (5–7 min, you present it):**
1. (0:30) Problem + dataset + why subject-wise split
2. (1:30) Run `predict.py` on valid sample and invalid sample
3. (2:00) Trace one result: `configs/cnn_final.yaml` → `outputs/checkpoints/cnn_seed0.pt` → row in `results_table.csv` → confusion matrix
4. (1:00) Design choice explained (e.g. CNN on raw windows vs MLP on features; or early stopping on val macro-F1)
5. (1:00) Observed failure (e.g. sit vs std confusion; why — similar posture, orientation-dependent)

**Acceptance checks**
- [ ] Fresh venv + README commands reproduce `results_table.csv`
- [ ] Smoke test under 1 minute
- [ ] Demo rehearsed and timed

---

### Phase 7 — Technical record (≤1500 words, excluding tables/captions)  ☐

Write this **yourself** (Claude can review). Structure:

| Section | Content | ~Words |
|---|---|---|
| T1 | User, target, unit, dataset/licence/ethics, audit findings (fig01–04), split & leakage control | 300 |
| T2 | NumPy MLP shapes, neuron analogy, loss curve, grad-check table, overfit check, convergence notes | 250 |
| T3 | Baseline, MLP & CNN layer tables, design choices, training control, tuning budget & log | 300 |
| T4 | Results table (model/config/seed/val/test), metrics justification, confusion matrices, ablation, errors, noise & subgroup | 350 |
| T5 | Recommendation (quality vs variance vs params vs latency), inference demo, limitations, next experiment | 200 |
| T6 | Reproducibility summary, demo link | 50 |
| Appendix | AI/borrowed-code declaration, unsuccessful experiments | — |

---

## 6. Submission ZIP checklist

- [ ] `src/`, `notebooks/`, `configs/`
- [ ] `README.md`, `requirements.txt`
- [ ] `outputs/splits/` (split.json, window indices)
- [ ] `outputs/preprocessing/` (scalers)
- [ ] `outputs/checkpoints/` (all final models, 3 seeds each)
- [ ] `outputs/metrics/` (results_table.csv, tuning_log.csv, ablation, noise, subgroup, benchmark JSON)
- [ ] `outputs/figures/` (fig01–fig15)
- [ ] `samples/`
- [ ] `report/technical_record` (PDF or MD, ≤1500 words)
- [ ] Demo video file or link
- [ ] **NOT included:** `MotionSense_dataset/` raw data

---

## 7. Common pitfalls (Claude will check these at each review)

| Pitfall | Why it costs marks | Prevention |
|---|---|---|
| Random window split | Identity leakage, inflated ~98% scores (T1c) | Subject split before windowing |
| Scaler fitted on all data | Preprocessing leakage | Fit on train, save, reuse |
| Picking best seed / epoch on test | Violates fair comparison rule | Select on val; test once |
| Tuning budget defined after tuning | Unfair comparison | Commit `tuning_budget.yaml` first |
| Second model is just wider MLP | Doesn't count as a 2nd architecture | MLP vs 1D-CNN |
| Grad check on ReLU / only W | Loses T2c marks | tanh hidden, entries from W1, b1, W2, b2 |
| Accuracy only | Imbalanced classes hide failures | Macro-F1 + per-class metrics |
| Ablation changes 2 things | Not a controlled test | Change exactly one factor |
| Hidden failed runs | Brief requires keeping them | Log every trial |
| Claiming "deployment ready" | Explicitly penalised (T5c) | State limits honestly |

---

## 8. Progress log

| Phase | Status | Date | Reviewed by Claude | Notes |
|---|---|---|---|---|
| 0 Setup | ☑ | 2026-11-09 | ☐ | Skeleton & pinned env ready |
| 1 T1 Data | ☐ | | ☐ | |
| 2 T2 NumPy MLP | ☐ | | ☐ | |
| 3 T3 Models | ☐ | | ☐ | |
| 4 T4 Experiments | ☐ | | ☐ | |
| 5 T5 Inference | ☐ | | ☐ | |
| 6 T6 Repro/Demo | ☐ | | ☐ | |
| 7 Tech record | ☐ | | ☐ | |
