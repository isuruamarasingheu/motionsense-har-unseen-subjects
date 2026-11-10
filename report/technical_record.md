# Technical Record — Human Activity Recognition on Unseen People (MotionSense)

**Module:** NB6057CEM Artificial Neural Networks — CW1 Practical
**Student ID:** KIC-BSCCSAI-251P-042
**Repository:** https://github.com/isuruamarasingheu/motionsense-har-unseen-subjects
**Word count (excluding tables and captions):** ≈1,490 [recount after your edits]

---

## T1 Problem definition and data preparation

**a. Problem and provenance.** Intended user: a developer of an activity-tracking phone app. The model looks at a **2.56-second window** of phone motion data (128 readings at 50 Hz, 12 sensor channels) and predicts one of six activities: downstairs, upstairs, walking, jogging, standing or sitting. Each window is one example; the key question is whether the model works on **people it has never seen**. The data is MotionSense (Malekzadeh et al., 2019), [ADD licence as stated on the repository]. After cleaning: 24 people, 359 recordings, 1,402,573 rows (7.79 hours), 21,380 windows. Ethics: walking style can identify people, and the dataset stores age, gender, height and weight.

**b. Data audit.** There were no missing values. Two of subject 1's recordings (`sit_5`, `sit_13`) were exact copies (10,292 rows each), so I removed `sit_13`. Classes are uneven: training has 3,488 sitting windows but only 1,226 downstairs (about 2.8 to 1; Fig. 1). Channels use very different scales (Fig. 4), so every input is standardised. Fig. 2 shows walking and stair signals look alike.

**c. Leakage-safe pipeline.** A random split would put the same person in both training and test data, so the model could recognise people, not activities, and look better than it is (identity leakage). So I split **by person** (seed 42): 16 training, 4 validation and 4 test (subjects 1, 8, 18, 21) (2 men, 2 women each). Windows never cross between recordings. Scalers are fitted on training data only, and no augmentation is used. Features come only from their own window, and person or trial numbers are never model inputs (no target leakage). Splits and scalers are saved; tests check no person is in two splits.

## T2 Learning from first principles (NumPy)

**a. Forward pass.** A hand-built network with 84 inputs, 32 hidden units and 6 outputs: `H = tanh(XW1 + b1)`, `P = softmax(HW2 + b2)`, loss = average cross-entropy. Shapes: X (B, 84), W1 (84, 32), b1 (32), W2 (32, 6), b2 (6), P (B, 6) (B = batch size). Like a brain cell, each hidden unit adds weighted inputs and applies a curve; unlike real neurons, it has no spikes and learns by sending errors backwards.

**b. Backpropagation.** `dZ2 = (P − Y)/B`, `dW2 = Hᵀ dZ2`, `dZ1 = dZ2 W2ᵀ ⊙ (1 − H²)`, `dW1 = Xᵀ dZ1`. Mini-batch gradient descent: learning rate 0.05, batch 64, Xavier starting weights, 3,000 training windows, full validation set. Loss fell from 1.877 (near ln 6 = 1.792, the random-guess value) to 0.112 (training) and 0.198 (validation); validation macro-F1 was 0.925 (Fig. 5).

**c. Gradient check.** I compared formula gradients with numerical estimates (central differences, ε = 1e-5) on a batch of 8, for 8 entries across W1, b1, W2 and b2. The largest relative difference was 1.2e-10, far below my 1e-6 tolerance. tanh is smooth, so there are no kinks.

**d. Numerical diagnosis.** Training stopped after 15 epochs without validation-loss improvement: best epoch 25, 40 epochs run. The network learned a tiny batch of 32 windows perfectly: loss 1.949 → 0.0036, 100% accuracy, 2,000 epochs (Fig. 6). So the updates work. A learning-rate test (Fig. 7) found a hidden problem: at learning rate 2.0, training loss reached 0.0004 but validation loss rose to 0.360 (vs 0.218 at 0.005), while macro-F1 stayed around 0.92. High learning rates made the model **too confident**, not more accurate.

## T3 Baseline and framework models

**a. Baseline.** A Random Forest (RF) on the same 84 features and splits, scored the same way.

**b. Two neural networks** (layer tables in Appendix A):
- **MLP on features**: 84 → 128 → 64 → 6, with BatchNorm, ReLU and dropout 0.2 (19,910 parameters). It cannot see the time order of readings.
- **1D-CNN on raw signals**: 12 channels × 128 time steps, three convolution blocks (64, 128, 128 filters, size 5) with BatchNorm, ReLU and pooling, then global average pooling, dropout 0.3 and an output layer (128,454 parameters). It learns time patterns directly.

These are different designs (hand-made features vs learned filters), not resized versions of one model. Both output six scores trained with cross-entropy. Starting weights: Kaiming-normal for the MLP (suits ReLU), PyTorch's default Kaiming-uniform for the CNN. Both use the Adam optimiser (learning rate 1e-3, weight decay 1e-4).

**c. Training control.** Batch 64, up to 60 epochs, weight decay and dropout against overfitting. Training stops when validation macro-F1 has not improved for 8 epochs, and the best weights are saved. Curves are in Figs. 8–9.

**d. Traceability and fair tuning.** Data preparation, training and evaluation are separate files. Each run saves its config, model and setup: Python 3.11.9, torch 2.15.0+cpu, scikit-learn 1.7.2, Intel i5-11400H, Windows. The tuning budget was fixed before searching (`tuning_budget.yaml`): 6 trials per model, seed 0, chosen by validation macro-F1 (Appendix B). No borrowed code or pretrained models. Compute was unequal (CNN trials 26–132 s, MLP 6–13 s), and the CNN uses raw signals, not features.

## T4 Experiments and evaluation

**a. Repeated evaluation.** Final settings were frozen, each model was trained with seeds 0, 1 and 2, and the test set was run once. Plots use the seed with the middle validation score, never the best test score.

**Table 1. Results (macro-F1; test = unseen subjects 1, 8, 18, 21).**

| Model | Config | Seed | Best epoch | Val macro-F1 | Test macro-F1 | Test acc. |
|---|---|---|---|---|---|---|
| RF | baseline_rf_final.yaml | 0 / 1 / 2 | – | 0.937 / 0.928 / 0.926 | 0.886 / 0.888 / 0.884 | 0.881 / 0.884 / 0.881 |
| MLP | mlp_final.yaml | 0 / 1 / 2 | 11 / 15 / 17 | 0.956 / 0.953 / 0.963 | 0.882 / 0.894 / 0.879 | 0.888 / 0.898 / 0.879 |
| CNN | cnn_final.yaml | 0 / 1 / 2 | 18 / 3 / 4 | 0.967 / 0.944 / 0.946 | 0.842 / 0.895 / 0.847 | 0.843 / 0.902 / 0.851 |
| **RF** mean ± std | | | | 0.930 ± 0.006 | **0.886 ± 0.002** | 0.882 ± 0.002 |
| **MLP** mean ± std | | | | 0.957 ± 0.005 | **0.885 ± 0.008** | 0.888 ± 0.009 |
| **CNN** mean ± std | | | | 0.953 ± 0.013 | **0.861 ± 0.029** | 0.865 ± 0.032 |

RF and MLP are effectively tied on test. CNN seed 0 had the best validation score (0.967) but the worst test score (0.842): with only 4 validation people, choosing by validation score is unreliable. Every model scored 0.04–0.09 lower on test than validation (Fig. 14).

**b. Metrics.** Macro-F1 is the main metric because it counts every activity equally, even rare ones. Accuracy and per-class precision, recall and F1 are also reported (Table 2, Fig. 13). The most common mistake (Figs. 10–12) is **walking predicted as upstairs** (RF 214, MLP 250, CNN 340 windows), lowering upstairs precision and walking recall. The RF also confuses sitting and standing (111 windows).

**Table 2. Per-class test precision / recall (representative seeds: RF 1, MLP 0, CNN 2).**

| Class | RF P / R | MLP P / R | CNN P / R |
|---|---|---|---|
| dws | 0.91 / 0.89 | 0.91 / 0.86 | 0.92 / 0.69 |
| ups | 0.65 / 0.96 | 0.60 / 0.93 | 0.50 / 0.91 |
| wlk | 0.94 / 0.73 | 0.92 / 0.65 | 0.87 / 0.59 |
| jog | 1.00 / 0.96 | 0.91 / 0.94 | 0.99 / 0.97 |
| std | 0.87 / 1.00 | 0.98 / 1.00 | 0.96 / 1.00 |
| sit | 1.00 / 0.86 | 1.00 / 1.00 | 1.00 / 0.98 |

**c. Ablation.** Claim tested: dropout 0.3 before the CNN's output layer improves validation macro-F1. I changed only dropout (to 0.0), with the same 3 seeds, on validation data. Result: 0.9525 ± 0.013 with dropout vs 0.9513 ± 0.021 without, a +0.001 difference, far below the seed-to-seed variation (Fig. 15). **The claim is not supported.** Confounders: early stopping chose very different epochs (3–18 with dropout), and three seeds are too few to judge stability.

**d. Errors and robustness.** I examined six typical errors (Fig. 16). Errors concentrate in a few recordings: in subject 1's walking trial 7, the CNN got 81 of 83 windows wrong. There the phone's median pitch was −0.86 rad, versus −1.28 rad in the same person's other walks, so the phone was tilted. This links errors to phone angle but does not prove the cause.

*Noise stress test (method fixed in advance; `noise_protocol.json`).* Random noise was added to test signals at five strengths (σ = 0–1.0 × each channel's training spread): 5 draws × 3 seeds × 3 models, 3,730 windows. I expected the CNN to suffer most. **I was wrong:** at σ = 0.5 the CNN kept 0.606 ± 0.011, while the MLP fell to 0.217 and the RF to 0.233 (Fig. 17). The MLP scored higher for women (0.946) than men (0.819), but with 2 people per group this reflects individuals, not gender.

## T5 Recommendation and use

**a. Recommendation.** Speed: i5-11400H CPU, one window at a time including preprocessing, mean of 100 runs after 10 warm-ups (RF 1 thread, PyTorch 6). I recommend the **MLP**. Its test macro-F1 (0.885 ± 0.008) matches the RF (0.886 ± 0.002), but it is about 280 times smaller (0.08 MB vs 22.37 MB) and 16 times faster (0.48 ms vs 7.56 ms per window). The simpler RF suits cases where consistency matters most (least seed variation). The CNN (0.50 MB, 0.44 ms) only makes sense for noisy sensors: it is less accurate on clean data and least stable (Fig. 19).

**b. Inference pipeline.** `python -m src.predict --model {rf,mlp,cnn} --input file.csv` loads the saved model and scaler and prints the activity with average confidence. A normal recording from an unseen person (subject 8, walking) returns WALKING (0.95). Files that are too short (60 rows), miss columns or contain empty values stop with exit code 1 and a clear message. A file with unusual value ranges only gets a **warning** and is still classified (JOGGING, 1.00).

**c. Limitations.** *Bias:* 24 people aged 18–46, one phone model, one carrying position, and only 4 test people. *Generalisation:* scores drop on new people and tilted phones, so no model is ready for real use. *Privacy and misuse:* motion data can reveal identity, health and routine, so predictions should run on the phone, with consent. *Next experiment:* train on randomly rotated signals and re-test on the tilted recordings.

## T6 Reproducibility and demonstration

The README covers setup, the pinned `requirements.txt`, how to download the data (not redistributed), the run order and a smoke test (`pytest -q` plus a one-epoch CNN run, about 35 s). Splits, seeds, final configs, model files and results are included. The demo shows predictions on a valid and an invalid file, follows `mlp_final.yaml` → `pytorch_mlp_seed0.pt` → its row in `results_table.csv` → Fig. 11, explains why I split by person, and shows the tilted-phone failure.

---

## Appendix A — Layer tables

**PyTorch MLP (input 84 features).**

| Layer | Output shape | Activation | Params |
|---|---|---|---|
| Linear 84→128 + BatchNorm | (B, 128) | ReLU, Dropout 0.2 | 10,880 + 256 |
| Linear 128→64 + BatchNorm | (B, 64) | ReLU, Dropout 0.2 | 8,256 + 128 |
| Linear 64→6 | (B, 6) | logits (softmax in loss) | 390 |
| **Total** | | | **19,910** |

**PyTorch 1D-CNN (input 12 × 128).**

| Layer | Output shape | Activation | Params |
|---|---|---|---|
| Conv1d 12→64, k5 + BatchNorm, MaxPool 2 | (B, 64, 64) | ReLU | 3,904 + 128 |
| Conv1d 64→128, k5 + BatchNorm, MaxPool 2 | (B, 128, 32) | ReLU | 41,088 + 256 |
| Conv1d 128→128, k5 + BatchNorm, MaxPool 2 | (B, 128, 16) | ReLU | 82,048 + 256 |
| Global average pool + Dropout 0.3 | (B, 128) | – | 0 |
| Linear 128→6 | (B, 6) | logits | 774 |
| **Total** | | | **128,454** |

## Appendix B — Tuning trials (validation macro-F1, seed 0; selected in bold)

| Model | Trial settings → val macro-F1 |
|---|---|
| RF | 100 trees: depth None 0.927, 20 0.929, 10 0.905; 300 trees: **None 0.937**, 20 0.937, 10 0.909 (all balanced) |
| MLP | [128,64] d0.2 **0.956**; [128,64] d0.4 0.953; [256,128] d0.2 0.944; [256,128] d0.4 0.945; [128,64] d0.2 weighted 0.947; [256,128] d0.2 weighted 0.934 |
| CNN | [32,64,64] k5 d0.3 0.938; k7 d0.3 0.940; k5 d0.5 0.955; [64,128,128] **k5 d0.3 0.967**; k7 d0.3 0.950; k5 d0.5 0.964 |

## Appendix C — Figures

| Fig. | File | Shows |
|---|---|---|
| 1 | fig01_class_distribution.png | Windows per class |
| 2 | fig02_signals_per_activity.png | Example signals per activity |
| 3 | fig03_rows_per_subject.png | Data per subject |
| 4 | fig04_channel_boxplots.png | Channel ranges |
| 5–7 | fig05–fig07 | NumPy MLP loss, tiny-batch overfit, learning-rate sweep |
| 8–9 | fig08_mlp_loss_curves.png, fig09_cnn_loss_curves.png | Train/val loss and val F1 |
| 10–12 | fig10–fig12 | Confusion matrices (RF, MLP, CNN) |
| 13 | fig13_per_class_f1.png | Per-class F1 |
| 14 | fig14_val_vs_test.png | Validation vs test |
| 15 | fig15_ablation_dropout.png | Dropout ablation |
| 16 | fig16_error_examples.png | Representative errors |
| 17 | fig17_noise_robustness.png | Noise stress test |
| 18 | fig18_per_subject_f1.png | Per-subject F1 |
| 19 | fig19_tradeoff.png | Latency vs test F1 |

## References

Malekzadeh, M., Clegg, R. G., Cavallaro, A. and Haddadi, H. (2019) 'Mobile sensor data anonymization', *Proceedings of the International Conference on Internet of Things Design and Implementation (IoTDI '19)*. ACM. Dataset: https://github.com/mmalekzadeh/motion-sense
