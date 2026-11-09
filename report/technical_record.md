# Technical Record: Human Activity Recognition with MotionSense

**Module:** NB6057CEM Artificial Neural Networks — CW1 Practical  
**Target:** 6-class Human Activity Recognition (dws, ups, wlk, jog, sit, std)  
**Input:** 12-channel smartphone motion data (128 samples / 2.56 s windows @ 50 Hz)  

---

## T1. Problem Definition & Data Preparation
<!-- ~300 words: Intended user, prediction target, unit of observation, practical value, licence, ethics, audit findings (fig01–04), subject-wise grouped splitting justification, leakage prevention -->

## T2. Mathematical Foundations & From-Scratch Implementation
<!-- ~250 words: Biological neuron analogy & limits, forward/backward derivations, array dimensions, loss convergence, grad-check table, tiny-batch overfit sanity check -->

## T3. Baseline & Framework Architectures
<!-- ~300 words: Non-neural Random Forest baseline, PyTorch MLP (84 features) vs 1D-CNN (raw windows), layer tables, trainable parameters, training controls, declared tuning budget & results -->

## T4. Experimental Evaluation & Comparative Analysis
<!-- ~350 words: Results table (mean ± std across seeds 0, 1, 2), primary metric (Macro-F1) justification, confusion matrices, ablation study (validation only), error analysis, noise stress test, demographic subgroup analysis -->

## T5. Practical Recommendation & Critical Appraisal
<!-- ~200 words: Multi-criteria recommendation (accuracy vs variance vs parameters vs CPU latency), CLI inference demonstration, honest failure modes & deployment limitations, future research -->

## T6. Reproducibility & Demonstration Summary
<!-- ~50 words: Environment replication commands, execution pipeline, demo walkthrough -->

---

## Appendix
### A1. Generative AI & Borrowed Code Declaration
### A2. Unsuccessful Experiments & Hyperparameter Trials Log

