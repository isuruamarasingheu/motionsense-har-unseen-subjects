"""
Script: scripts/build_notebook_06.py
Purpose: Construct and execute notebooks/06_recommendation_inference.ipynb
with presentation-quality markdown, rendered benchmark evidence tables,
Figure 19 visualization, subprocess prediction demos, and limitations headings.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scripts.build_notebook_06")


def build_and_run_notebook() -> None:
    """Create and execute 06_recommendation_inference.ipynb."""
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "language_info": {
            "name": "python",
            "version": "3.11.9",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        }
    }

    cells = []

    # Cell 1: Title & Overview
    c1_md = """# 06_recommendation_inference.ipynb — Model Recommendation Evidence, Inference Pipeline & Limitations

**Module:** NB6057CEM Artificial Neural Networks — CW1 Practical  
**Task:** Task T5a, T5b & T5c — Benchmarking, Evidence, Recommendation, Inference CLI & Limitations (8 marks); plus a T6b config-to-result trace  
**Dataset:** MotionSense (A_DeviceMotion_data) — Malekzadeh et al. (2018/2019)  

---

### Coursework Brief Requirements (Task T5)
Task T5 evaluates the translation of empirical machine learning models into real-world deployable Human Activity Recognition (HAR) systems:
1. **Model Complexity & Latency Benchmarking (Task T5a)**:
   - Measure trainable parameter counts for neural networks and total tree nodes for Random Forest.
   - Measure model serialization file sizes on disk (in MB).
   - Profile CPU inference latency on **validation windows strictly** under two conditions:
     - **Model-Only**: preprocessed feature vectors or preprocessed raw sequence windows.
     - **End-to-End**: raw sensor stream windows ($128 \\times 12$) through feature extraction/scaling to final prediction.
   - Profile batch size 1 (real-time single-window streaming) and batch size 256 (batched offline throughput).
   - Fix PyTorch CPU thread count and report mean, standard deviation, and median latency over 100 runs following 10 warm-up runs.
2. **Recommendation Evidence Matrix (Task T5a)**:
   - Combine validation and test macro-F1, worst-subject test F1, noise robustness at $\\sigma \\in \\{0.25, 0.50\\}$, parameter counts, file sizes, and CPU latencies into a structured evidence table (`outputs/metrics/recommendation_evidence.csv`).
   - Plot Figure 19: Latency vs Test Macro-F1 trade-off (`outputs/figures/fig19_tradeoff.png`).
   - Give a recommendation that follows from this evidence, including when a simpler model is preferable (section 3).
3. **Inference CLI Pipeline (Task T5b)**:
   - Provide a modular CLI engine (`src/predict.py`) operating on arbitrary 50 Hz MotionSense sensor recordings.
   - Implement strict input integrity validation with clear error messages and non-zero exit codes (no Python tracebacks).
   - Implement out-of-scope domain shift warnings (channel range deviations $>10\\%$, gravity norm $|\\text{norm} - 1| > 0.05\\text{ g}$, low confidence $< 0.50$).
   - Support both human-readable ASCII tables and structured `--json` output.
4. **Limitations & Future Directions (Task T5c)**:
   - Outline critical limitations: dataset bias, generalisation boundaries, privacy & misuse risks, and next proposed experiments.
"""
    cells.append(nbf.v4.new_markdown_cell(c1_md))

    # Cell 2: Imports and Setup
    c2_code = """import json
import subprocess
import sys
from pathlib import Path

import IPython.display as ipd
import numpy as np
import pandas as pd

# Confirm working directory and paths
print(f"Python executable: {sys.executable}")
print(f"Working directory: {Path.cwd()}")
"""
    cells.append(nbf.v4.new_code_cell(c2_code))

    # Cell 3: Hardware Environment & Profiling Protocol
    c3_md = """## 1. Hardware Environment & Benchmark Methodology (Task T5a)

To ensure scientific reproducibility, latency profiling was executed under standardized hardware conditions:
- **Processor**: Intel 11th Gen Core i5-11400H @ 2.70 GHz (6 physical cores, 12 threads).
- **Execution Mode**: CPU-only inference on a laptop. A phone would be slower, so these timings compare the models with each other rather than predicting phone latency.
- **PyTorch Threading**: Pinned to 6 threads (`torch.set_num_threads(6)`).
- **Profiling Input**: **Validation windows strictly** (`outputs/cache/X_val_raw.npy`, `X_val_feat_scaled.npy`, `X_val_raw_scaled.npy`). In strict adherence to Project Rules, the test set was **never loaded or unblinded** for timing.
- **Timing Protocol**: High-resolution `time.perf_counter()` over 100 timed runs following 10 warm-up runs to eliminate JIT compilation, caching, and CPU frequency governor artifacts.
"""
    cells.append(nbf.v4.new_markdown_cell(c3_md))

    # Cell 4: Code - Load and Display Benchmark Table
    c4_code = """# Load benchmark metrics from outputs/metrics/benchmark.json
benchmark_path = Path("outputs/metrics/benchmark.json")
with open(benchmark_path, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

env_info = benchmark_data["environment"]
print("=" * 75)
print("BENCHMARK EXECUTION ENVIRONMENT:")
print(f"  CPU Model:       {env_info['cpu_name']}")
print(f"  Operating System:{env_info['os']}")
print(f"  Python Version:  {env_info['python_version']}")
print(f"  PyTorch Version: {env_info['torch_version']}")
print(f"  scikit-learn:    {env_info['sklearn_version']}")
print(f"  CPU Threads:     {env_info['torch_num_threads']}")
print("=" * 75)

# Build comprehensive latency profiling summary table
bench_rows = []
for model_key, m_info in benchmark_data["models"].items():
    lat = m_info["latency"]
    
    # Model-only
    mo_b1_mean = lat["model_only"]["batch_1"]["mean_ms"]
    mo_b1_std = lat["model_only"]["batch_1"]["std_ms"]
    mo_b1_med = lat["model_only"]["batch_1"]["median_ms"]
    mo_b256_per_win = lat["model_only"]["batch_256"]["per_window_mean_ms"]
    
    # End-to-end
    e2e_b1_mean = lat["end_to_end"]["batch_1"]["mean_ms"]
    e2e_b1_std = lat["end_to_end"]["batch_1"]["std_ms"]
    e2e_b1_med = lat["end_to_end"]["batch_1"]["median_ms"]
    e2e_b256_per_win = lat["end_to_end"]["batch_256"]["per_window_mean_ms"]
    
    size_str = f"{m_info['size_metric_value']:,} {m_info['size_metric_name']}"
    file_mb = f"{m_info['checkpoint_file_size_mb']:.2f} MB"
    
    bench_rows.append({
        "Model": m_info["display_name"],
        "Complexity": size_str,
        "File Size": file_mb,
        "Model-Only B=1 [ms]": f"{mo_b1_mean:.3f} ± {mo_b1_std:.3f} (med: {mo_b1_med:.3f})",
        "Model-Only B=256 [ms/win]": f"{mo_b256_per_win:.4f}",
        "End-to-End B=1 [ms]": f"{e2e_b1_mean:.2f} ± {e2e_b1_std:.2f} (med: {e2e_b1_med:.2f})",
        "End-to-End B=256 [ms/win]": f"{e2e_b256_per_win:.4f}",
    })

df_bench = pd.DataFrame(bench_rows)
ipd.display(df_bench)
"""
    cells.append(nbf.v4.new_code_cell(c4_code))

    # Cell 5: Markdown - Recommendation Evidence Matrix
    c5_md = """## 2. Multi-Criteria Recommendation Evidence Matrix (Task T5a)

The table below synthesises the complete empirical evidence across accuracy, generalisation gap, demographic fairness, sensor noise robustness, model size, and inference latency.

The recommendation based on this table is given in section 3. All numbers originate strictly from saved artifacts (`results_summary.json`, `noise_robustness.json`, `subgroup.json`, `benchmark.json`).
"""
    cells.append(nbf.v4.new_markdown_cell(c5_md))

    # Cell 6: Code - Load and Display Recommendation Evidence CSV
    c6_code = """evidence_csv_path = Path("outputs/metrics/recommendation_evidence.csv")
df_evidence = pd.read_csv(evidence_csv_path)

# Display formatted table
styled_table = df_evidence.style.set_table_styles([
    {"selector": "th", "props": [("background-color", "#2b5c8f"), ("color", "white"), ("font-weight", "bold"), ("text-align", "center")]},
    {"selector": "td", "props": [("text-align", "center"), ("padding", "6px 12px")]},
    {"selector": "caption", "props": [("caption-side", "top"), ("font-size", "14px"), ("font-weight", "bold"), ("padding-bottom", "8px")]}
]).set_caption("Table 1: Multi-Criteria HAR Model Recommendation Evidence Matrix (Tasks T5a)")

ipd.display(styled_table)
"""
    cells.append(nbf.v4.new_code_cell(c6_code))

    # Cell 7: Markdown - Trade-Off Analysis & Figure 19
    c7_md = """## 3. Generalisation vs Latency Trade-Off Analysis (Figure 19)

Figure 19 illustrates the multi-objective Pareto trade-off between **held-out test macro-F1** (y-axis, with $\\pm 1\\sigma$ inter-seed error bars) and **end-to-end CPU inference latency** for batch size 1 (x-axis, log scale). Marker bubble size is proportional to checkpoint serialization file size on disk (MB).

### Key Empirical Findings:
1. **Random Forest Baseline**:
   - Ties with the MLP for the highest test macro-F1 ($0.8860 \\pm 0.0016$ vs $0.8849 \\pm 0.0078$) and has the highest worst-subject test score ($0.8244$ on Subject 18).
   - High memory/disk footprint ($22.37\\text{ MB}$) due to storing 300 deep unpruned decision trees ($208,268$ total nodes).
   - Single-window latency: With single-threaded execution (`n_jobs=1`), single-window latency is $7.56\\text{ ms}$ (drastically reducing overhead compared to multi-threaded `n_jobs=-1` at $31.95\\text{ ms}$, where thread-pool dispatch overhead dominates single-sample evaluation).
2. **PyTorch MLP**:
   - Ultra-lightweight footprint: $19,910$ parameters ($0.08\\text{ MB}$ checkpoint).
   - Competitive test macro-F1 ($0.8849 \\pm 0.0078$), matching Random Forest within error margins.
   - Extremely rapid single-window end-to-end latency ($0.48\\text{ ms}$).
   - Vulnerable to sensor noise: F1 drops to $0.7467$ at $\\sigma=0.25$ and collapses to $0.2166$ at $\\sigma=0.50$.
3. **PyTorch 1D-CNN**:
   - Eliminates feature engineering: directly processes raw 12-channel motion streams ($0.44\\text{ ms}$ single-window end-to-end latency).
   - Highest noise robustness: retains $\\text{F1} = 0.8518$ at $\\sigma=0.25$ and $\\text{F1} = 0.6064$ at $\\sigma=0.50$ (where tabular models fail).
   - Lowest clean test macro-F1 and the largest seed-to-seed spread ($0.8615 \\pm 0.0293$, about 4 times the MLP's $\\pm 0.0078$).
"""
    cells.append(nbf.v4.new_markdown_cell(c7_md))

    # Cell 8: Code - Display Figure 19
    c8_code = """fig19_path = Path("outputs/figures/fig19_tradeoff.png")
if fig19_path.is_file():
    ipd.display(ipd.Image(filename=str(fig19_path), width=750))
else:
    print(f"Warning: Figure 19 not found at {fig19_path}")
"""
    cells.append(nbf.v4.new_code_cell(c8_code))

    # Recommendation (Task T5a)
    rec_md = """### Recommendation (Task T5a)

**I recommend the PyTorch MLP** for an activity-tracking phone app used by new people.

- **Accuracy:** its test macro-F1 (0.885 ± 0.008) is the same as the Random Forest's (0.886 ± 0.002) within seed-to-seed variation.
- **Size and speed:** it is about 280 times smaller (0.08 MB vs 22.37 MB) and about 16 times faster per window (0.48 ms vs 7.56 ms, batch size 1, end-to-end, on the laptop CPU above). Both matter on a phone.
- **When the simpler model is better:** the Random Forest is the safer choice when consistent results matter most. It varied least between seeds, had the best worst-subject score (0.824 on subject 18), and coped better with the tilted-phone recording (49% of windows wrong, vs 96% for the MLP and 98% for the CNN).
- **When the CNN is better:** only if noisy sensor input is expected. It kept 0.606 macro-F1 at noise level σ = 0.5, while the MLP fell to 0.217 and the Random Forest to 0.233. On clean data it is the least accurate (0.861) and the least stable (± 0.029).

**Not deployment-ready:** all models score 0.04–0.09 lower on test than on validation, the test set has only 4 people, and the tilted-phone recording defeats the recommended MLP (section 4.2).
"""
    cells.append(nbf.v4.new_markdown_cell(rec_md))

    # Cell 9: Markdown - Inference CLI Engine Overview
    c9_md = """## 4. Practical Real-Time Inference CLI Pipeline (Task T5b)

The inference module `src/predict.py` is a working command-line prototype (not deployment-ready; see section 6):
- **CLI Syntax**:
  ```bash
  python -m src.predict --model {rf,mlp,cnn} --input <file.csv> [--seed K] [--json]
  ```
- **Input Contract**:
  - Exactly 12 MotionSense sensor columns in any order (extra columns and unnamed index columns ignored).
  - Assumed 50 Hz sampling rate. Minimum length: $\\ge 128$ samples (2.56 s).
  - Sliding windowing: window length 128 samples, stride 64 samples (50% overlap).
- **Out-of-Scope Domain Shift Guardrails**:
  1. **Sensor Value Bounds**: Flags any channel exceeding the training min/max bounds by $>10\\%$ (derived strictly from `outputs/preprocessing/channel_ranges.json`).
  2. **Gravity Norm Invariance**: Checks $|\\|\\mathbf{g}\\| - 1.0| > 0.05\\text{ g}$ to detect uncalibrated or non-MotionSense sensors.
  3. **Low Confidence Alert**: Flags predictions where $\\max P(\\text{class}) < 0.50$.
- **Validation & Error Handling**:
  - Rejects missing files, empty files, malformed columns, NaNs/Infs, or insufficient lengths with clean error messages and non-zero exit codes (no Python tracebacks).
"""
    cells.append(nbf.v4.new_markdown_cell(c9_md))

    # Cell 10: Code - Subprocess Demo: Valid Walking Sample
    c10_code = """# 4.1 Valid Sample Demonstration (Test Subject 8, wlk_7, rows 500-999)
valid_sample = "samples/valid_walking_subject8.csv"

print("--- Running recommended PyTorch MLP (seed 0) on Valid Sample ---")
res_mlp = subprocess.run(
    [sys.executable, "-m", "src.predict", "--model", "mlp", "--input", valid_sample],
    capture_output=True, text=True
)
print(res_mlp.stdout)
"""
    cells.append(nbf.v4.new_code_cell(c10_code))

    # Cell 11: Code - Subprocess Demo: Orientation Failure Case
    c11_code = """# 4.2 Known Physical Failure Demonstration (Test Subject 1, wlk_7, rows 500-999)
# Demonstrates sensor orientation sensitivity identified in Phase 4 error analysis
failure_sample = "samples/failure_walking_subject1.csv"

print("--- Running PyTorch MLP (seed 0) on Tilted-Phone Walking Sample (true activity: walking) ---")
res_fail = subprocess.run(
    [sys.executable, "-m", "src.predict", "--model", "mlp", "--input", failure_sample],
    capture_output=True, text=True
)
print(res_fail.stdout)
"""
    cells.append(nbf.v4.new_code_cell(c11_code))

    # Cell 12: Code - Subprocess Demo: Out-of-Scope Warning
    c12_code = """# 4.3 Out-of-Scope Warning Demonstration (Magnified Acceleration x20)
oos_sample = "samples/out_of_scope_scaled.csv"

print("--- Running PyTorch MLP (seed 0) on Out-of-Scope Scaled Sample ---")
res_oos = subprocess.run(
    [sys.executable, "-m", "src.predict", "--model", "mlp", "--input", oos_sample],
    capture_output=True, text=True
)
print(res_oos.stdout)
"""
    cells.append(nbf.v4.new_code_cell(c12_code))

    # Cell 13: Code - Subprocess Demo: Input Validation and Rejection
    c13_code = """# 4.4 Schema and Integrity Validation Demonstrations (Expected Non-Zero Exit Codes)
invalid_cases = [
    ("Short Input (<128 rows)", "samples/invalid_short.csv"),
    ("Missing Required Column", "samples/invalid_columns.csv"),
    ("Corrupted NaN Entries", "samples/invalid_nan.csv"),
    ("Non-Existent File", "samples/non_existent.csv"),
]

for label, filepath in invalid_cases:
    res = subprocess.run(
        [sys.executable, "-m", "src.predict", "--model", "mlp", "--input", filepath],
        capture_output=True, text=True
    )
    print(f"Case: {label}")
    print(f"  Command: python -m src.predict --model mlp --input {filepath}")
    print(f"  Exit Code: {res.returncode} (Expected Non-Zero)")
    print(f"  Stderr Output: {res.stderr.strip()}")
    print("-" * 75)
"""
    cells.append(nbf.v4.new_code_cell(c13_code))

    # Cell 14: Code - Subprocess Demo: JSON Mode
    c14_code = """# 4.5 Structured JSON Mode Demonstration (--json flag)
res_json = subprocess.run(
    [sys.executable, "-m", "src.predict", "--model", "mlp", "--input", valid_sample, "--json"],
    capture_output=True, text=True
)
parsed_json = json.loads(res_json.stdout)

print("Parsed JSON Summary:")
print(f"  Model:              {parsed_json['model']} (seed {parsed_json['seed']})")
print(f"  Predicted Activity: {parsed_json['predicted_activity']} ({parsed_json['predicted_code']})")
print(f"  Mean Confidence:    {parsed_json['mean_confidence']:.4f}")
print(f"  Windows Analyzed:   {parsed_json['n_windows']}")
print("  Class Probabilities:")
for code, prob in parsed_json['mean_class_probabilities'].items():
    print(f"    {code}: {prob:.4f}")
"""
    cells.append(nbf.v4.new_code_cell(c14_code))

    # Trace section (T6b): config -> checkpoint -> results row -> figure
    trace_md = """## 5. Tracing One Result: Config → Checkpoint → Metrics → Figure (Task T6b)

This follows the recommended model (PyTorch MLP, seed 0) from its frozen configuration to its reported test score.
Every value below is read from saved files; nothing is retrained or re-evaluated.
"""
    cells.append(nbf.v4.new_markdown_cell(trace_md))

    trace_code = """import yaml

cfg_path = Path("configs/mlp_final.yaml")
snap_path = Path("outputs/checkpoints/pytorch_mlp_seed0_config.yaml")
ckpt_path = Path("outputs/checkpoints/pytorch_mlp_seed0.pt")
info_path = Path("outputs/checkpoints/run_info_pytorch_mlp_seed0.json")

# Step 1: frozen configuration
print("STEP 1 - Frozen configuration:", cfg_path)
print(cfg_path.read_text())

# Step 2: the config copy saved next to the checkpoint must match the frozen config
cfg = yaml.safe_load(cfg_path.read_text())
snap = yaml.safe_load(snap_path.read_text())
print("STEP 2 - Checkpoint:", ckpt_path, f"({ckpt_path.stat().st_size / 1024:.1f} KB)")
for section in ["model", "training"]:
    same = cfg.get(section) == snap.get(section)
    print(f"  '{section}' settings in saved snapshot match frozen config: {same}")

# Step 3: environment recorded at training time
info = json.loads(info_path.read_text())
print()
print("STEP 3 - Run environment:", info_path)
for key in ["timestamp", "python_version", "torch_version", "cpu_name", "seed"]:
    if key in info:
        print(f"  {key}: {info[key]}")

# Step 4: the row in the results table produced by src/evaluate.py (test set used once)
results = pd.read_csv("outputs/metrics/results_table.csv")
row = results[(results["model"] == "pytorch_mlp") & (results["seed"] == 0)]
print()
print("STEP 4 - Row in outputs/metrics/results_table.csv:")
print(row.to_string(index=False))
"""
    cells.append(nbf.v4.new_code_cell(trace_code))

    trace_fig = """# Step 5: confusion matrix for this checkpoint on the test set
fig11_path = Path("outputs/figures/fig11_cm_mlp.png")
print("STEP 5 - Test confusion matrix:", fig11_path)
ipd.display(ipd.Image(filename=str(fig11_path), width=600))
"""
    cells.append(nbf.v4.new_code_cell(trace_fig))

    # Cell 15: Markdown - Limitations & Future Work (Task T5c)
    c15_md = """## 6. Limitations & Future Directions (Task T5c)

These results are useful for comparing the three models with each other. They do **not** show that any of them is ready to be used in a real app. This section explains why.

### Dataset bias

**What the problem is:** the data comes from a small and very similar group.
- Only **24 people**, all adults aged **18 to 46**. There are no children or people over 46, and the dataset says nothing about people with mobility problems (for example walking with a stick).
- Everyone used the **same phone model** (an iPhone) and carried it in the **same way**. A different phone, or a phone in a bag, hand or jacket, would give different signals.
- The activities were recorded as planned trials, not during normal daily life, so they may be "cleaner" than real movement.
- The classes are uneven: in training there are 3,488 sitting windows but only 1,226 downstairs windows.

**Why it matters:** a model can only learn what it has seen. People who move differently from these 24 are likely to get worse predictions, and we have no data to measure how much worse.

### Generalisation

**What the problem is:** the models do worse on new people, and some recordings break them completely.
- Every model scored **0.04 to 0.09 lower** on the 4 test people than on the 4 validation people. New people are harder than the numbers on validation suggest.
- The test score comes from only **4 people**. One unusual person can move the result a lot, so the "true" score for the general public could be noticeably higher or lower.
- **Tilted phone (subject 1, walking trial 7):** the phone sat at a different angle in the pocket. The MLP got 96% of windows wrong and the CNN 98%; the Random Forest did better but still got 49% wrong. Instead of walking, the CNN mostly predicted "upstairs" and the MLP "upstairs" or "jogging", often with high confidence.
- **Another walking recording (subject 21, trial 7)** also fails for all three models (86–93% of windows wrong). We have not yet found the cause, which shows we do not fully understand where the models break.
- The models are often **confidently wrong** (for example 0.98 confidence for the wrong activity), so the confidence score cannot be trusted to warn the user.

**Why it matters:** an app built on these models would work well for some people and badly for others, and it could not tell which is which.

### Privacy & misuse

**What the problem is:** motion data is more personal than it looks.
- The way someone walks is close to a **fingerprint**: it can be used to identify a person.
- Activity patterns can reveal **health** (for example, slowing down over time, or difficulty with stairs) and **daily routine** (when someone is home, asleep or out).
- The same model could be **misused** to track people without their knowledge, for example by an employer or an insurer.

**What should be done:** run predictions **on the phone itself** so raw sensor data never leaves the device, ask for clear **consent**, store only the predicted activity (not the raw signals), and let users delete their data.

### Next experiment

**Idea:** many errors seem to come from the phone being held at a different angle. So, during training, **randomly rotate the sensor signals** to imitate the phone being tilted in different ways. This teaches the model that the same movement can look different depending on how the phone sits.

**How to test it fairly:**
- Keep everything else the same: the same subject split, the same 3 seeds, the same settings, chosen on validation only.
- Compare against the current MLP using macro-F1 on the test set, and look at the tilted recording (subject 1, walking trial 7) separately.
- **The idea has failed if** test macro-F1 does not improve beyond the seed-to-seed variation, or if the tilted recording is still mostly wrong.
"""
    cells.append(nbf.v4.new_markdown_cell(c15_md))

    # Set cells on notebook
    nb.cells = cells

    out_notebook_path = Path("notebooks/06_recommendation_inference.ipynb")
    out_notebook_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Executing notebook %s ...", out_notebook_path)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    with open(out_notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    logger.info("Successfully generated and executed %s", out_notebook_path)


if __name__ == "__main__":
    build_and_run_notebook()

