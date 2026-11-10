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
**Task:** Task T5a, T5b & T5c — Model Benchmarking, Evidence Matrix, Inference CLI & Limitations (8 marks)  
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
   - *Note:* Per coursework instructions, this notebook produces only the factual evidence table without subjective recommendation prose, enabling transparent evaluation.
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
- **Execution Mode**: CPU-only inference, strictly matching real-world edge wearable device constraints.
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

*Note on Coursework Constraints:* Per project instructions, this section provides the **factual evidence table only**, without a subjective narrative recommendation. All numbers originate strictly from saved artifacts (`results_summary.json`, `noise_robustness.json`, `subgroup.json`, `benchmark.json`).
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
   - Achieves the highest test macro-F1 ($0.8860 \\pm 0.0016$) and highest worst-subject test score ($0.8244$ on Subject 18).
   - High memory/disk footprint ($22.37\\text{ MB}$) due to storing 300 deep unpruned decision trees ($208,268$ total nodes).
   - Single-window latency: With single-threaded execution (`n_jobs=1`), single-window latency is $7.56\\text{ ms}$ (drastically reducing overhead compared to multi-threaded `n_jobs=-1` at $31.95\\text{ ms}$, where thread-pool dispatch overhead dominates single-sample evaluation).
2. **PyTorch MLP**:
   - Ultra-lightweight footprint: $19,910$ parameters ($0.08\\text{ MB}$ checkpoint).
   - Competitive test macro-F1 ($0.8849 \\pm 0.0078$), matching Random Forest within error margins.
   - Extremely rapid single-window end-to-end latency ($0.48\\text{ ms}$).
   - Vulnerable to sensor noise: F1 collapses to $0.7467$ at $\\sigma=0.25$ and $0.2166$ at $\\sigma=0.50$.
3. **PyTorch 1D-CNN**:
   - Eliminates feature engineering: directly processes raw 12-channel motion streams ($0.44\\text{ ms}$ single-window end-to-end latency).
   - Highest noise robustness: retains $\\text{F1} = 0.8518$ at $\\sigma=0.25$ and $\\text{F1} = 0.6064$ at $\\sigma=0.50$ (where tabular models fail).
   - Slightly wider inter-seed variance ($0.8615 \\pm 0.0293$) on the held-out test split.
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

    # Cell 9: Markdown - Inference CLI Engine Overview
    c9_md = """## 4. Practical Real-Time Inference CLI Pipeline (Task T5b)

The inference module `src/predict.py` implements a production-ready, trial-bounded HAR inference engine:
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

print("--- Running PyTorch 1D-CNN (seed 2) on Valid Sample ---")
res_cnn = subprocess.run(
    ["python", "-m", "src.predict", "--model", "cnn", "--input", valid_sample],
    capture_output=True, text=True
)
print(res_cnn.stdout)

print("--- Running PyTorch MLP (seed 0) on Valid Sample ---")
res_mlp = subprocess.run(
    ["python", "-m", "src.predict", "--model", "mlp", "--input", valid_sample],
    capture_output=True, text=True
)
print(res_mlp.stdout)
"""
    cells.append(nbf.v4.new_code_cell(c10_code))

    # Cell 11: Code - Subprocess Demo: Orientation Failure Case
    c11_code = """# 4.2 Known Physical Failure Demonstration (Test Subject 1, wlk_7, rows 500-999)
# Demonstrates sensor orientation sensitivity identified in Phase 4 error analysis
failure_sample = "samples/failure_walking_subject1.csv"

print("--- Running PyTorch 1D-CNN on Phone-Orientation Inversion Sample ---")
res_fail = subprocess.run(
    ["python", "-m", "src.predict", "--model", "cnn", "--input", failure_sample],
    capture_output=True, text=True
)
print(res_fail.stdout)
"""
    cells.append(nbf.v4.new_code_cell(c11_code))

    # Cell 12: Code - Subprocess Demo: Out-of-Scope Warning
    c12_code = """# 4.3 Out-of-Scope Warning Demonstration (Magnified Acceleration x20)
oos_sample = "samples/out_of_scope_scaled.csv"

print("--- Running PyTorch 1D-CNN on Out-of-Scope Scaled Sample ---")
res_oos = subprocess.run(
    ["python", "-m", "src.predict", "--model", "cnn", "--input", oos_sample],
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
        ["python", "-m", "src.predict", "--model", "cnn", "--input", filepath],
        capture_output=True, text=True
    )
    print(f"Case: {label}")
    print(f"  Command: python -m src.predict --model cnn --input {filepath}")
    print(f"  Exit Code: {res.returncode} (Expected Non-Zero)")
    print(f"  Stderr Output: {res.stderr.strip()}")
    print("-" * 75)
"""
    cells.append(nbf.v4.new_code_cell(c13_code))

    # Cell 14: Code - Subprocess Demo: JSON Mode
    c14_code = """# 4.5 Structured JSON Mode Demonstration (--json flag)
res_json = subprocess.run(
    ["python", "-m", "src.predict", "--model", "cnn", "--input", valid_sample, "--json"],
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

    # Cell 15: Markdown - Limitations & Future Work (Task T5c)
    c15_md = """## 5. Limitations & Future Directions (Task T5c)

*Instructions for Student:* This section outlines critical limitations, failure boundaries, ethical considerations, and next experimental steps. Complete the analysis under each designated heading below.

### Dataset bias

### Generalisation

### Privacy & misuse

### Next experiment
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

