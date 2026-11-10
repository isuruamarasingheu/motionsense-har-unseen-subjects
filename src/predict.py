"""
Module: src.predict
Purpose: T5b CLI inference engine for practical single-file activity prediction.

Usage:
    python -m src.predict --model {rf,mlp,cnn} --input <file.csv> [--seed K] [--json]

Coursework brief & plan requirements:
1. Default seed = representative seed (RF: 1, MLP: 0, CNN: 2). Load saved checkpoint + saved scalers
   (never refit). Reuse src/data/windows.py and src/data/features.py.
2. Input: CSV with the 12 MotionSense sensor columns (exact names, any order, extra columns ignored,
   optional unnamed index column ignored), 50 Hz.
3. Window it (128 samples, stride 64), predict each window, then output:
     Predicted activity: WALKING (mean confidence 0.91)
     Windows: 12 | per-class mean probability table | per-window predictions
   --json prints the same as JSON.
4. Validation with clear messages and NON-ZERO exit code, no Python traceback:
   - file missing / not CSV / empty
   - missing required columns (list which)
   - non-numeric values, NaN or inf (report row numbers)
   - fewer than 128 rows ("need at least 2.56 s of data at 50 Hz")
5. Out-of-scope WARNINGS (still predict, but flag clearly):
   - any channel outside the TRAINING min/max range by >10%: compute ranges from the TRAIN split only
     and save to outputs/preprocessing/channel_ranges.json
   - gravity vector norm not ≈ 1 g (|norm - 1| > 0.05) -> not MotionSense-like data
   - max class probability < 0.5 -> "low confidence, prediction unreliable"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.data.features import extract_features
from src.data.load import ACTIVITY_NAMES, SENSOR_COLUMNS
from src.data.preprocess import FeatureScaler, RawWindowScaler
from src.models.cnn import PyTorchCNN1D
from src.models.mlp import PyTorchMLP

# MotionSense 6 activity classes in numeric index order (0..5)
CLASS_NAMES: list[str] = ["dws", "ups", "wlk", "jog", "std", "sit"]

# Human-readable activity display labels in capital letters
ACTIVITY_DISPLAY: dict[str, str] = {
    "dws": "DOWNSTAIRS",
    "ups": "UPSTAIRS",
    "wlk": "WALKING",
    "jog": "JOGGING",
    "std": "STANDING",
    "sit": "SITTING",
}

# Representative seeds selected strictly on validation set performance
DEFAULT_SEEDS: dict[str, int] = {
    "rf": 1,
    "mlp": 0,
    "cnn": 2,
}


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MotionSense Human Activity Recognition Inference Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["rf", "mlp", "cnn"],
        help="Model architecture to use for prediction: rf (Random Forest), mlp (PyTorch MLP), cnn (PyTorch 1D-CNN)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to continuous sensor input CSV recording (50 Hz)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Model seed to load (default: representative seed: rf=1, mlp=0, cnn=2)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output structured prediction results as JSON instead of human-readable text",
    )
    return parser.parse_args(args)


def emit_error_and_exit(message: str, exit_code: int = 1) -> None:
    """Print clean error message to stderr without traceback and exit."""
    sys.stderr.write(f"Error: {message}\n")
    sys.exit(exit_code)


def validate_and_load_input(input_path_str: str) -> pd.DataFrame:
    """
    Validate input file existence, CSV format, required sensor columns,
    numeric validity, and minimum sample count.

    Exits with clear non-zero code on failure without Python traceback.
    """
    input_path = Path(input_path_str)

    # 1. Check file existence
    if not input_path.exists():
        emit_error_and_exit(f"Input file '{input_path}' does not exist.")
    if not input_path.is_file():
        emit_error_and_exit(f"Input path '{input_path}' is not a regular file.")

    # 2. Check non-empty file size
    if input_path.stat().st_size == 0:
        emit_error_and_exit(f"Input file '{input_path}' is empty (0 bytes).")

    # 3. Parse CSV (allowing comment header lines)
    try:
        df = pd.read_csv(input_path, comment="#")
    except Exception as e:
        emit_error_and_exit(f"Failed to parse '{input_path}' as CSV: {e}")

    if df.empty:
        emit_error_and_exit(f"Input CSV '{input_path}' contains no data rows.")

    # Drop optional unnamed index column if present
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 4. Check for required sensor columns
    missing_cols = [col for col in SENSOR_COLUMNS if col not in df.columns]
    if missing_cols:
        emit_error_and_exit(
            f"Missing required sensor column(s): {missing_cols}. "
            f"Input must contain all 12 MotionSense sensor columns: {SENSOR_COLUMNS}."
        )

    # 5. Check for non-numeric values, NaNs, or Infs
    sensor_df = df[SENSOR_COLUMNS]

    # Check for NaN / non-numeric
    nan_mask = sensor_df.isna().any(axis=1)
    nan_rows = df.index[nan_mask].tolist()

    # Check for infs
    try:
        arr_numeric = sensor_df.to_numpy(dtype=float)
        inf_mask = np.isinf(arr_numeric).any(axis=1)
        inf_rows = df.index[inf_mask].tolist()
    except (ValueError, TypeError):
        # Could not convert to float, means non-numeric string values exist
        bad_rows = []
        for col in SENSOR_COLUMNS:
            coerced = pd.to_numeric(sensor_df[col], errors="coerce")
            bad = df.index[coerced.isna()].tolist()
            bad_rows.extend(bad)
        bad_rows = sorted(list(set(bad_rows)))
        emit_error_and_exit(
            f"Non-numeric values found in sensor columns at row(s): {bad_rows[:15]}"
            f"{'...' if len(bad_rows) > 15 else ''}."
        )

    invalid_rows = sorted(list(set(nan_rows + inf_rows)))
    if invalid_rows:
        row_display = ", ".join(map(str, invalid_rows[:15]))
        if len(invalid_rows) > 15:
            row_display += f"... (total {len(invalid_rows)} invalid rows)"
        emit_error_and_exit(f"Non-numeric, NaN, or Inf values found at row index/indices: [{row_display}].")

    # 6. Minimum row requirement: >= 128 rows for at least one full window
    if len(df) < 128:
        emit_error_and_exit(
            f"Input CSV has only {len(df)} rows. Need at least 2.56 s of data at 50 Hz (128 samples)."
        )

    return df


def check_out_of_scope_warnings(
    df: pd.DataFrame,
    channel_ranges_path: Path = Path("outputs/preprocessing/channel_ranges.json"),
) -> list[str]:
    """
    Check for out-of-scope domain shifts and flag clear warnings:
    1. Any channel outside training min/max range by >10%.
    2. Gravity vector norm not approx 1.0 g (|norm - 1| > 0.05).
    """
    warnings: list[str] = []

    # 1. Channel ranges check (>10% outside training distribution)
    if channel_ranges_path.is_file():
        try:
            with open(channel_ranges_path, "r", encoding="utf-8") as f:
                channel_ranges = json.load(f)

            for col in SENSOR_COLUMNS:
                if col in channel_ranges:
                    c_min = float(df[col].min())
                    c_max = float(df[col].max())
                    lower_bound = float(channel_ranges[col]["lower_bound_10pct"])
                    upper_bound = float(channel_ranges[col]["upper_bound_10pct"])

                    if c_min < lower_bound or c_max > upper_bound:
                        warnings.append(
                            f"Channel '{col}' has values outside training range by >10% "
                            f"(observed range [{c_min:.4f}, {c_max:.4f}], "
                            f"training bound [{lower_bound:.4f}, {upper_bound:.4f}]). "
                            "Input exhibits large domain shift; prediction may be unreliable."
                        )
        except Exception:
            pass

    # 2. Gravity vector norm check (|norm - 1| > 0.05 g)
    gx = df["gravity.x"].to_numpy(dtype=float)
    gy = df["gravity.y"].to_numpy(dtype=float)
    gz = df["gravity.z"].to_numpy(dtype=float)
    g_norms = np.sqrt(gx**2 + gy**2 + gz**2)
    mean_g_norm = float(np.mean(g_norms))
    norm_dev = abs(mean_g_norm - 1.0)

    if norm_dev > 0.05:
        warnings.append(
            f"Gravity vector mean norm is {mean_g_norm:.4f} g (|norm - 1| = {norm_dev:.4f} > 0.05). "
            "Input sensor data may not be MotionSense-like or device accelerometer calibration differs."
        )

    return warnings


def segment_into_windows(
    df: pd.DataFrame,
    window_size: int = 128,
    stride: int = 64,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Segment input continuous sensor data into sliding windows of length 128 (stride 64).

    Returns:
        windows: NumPy array of shape (N, 128, 12), dtype float32.
        spans: List of (start_idx, end_idx) sample indices.
    """
    sensor_data = df[SENSOR_COLUMNS].to_numpy(dtype=np.float32)
    total_len = len(sensor_data)

    window_list: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []

    start_indices = range(0, total_len - window_size + 1, stride)
    for start in start_indices:
        end = start + window_size
        win = sensor_data[start:end]
        window_list.append(win)
        spans.append((start, end))

    return np.array(window_list, dtype=np.float32), spans


def predict_rf(
    windows: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Predict class posterior probabilities using Random Forest checkpoint."""
    ckpt_path = Path(f"outputs/checkpoints/baseline_rf_seed{seed}.joblib")
    if not ckpt_path.is_file():
        emit_error_and_exit(f"Random Forest checkpoint not found: '{ckpt_path}'.")

    scaler = FeatureScaler.load()
    rf = joblib.load(ckpt_path)
    # Coursework requirement: Use single-threaded inference (n_jobs=1) for single-file streaming
    rf.set_params(n_jobs=1)

    # Hand-crafted feature extraction -> standardization -> RF predict_proba
    features = extract_features(windows)
    features_scaled = scaler.transform(features)
    probs = rf.predict_proba(features_scaled)
    return probs


def predict_mlp(
    windows: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Predict class posterior probabilities using PyTorch MLP checkpoint."""
    ckpt_path = Path(f"outputs/checkpoints/pytorch_mlp_seed{seed}.pt")
    if not ckpt_path.is_file():
        emit_error_and_exit(f"PyTorch MLP checkpoint not found: '{ckpt_path}'.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ckpt["config"]["model"]
    model = PyTorchMLP(
        input_dim=int(model_cfg.get("input_dim", 84)),
        hidden_sizes=list(model_cfg.get("hidden_sizes", [128, 64])),
        dropout=float(model_cfg.get("dropout", 0.2)),
        num_classes=int(model_cfg.get("num_classes", 6)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = FeatureScaler.load()
    features = extract_features(windows)
    features_scaled = scaler.transform(features)

    with torch.no_grad():
        t = torch.from_numpy(features_scaled).float()
        logits = model(t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def predict_cnn(
    windows: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Predict class posterior probabilities using PyTorch 1D-CNN checkpoint."""
    ckpt_path = Path(f"outputs/checkpoints/pytorch_cnn_1d_seed{seed}.pt")
    if not ckpt_path.is_file():
        emit_error_and_exit(f"PyTorch 1D-CNN checkpoint not found: '{ckpt_path}'.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ckpt["config"]["model"]
    model = PyTorchCNN1D(
        in_channels=int(model_cfg.get("in_channels", 12)),
        sequence_length=int(model_cfg.get("sequence_length", 128)),
        filters=list(model_cfg.get("filters", [64, 128, 128])),
        kernel_size=int(model_cfg.get("kernel_size", 5)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        num_classes=int(model_cfg.get("num_classes", 6)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = RawWindowScaler.load()
    windows_scaled = scaler.transform(windows)  # (N, 128, 12)
    cnn_input = np.transpose(windows_scaled, (0, 2, 1))  # (N, 12, 128)

    with torch.no_grad():
        t = torch.from_numpy(cnn_input).float()
        logits = model(t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def run_inference(
    model_name: str,
    input_file: str,
    seed: int | None = None,
    as_json: bool = False,
) -> dict[str, Any]:
    """Execute complete CLI inference pipeline."""
    # 1. Determine random seed
    chosen_seed = DEFAULT_SEEDS[model_name] if seed is None else seed

    # 2. Validate and load CSV input
    df = validate_and_load_input(input_file)

    # 3. Check out-of-scope domain shifts
    warnings = check_out_of_scope_warnings(df)

    # 4. Segment into windows
    windows, spans = segment_into_windows(df, window_size=128, stride=64)
    n_windows = len(windows)

    # 5. Model forward pass
    if model_name == "rf":
        probs = predict_rf(windows, chosen_seed)
    elif model_name == "mlp":
        probs = predict_mlp(windows, chosen_seed)
    elif model_name == "cnn":
        probs = predict_cnn(windows, chosen_seed)
    else:
        emit_error_and_exit(f"Unknown model architecture: '{model_name}'.")

    # 6. Aggregate probabilities across windows
    mean_probs = np.mean(probs, axis=0)  # shape (6,)
    top_class_idx = int(np.argmax(mean_probs))
    top_code = CLASS_NAMES[top_class_idx]
    top_activity_display = ACTIVITY_DISPLAY[top_code]
    top_confidence = float(mean_probs[top_class_idx])

    # 7. Low confidence check (< 0.50)
    if top_confidence < 0.50:
        warnings.append(
            f"Maximum class probability ({top_confidence:.4f}) < 0.50: "
            "low confidence, prediction unreliable."
        )

    # 8. Build per-class mean probability table
    mean_probs_dict = {
        code: float(mean_probs[i]) for i, code in enumerate(CLASS_NAMES)
    }

    # 9. Build per-window predictions
    per_window_records = []
    for i in range(n_windows):
        start_idx, end_idx = spans[i]
        win_probs = probs[i]
        win_top_idx = int(np.argmax(win_probs))
        win_code = CLASS_NAMES[win_top_idx]
        win_conf = float(win_probs[win_top_idx])
        per_window_records.append({
            "window_index": i + 1,
            "start_sample": start_idx,
            "end_sample": end_idx,
            "predicted_activity_code": win_code,
            "predicted_activity": ACTIVITY_DISPLAY[win_code],
            "confidence": win_conf,
            "probabilities": {code: float(win_probs[c_i]) for c_i, code in enumerate(CLASS_NAMES)},
        })

    result_payload = {
        "model": model_name,
        "seed": chosen_seed,
        "input_file": str(input_file),
        "total_rows": len(df),
        "sampling_rate_hz": 50,
        "window_size": 128,
        "stride": 64,
        "n_windows": n_windows,
        "predicted_activity": top_activity_display,
        "predicted_code": top_code,
        "mean_confidence": top_confidence,
        "mean_class_probabilities": mean_probs_dict,
        "per_window_predictions": per_window_records,
        "warnings": warnings,
    }

    # 10. Output formatting
    if as_json:
        print(json.dumps(result_payload, indent=2))
    else:
        # Print prominent warnings if present
        if warnings:
            print("\n" + "!" * 72)
            print("OUT-OF-SCOPE WARNING(S) DETECTED:")
            for w in warnings:
                print(f"  [!] {w}")
            print("!" * 72 + "\n")

        # Standard coursework output line
        print(f"Predicted activity: {top_activity_display} (mean confidence {top_confidence:.2f})")
        print(f"Windows: {n_windows}\n")

        # Clean per-class mean probability distribution table
        print("Per-Class Mean Probability Table:")
        print(f"{'Code':<6} {'Activity':<14} {'Mean Probability':<18} {'Visual Bar'}")
        print("-" * 55)
        for code in CLASS_NAMES:
            p = mean_probs_dict[code]
            bar = "=" * int(round(p * 25))
            print(f"{code:<6} {ACTIVITY_NAMES[code]:<14} {p:<18.4f} |{bar}")
        print("-" * 55)

        # Per-window breakdown table (show first 10, indicate if truncated)
        print(f"\nPer-Window Predictions (total {n_windows} windows):")
        print(f"{'Win #':<7} {'Start..End':<14} {'Predicted Activity':<18} {'Confidence'}")
        print("-" * 55)
        display_limit = 12
        for rec in per_window_records[:display_limit]:
            span_str = f"{rec['start_sample']}..{rec['end_sample']}"
            act_str = f"{rec['predicted_activity']} ({rec['predicted_activity_code']})"
            print(f"{rec['window_index']:<7} {span_str:<14} {act_str:<18} {rec['confidence']:.4f}")
        if n_windows > display_limit:
            print(f"... [{n_windows - display_limit} additional windows omitted for brevity]")
        print("-" * 55 + "\n")

    return result_payload


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    try:
        run_inference(
            model_name=args.model,
            input_file=args.input,
            seed=args.seed,
            as_json=args.json,
        )
    except Exception as e:
        # Catch any unexpected unhandled exceptions and exit cleanly without ugly traceback
        emit_error_and_exit(f"Inference execution failed: {e}")


if __name__ == "__main__":
    main()
