"""
Utility functions for the Human Activity Recognition (HAR) project.

This module provides common utilities for:
- Reproducibility (setting random seeds across Python, NumPy, and PyTorch).
- Serialization (saving and loading JSON data and YAML configuration files).
- Provenance tracking (recording runtime environment, package versions, OS, CPU,
  and timestamps into run_info.json).

Design constraints:
- Every run records run_info.json (seed, versions, CPU, timestamp).
- Set all seeds (python random, numpy, torch) from config.
- Docstrings and comments explain WHY, not just what.
- Keep code simple, clean, and easily explainable.
"""

from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set seeds across all random number generators to ensure reproducibility.

    Why:
        Neural network weight initialisation, data shuffling, and stochastic
        optimisation introduce non-determinism. Pinned seeds guarantee that
        experiments, ablation studies, and evaluation results are exactly
        reproducible across different runs on CPU.

    Args:
        seed: Integer random seed (default: 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _json_serializable_fallback(obj: Any) -> Any:
    """Fallback converter for objects not natively serializable by json."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return str(obj)


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """
    Save data to a JSON file, automatically creating parent directories.

    Why:
        All metrics, experiment records, and split maps must be persisted
        in human-readable JSON files rather than hard-coded into reports.

    Args:
        data: Python dictionary, list, or JSON-serializable object.
        path: Destination file path.
        indent: Indentation spaces for formatting (default: 2).

    Returns:
        Path object pointing to the written file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_json_serializable_fallback)
    return out_path


def load_json(path: str | Path) -> Any:
    """
    Load data from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content (dict, list, etc.).

    Raises:
        FileNotFoundError: If the target file does not exist.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"JSON file not found: {target}")
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(yaml_path: str | Path) -> dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Why:
        Decouples hyperparameters, data paths, and model architectures from
        implementation scripts, enabling frozen configs and reproducible runs.

    Args:
        yaml_path: Path to the .yaml configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is missing.
    """
    cfg_file = Path(yaml_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def get_cpu_name() -> str:
    """
    Detect the host CPU brand and model name across Windows, Linux, and macOS.

    Why:
        Inference benchmarking and execution provenance require logging the exact
        processor used, especially for CPU-only training and inference evaluation.

    Returns:
        String identifying the processor model.
    """
    # 1. On Windows, read the descriptive ProcessorNameString from the registry
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if name and str(name).strip():
                return str(name).strip()
        except Exception:
            pass

    # 2. On Linux, check /proc/cpuinfo
    if Path("/proc/cpuinfo").exists():
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    # 3. Fallbacks: environment variable or platform calls
    return (
        os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.processor()
        or platform.machine()
        or "Unknown CPU"
    )


def write_run_info(
    path: str | Path,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Record runtime environment metadata into a JSON file for reproducibility.

    Why:
        Every training or evaluation run must record
        environment metadata (Python, library versions, OS, CPU, timestamp)
        alongside checkpoints to ensure audits and reproduction are verifiable.

    Args:
        path: Destination path for the run_info.json file.
        extra_info: Optional dictionary of additional run metadata (e.g., seed,
            config path, model name, hyperparameter overrides).

    Returns:
        Dictionary containing the recorded run metadata.
    """
    run_info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "matplotlib_version": matplotlib.__version__,
        "joblib_version": joblib.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu_name": get_cpu_name(),
    }

    if extra_info:
        run_info.update(extra_info)

    save_json(run_info, path)
    return run_info

