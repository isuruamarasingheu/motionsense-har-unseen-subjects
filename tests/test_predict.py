"""
Unit and Integration Tests: tests/test_predict.py
Purpose: T5b Verify robust validation, error handling, warning triggers,
and inference prediction for src.predict CLI engine.

Coursework requirements:
- Valid input returns exit 0 (tested on RF, MLP, CNN).
- Short input (<128 rows) returns non-zero exit code with clean message.
- Missing column input returns non-zero exit code listing missing column.
- NaN/Inf corrupted input returns non-zero exit code reporting row numbers.
- Out-of-scope scaled input prints clear warning and still returns exit 0.
- JSON output (--json) produces valid JSON structure with expected fields.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestPredictCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples_dir = Path("samples")
        cls.valid_sample = cls.samples_dir / "valid_walking_subject8.csv"
        cls.failure_sample = cls.samples_dir / "failure_walking_subject1.csv"
        cls.short_sample = cls.samples_dir / "invalid_short.csv"
        cls.missing_col_sample = cls.samples_dir / "invalid_columns.csv"
        cls.nan_sample = cls.samples_dir / "invalid_nan.csv"
        cls.out_of_scope_sample = cls.samples_dir / "out_of_scope_scaled.csv"

        # Ensure sample files exist
        for s in [
            cls.valid_sample,
            cls.failure_sample,
            cls.short_sample,
            cls.missing_col_sample,
            cls.nan_sample,
            cls.out_of_scope_sample,
        ]:
            if not s.is_file():
                raise FileNotFoundError(f"Required test sample not found: {s}")

    def run_cli(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Execute src.predict CLI as a subprocess."""
        cmd = [sys.executable, "-m", "src.predict"] + args
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_valid_input_cnn_exits_zero_and_predicts_walking(self) -> None:
        """Valid sample with CNN should exit 0 and predict WALKING."""
        res = self.run_cli(["--model", "cnn", "--input", str(self.valid_sample)])
        self.assertEqual(res.returncode, 0, f"Expected exit 0, got {res.returncode}. Stderr: {res.stderr}")
        self.assertIn("Predicted activity: WALKING", res.stdout)
        self.assertIn("Windows: 6", res.stdout)
        self.assertNotIn("Traceback", res.stderr)

    def test_valid_input_mlp_exits_zero_and_predicts_walking(self) -> None:
        """Valid sample with MLP should exit 0 and predict WALKING."""
        res = self.run_cli(["--model", "mlp", "--input", str(self.valid_sample)])
        self.assertEqual(res.returncode, 0, f"Expected exit 0, got {res.returncode}. Stderr: {res.stderr}")
        self.assertIn("Predicted activity: WALKING", res.stdout)
        self.assertIn("Windows: 6", res.stdout)
        self.assertNotIn("Traceback", res.stderr)

    def test_valid_input_rf_exits_zero(self) -> None:
        """Valid sample with RF should exit 0."""
        res = self.run_cli(["--model", "rf", "--input", str(self.valid_sample)])
        self.assertEqual(res.returncode, 0, f"Expected exit 0, got {res.returncode}. Stderr: {res.stderr}")
        self.assertIn("Predicted activity: WALKING", res.stdout)
        self.assertNotIn("Traceback", res.stderr)

    def test_invalid_short_rejected_with_nonzero_exit(self) -> None:
        """Input with <128 rows must be rejected with non-zero exit code and clean error message."""
        res = self.run_cli(["--model", "cnn", "--input", str(self.short_sample)])
        self.assertNotEqual(res.returncode, 0, "Short input must return non-zero exit code.")
        self.assertIn("Need at least 2.56 s of data at 50 Hz (128 samples)", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_invalid_missing_columns_rejected_with_nonzero_exit(self) -> None:
        """Input missing userAcceleration.z must be rejected and list missing column."""
        res = self.run_cli(["--model", "cnn", "--input", str(self.missing_col_sample)])
        self.assertNotEqual(res.returncode, 0, "Missing columns must return non-zero exit code.")
        self.assertIn("Missing required sensor column", res.stderr)
        self.assertIn("userAcceleration.z", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_invalid_nan_rejected_with_row_indices(self) -> None:
        """Input with NaN values must be rejected and report corrupted row numbers."""
        res = self.run_cli(["--model", "cnn", "--input", str(self.nan_sample)])
        self.assertNotEqual(res.returncode, 0, "NaN input must return non-zero exit code.")
        self.assertIn("Non-numeric, NaN, or Inf values found", res.stderr)
        # Verify specific row indices 12, 25, 50 are identified
        self.assertTrue("12" in res.stderr and "25" in res.stderr and "50" in res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_nonexistent_file_rejected_cleanly(self) -> None:
        """Non-existent input path must return non-zero exit without traceback."""
        res = self.run_cli(["--model", "cnn", "--input", "samples/does_not_exist.csv"])
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("does not exist", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_out_of_scope_scaled_prints_warning_and_predicts(self) -> None:
        """Magnified sensor inputs must print out-of-scope warnings but complete prediction (exit 0)."""
        res = self.run_cli(["--model", "cnn", "--input", str(self.out_of_scope_sample)])
        self.assertEqual(res.returncode, 0, f"Expected exit 0, got {res.returncode}. Stderr: {res.stderr}")
        self.assertIn("OUT-OF-SCOPE WARNING", res.stdout)
        self.assertIn("outside training range by >10%", res.stdout)
        self.assertIn("Predicted activity:", res.stdout)

    def test_json_output_mode(self) -> None:
        """Flag --json must produce valid, parseable JSON with all schema keys."""
        res = self.run_cli(["--model", "cnn", "--input", str(self.valid_sample), "--json"])
        self.assertEqual(res.returncode, 0, f"Expected exit 0, got {res.returncode}. Stderr: {res.stderr}")
        parsed = json.loads(res.stdout)
        self.assertEqual(parsed["model"], "cnn")
        self.assertEqual(parsed["predicted_activity"], "WALKING")
        self.assertEqual(parsed["n_windows"], 6)
        self.assertIn("mean_class_probabilities", parsed)
        self.assertIn("per_window_predictions", parsed)
        self.assertEqual(len(parsed["per_window_predictions"]), 6)


if __name__ == "__main__":
    unittest.main()

