import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_assist2012_ablation


class AblationSummaryTest(unittest.TestCase):
    def test_aggregate_experiment_rows(self):
        rows = [
            {"experiment": "target_only", "seed": 2024, "best_valid_auc": 0.78, "test_auc": 0.79, "test_acc": 0.76, "test_rmse": 0.40, "test_loss": 0.50},
            {"experiment": "target_only", "seed": 2025, "best_valid_auc": 0.80, "test_auc": 0.81, "test_acc": 0.77, "test_rmse": 0.39, "test_loss": 0.49},
            {"experiment": "target_difficulty", "seed": 2024, "best_valid_auc": 0.82, "test_auc": 0.83, "test_acc": 0.78, "test_rmse": 0.38, "test_loss": 0.48},
        ]

        aggregate_rows = run_assist2012_ablation.aggregate_experiment_rows(rows)
        self.assertEqual(len(aggregate_rows), 2)

        target_only_row = next(row for row in aggregate_rows if row["experiment"] == "target_only")
        self.assertEqual(target_only_row["seeds"], "2024,2025")
        self.assertEqual(target_only_row["runs"], 2)
        self.assertAlmostEqual(target_only_row["test_auc_mean"], 0.80, places=8)
        self.assertAlmostEqual(target_only_row["test_auc_std"], 0.01, places=8)
        self.assertTrue("±" in target_only_row["test_auc_mean_pm_std"])


if __name__ == "__main__":
    unittest.main()
