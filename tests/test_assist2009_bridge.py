import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ahskt.data.assist2009 import build_split_bundle, fit_behavior_clusters


class Assist2009BridgeTest(unittest.TestCase):
    def test_build_split_bundle_with_padding(self):
        records = {
            "1": [
                [101, 1.0, 1, 1000, 1.0, 1, 0, 0],
                [102, 2.0, 0, 2000, 2.0, 2, 1, 0],
                [103, 1.0, 1, 3000, 3.0, 1, 0, 0],
            ]
        }
        topic2index = {"101": 0, "102": 1, "103": 2}
        kc2index = {"1": 0, "2": 1}
        time_factor = {"1": [0.1, 0.2, 0.3]}
        attempts_factor = {"1": [0.4, 0.5, 0.6]}
        hints_factor = {"1": [0.7, 0.8, 0.9]}
        topic_difficulty = {101: 51, 102: 26, 103: 88}
        kc_difficulty = {"1": 66, "2": 44}
        behavior_cluster_assets = fit_behavior_clusters(
            train_records=records,
            time_factor=time_factor,
            attempts_factor=attempts_factor,
            hints_factor=hints_factor,
            n_clusters=2,
            random_seed=7,
        )

        bundle = build_split_bundle(
            records=records,
            topic2index=topic2index,
            kc2index=kc2index,
            time_factor=time_factor,
            attempts_factor=attempts_factor,
            hints_factor=hints_factor,
            topic_difficulty=topic_difficulty,
            kc_difficulty=kc_difficulty,
            default_difficulty=50,
            sequence_length=5,
            remainder_min_len=2,
            behavior_cluster_assets=behavior_cluster_assets,
        )

        self.assertEqual(bundle.question_ids.shape, (1, 5))
        self.assertEqual(bundle.mask.tolist(), [[1, 1, 1, 0, 0]])
        self.assertEqual(bundle.question_ids[0, :3].tolist(), [1, 2, 3])
        self.assertEqual(bundle.concept_ids[0, :3].tolist(), [1, 2, 1])
        self.assertTrue(all(cluster_id in {1, 2} for cluster_id in bundle.behavior_cluster[0, :3].tolist()))
        self.assertEqual(bundle.speed_relative_student.shape, (1, 5))
        self.assertEqual(bundle.speed_relative_question.shape, (1, 5))
        self.assertEqual(bundle.behavior_soft_membership.shape, (1, 5, 3))
        self.assertAlmostEqual(float(bundle.behavior_soft_membership[0, 0, :].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(bundle.behavior_soft_membership[0, 1, :].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(bundle.behavior_soft_membership[0, 2, :].sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
