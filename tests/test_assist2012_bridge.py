import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ahskt.data.assist2012 import build_sequence_records, records_to_bundle, shuffle_records


class Assist2012BridgeTest(unittest.TestCase):
    def test_sequence_records_and_bundle_padding(self):
        data_frame = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "timestamp": 1,
                    "question_ids": 11,
                    "question_difficulty": 21,
                    "correct": 1,
                    "concept_ids_internal": 0,
                    "concept_ids": 1,
                    "concept_difficulty": 31,
                    "attempts": 0.69,
                    "hints": 0.0,
                    "speed": 1.20,
                    "behavior_cluster": 1,
                },
                {
                    "user_id": 1,
                    "timestamp": 2,
                    "question_ids": 12,
                    "question_difficulty": 22,
                    "correct": 0,
                    "concept_ids_internal": 1,
                    "concept_ids": 2,
                    "concept_difficulty": 32,
                    "attempts": 1.10,
                    "hints": 0.2,
                    "speed": 0.80,
                    "behavior_cluster": 2,
                },
                {
                    "user_id": 1,
                    "timestamp": 3,
                    "question_ids": 13,
                    "question_difficulty": 23,
                    "correct": 1,
                    "concept_ids_internal": 2,
                    "concept_ids": 3,
                    "concept_difficulty": 33,
                    "attempts": 0.50,
                    "hints": 0.1,
                    "speed": 1.50,
                    "behavior_cluster": 3,
                },
                {
                    "user_id": 1,
                    "timestamp": 4,
                    "question_ids": 14,
                    "question_difficulty": 24,
                    "correct": 1,
                    "concept_ids_internal": 3,
                    "concept_ids": 4,
                    "concept_difficulty": 34,
                    "attempts": 0.40,
                    "hints": 0.0,
                    "speed": 1.70,
                    "behavior_cluster": 4,
                },
                {
                    "user_id": 2,
                    "timestamp": 5,
                    "question_ids": 21,
                    "question_difficulty": 41,
                    "correct": 1,
                    "concept_ids_internal": 0,
                    "concept_ids": 1,
                    "concept_difficulty": 51,
                    "attempts": 0.80,
                    "hints": 0.3,
                    "speed": 1.00,
                    "behavior_cluster": 2,
                },
                {
                    "user_id": 2,
                    "timestamp": 6,
                    "question_ids": 22,
                    "question_difficulty": 42,
                    "correct": 0,
                    "concept_ids_internal": 1,
                    "concept_ids": 2,
                    "concept_difficulty": 52,
                    "attempts": 0.90,
                    "hints": 0.2,
                    "speed": 0.90,
                    "behavior_cluster": 1,
                },
            ]
        )

        records = build_sequence_records(data_frame, user_ids=[1, 2], sequence_length=3)
        self.assertEqual(len(records), 2)
        shuffled_records = shuffle_records(records, seed=2)
        bundle = records_to_bundle(shuffled_records, sequence_length=3)

        self.assertEqual(bundle.question_ids.shape, (2, 3))
        self.assertEqual(bundle.mask.shape, (2, 3))
        self.assertTrue(np.all(bundle.mask.sum(axis=1) >= 2))
        self.assertEqual(int(bundle.behavior_cluster.max()), 3)
        self.assertEqual(bundle.speed_relative_student.shape, (2, 3))
        self.assertEqual(bundle.speed_relative_question.shape, (2, 3))
        self.assertEqual(bundle.behavior_soft_membership.shape, (2, 3, 5))


if __name__ == "__main__":
    unittest.main()
