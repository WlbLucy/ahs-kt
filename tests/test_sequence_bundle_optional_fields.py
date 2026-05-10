import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ahskt.data.dataset import SequenceBundle


class SequenceBundleOptionalFieldsTest(unittest.TestCase):
    def test_optional_fields_default_to_zero(self):
        bundle = SequenceBundle(
            question_ids=np.ones((2, 3), dtype=np.int32),
            concept_ids=np.ones((2, 3), dtype=np.int32),
            responses=np.ones((2, 3), dtype=np.int32),
            question_difficulty=np.ones((2, 3), dtype=np.int32),
            concept_difficulty=np.ones((2, 3), dtype=np.int32),
            attempts=np.ones((2, 3), dtype=np.float32),
            hints=np.ones((2, 3), dtype=np.float32),
            speed=np.ones((2, 3), dtype=np.float32),
            behavior_cluster=np.ones((2, 3), dtype=np.int32),
            mask=np.ones((2, 3), dtype=np.int32),
        )
        self.assertTrue(np.allclose(bundle.question_easiness, 0.0))
        self.assertTrue(np.allclose(bundle.question_confidence, 0.0))
        self.assertTrue(np.allclose(bundle.speed_relative_student, 0.0))
        self.assertTrue(np.allclose(bundle.speed_relative_question, 0.0))
        self.assertEqual(bundle.behavior_soft_membership.shape, (2, 3, 2))
        self.assertTrue(np.allclose(bundle.behavior_soft_membership, 0.0))


if __name__ == "__main__":
    unittest.main()
