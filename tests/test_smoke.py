import os
import sys
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ahskt.config import load_config
from ahskt.data.synthetic import generate_demo_bundles
from ahskt.models.ahs_kt import AHSKTModel
from ahskt.training.engine import evaluate, train_one_epoch


class AHSKTSmokeTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(PROJECT_ROOT / "configs" / "ahskt_demo.json", project_root=PROJECT_ROOT)
        self.train_bundle, self.valid_bundle, _ = generate_demo_bundles(self.config)

    def test_forward_shape(self):
        model = AHSKTModel(self.config.model)
        batch = next(iter(self.train_bundle.to_tf_dataset(batch_size=4, shuffle=False, seed=self.config.seed)))
        self.assertIn("speed_relative_student", batch)
        self.assertIn("speed_relative_question", batch)
        self.assertIn("behavior_soft_membership", batch)
        logits = model(batch, training=False)
        self.assertEqual(tuple(logits.shape), (4, self.config.model.sequence_length))

    def test_one_epoch_train_and_eval(self):
        model = AHSKTModel(self.config.model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.training.learning_rate)
        train_dataset = self.train_bundle.to_tf_dataset(batch_size=8, shuffle=True, seed=self.config.seed)
        valid_dataset = self.valid_bundle.to_tf_dataset(batch_size=8, shuffle=False, seed=self.config.seed)
        train_metrics = train_one_epoch(model, optimizer, train_dataset)
        valid_metrics = evaluate(model, valid_dataset)
        self.assertIn("auc", train_metrics)
        self.assertIn("auc", valid_metrics)

    def test_forward_without_difficulty_and_behavior(self):
        self.config.model.use_target_interaction = True
        self.config.model.use_difficulty_features = False
        self.config.model.use_behavior_features = False
        model = AHSKTModel(self.config.model)
        batch = next(iter(self.train_bundle.to_tf_dataset(batch_size=4, shuffle=False, seed=self.config.seed)))
        logits = model(batch, training=False)
        self.assertEqual(tuple(logits.shape), (4, self.config.model.sequence_length))

    def test_forward_with_late_residual_fusion(self):
        self.config.model.use_target_interaction = True
        self.config.model.use_difficulty_features = True
        self.config.model.use_behavior_features = True
        self.config.model.fusion_mode = "late_residual"
        self.config.model.behavior_condition_on_difficulty = False
        self.config.model.aux_residual_scale = 0.2
        model = AHSKTModel(self.config.model)
        batch = next(iter(self.train_bundle.to_tf_dataset(batch_size=4, shuffle=False, seed=self.config.seed)))
        logits = model(batch, training=False)
        self.assertEqual(tuple(logits.shape), (4, self.config.model.sequence_length))

    def test_forward_with_scalar_difficulty_bias(self):
        self.config.model.use_target_interaction = True
        self.config.model.use_difficulty_features = True
        self.config.model.use_behavior_features = True
        self.config.model.fusion_mode = "late_residual"
        self.config.model.behavior_condition_on_difficulty = False
        self.config.model.difficulty_mode = "scalar_bias"
        self.config.model.difficulty_bias_scale = 0.1
        model = AHSKTModel(self.config.model)
        batch = next(iter(self.train_bundle.to_tf_dataset(batch_size=4, shuffle=False, seed=self.config.seed)))
        logits = model(batch, training=False)
        self.assertEqual(tuple(logits.shape), (4, self.config.model.sequence_length))


if __name__ == "__main__":
    unittest.main()
