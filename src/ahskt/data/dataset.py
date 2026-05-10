from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


REQUIRED_FIELDS = [
    "question_ids",
    "concept_ids",
    "responses",
    "question_difficulty",
    "concept_difficulty",
    "attempts",
    "hints",
    "speed",
    "behavior_cluster",
    "mask",
]

OPTIONAL_FIELDS = [
    "question_easiness",
    "concept_easiness",
    "question_confidence",
    "concept_confidence",
    "speed_relative_student",
    "speed_relative_question",
    "behavior_soft_membership",
]


@dataclass
class SequenceBundle:
    question_ids: np.ndarray
    concept_ids: np.ndarray
    responses: np.ndarray
    question_difficulty: np.ndarray
    concept_difficulty: np.ndarray
    attempts: np.ndarray
    hints: np.ndarray
    speed: np.ndarray
    behavior_cluster: np.ndarray
    mask: np.ndarray
    question_easiness: np.ndarray = None
    concept_easiness: np.ndarray = None
    question_confidence: np.ndarray = None
    concept_confidence: np.ndarray = None
    speed_relative_student: np.ndarray = None
    speed_relative_question: np.ndarray = None
    behavior_soft_membership: np.ndarray = None

    def __post_init__(self):
        lengths = {field: getattr(self, field).shape for field in REQUIRED_FIELDS}
        first_shape = next(iter(lengths.values()))
        for field_name, shape in lengths.items():
            if shape != first_shape:
                raise ValueError(f"inconsistent shape for {field_name}: {shape} vs {first_shape}")
        for field_name in OPTIONAL_FIELDS:
            field_value = getattr(self, field_name)
            if field_name == "behavior_soft_membership":
                expected_shape = (
                    first_shape[0],
                    first_shape[1],
                    max(int(np.max(self.behavior_cluster)) + 1, 1),
                )
                if field_value is None:
                    setattr(self, field_name, np.zeros(expected_shape, dtype=np.float32))
                    continue
                if field_value.shape[:2] != first_shape:
                    raise ValueError(
                        f"inconsistent shape for {field_name}: {field_value.shape} vs {expected_shape}"
                    )
                continue
            if field_value is None:
                setattr(self, field_name, np.zeros(first_shape, dtype=np.float32))
                continue
            if field_value.shape != first_shape:
                raise ValueError(f"inconsistent shape for {field_name}: {field_value.shape} vs {first_shape}")

    @property
    def num_samples(self):
        return int(self.question_ids.shape[0])

    @property
    def sequence_length(self):
        return int(self.question_ids.shape[1])

    def as_dict(self):
        return {
            "question_ids": self.question_ids.astype(np.int32),
            "concept_ids": self.concept_ids.astype(np.int32),
            "responses": self.responses.astype(np.int32),
            "question_difficulty": self.question_difficulty.astype(np.int32),
            "concept_difficulty": self.concept_difficulty.astype(np.int32),
            "attempts": self.attempts.astype(np.float32),
            "hints": self.hints.astype(np.float32),
            "speed": self.speed.astype(np.float32),
            "behavior_cluster": self.behavior_cluster.astype(np.int32),
            "mask": self.mask.astype(np.int32),
            "question_easiness": self.question_easiness.astype(np.float32),
            "concept_easiness": self.concept_easiness.astype(np.float32),
            "question_confidence": self.question_confidence.astype(np.float32),
            "concept_confidence": self.concept_confidence.astype(np.float32),
            "speed_relative_student": self.speed_relative_student.astype(np.float32),
            "speed_relative_question": self.speed_relative_question.astype(np.float32),
            "behavior_soft_membership": self.behavior_soft_membership.astype(np.float32),
        }

    def to_tf_dataset(self, batch_size, shuffle=False, seed=42):
        dataset = tf.data.Dataset.from_tensor_slices(self.as_dict())
        if shuffle:
            dataset = dataset.shuffle(self.num_samples, seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    @classmethod
    def from_npz(cls, file_path):
        file_path = Path(file_path)
        payload = np.load(file_path, allow_pickle=False)
        missing = [field for field in REQUIRED_FIELDS if field not in payload]
        if missing:
            raise KeyError(f"missing fields in {file_path}: {missing}")
        init_payload = {field: payload[field] for field in REQUIRED_FIELDS}
        for field in OPTIONAL_FIELDS:
            if field in payload:
                init_payload[field] = payload[field]
        return cls(**init_payload)


def load_bundle_from_config(config):
    train_bundle = SequenceBundle.from_npz(config.project_root / config.dataset.train_path)
    valid_bundle = SequenceBundle.from_npz(config.project_root / config.dataset.valid_path)
    test_bundle = SequenceBundle.from_npz(config.project_root / config.dataset.test_path)
    return train_bundle, valid_bundle, test_bundle
