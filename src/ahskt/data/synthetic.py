import numpy as np

from .dataset import SequenceBundle


def _soft_membership_from_cluster(cluster_ids, num_behavior_clusters, smoothing=0.15):
    membership = np.full(
        cluster_ids.shape + (num_behavior_clusters,),
        fill_value=0.0,
        dtype=np.float32,
    )
    if num_behavior_clusters <= 1:
        membership[..., 0] = 1.0
        return membership
    residual = smoothing / max(num_behavior_clusters - 2, 1)
    for cluster_id in range(1, num_behavior_clusters):
        membership[..., cluster_id] = np.where(cluster_ids == cluster_id, 1.0 - smoothing, residual)
    membership[..., 0] = 0.0
    return membership


def _make_behavior_cluster(attempts, hints, speed):
    cluster = np.ones_like(attempts, dtype=np.int32)
    cluster[(attempts <= 1.3) & (speed <= 2.4) & (hints <= 0.2)] = 2
    cluster[(attempts <= 1.3) & (speed > 2.4)] = 3
    cluster[(attempts > 1.3) & (speed > 2.4) & (hints > 0.7)] = 4
    return cluster


def _generate_bundle(num_samples, config, seed):
    rng = np.random.default_rng(seed)
    sequence_length = config.model.sequence_length

    question_ids = rng.integers(1, config.model.num_questions + 1, size=(num_samples, sequence_length), dtype=np.int32)
    concept_ids = rng.integers(1, config.model.num_concepts + 1, size=(num_samples, sequence_length), dtype=np.int32)
    question_difficulty = rng.integers(1, config.model.num_question_difficulty, size=(num_samples, sequence_length), dtype=np.int32)
    concept_difficulty = rng.integers(1, config.model.num_concept_difficulty, size=(num_samples, sequence_length), dtype=np.int32)

    student_ability = rng.normal(loc=0.0, scale=1.0, size=(num_samples, 1)).astype(np.float32)
    question_difficulty_score = (question_difficulty / max(1, config.model.num_question_difficulty - 1)).astype(np.float32)
    concept_difficulty_score = (concept_difficulty / max(1, config.model.num_concept_difficulty - 1)).astype(np.float32)
    combined_difficulty = 0.6 * question_difficulty_score + 0.4 * concept_difficulty_score

    attempts = np.clip(1.0 + 2.5 * combined_difficulty - 0.8 * student_ability + rng.normal(0.0, 0.25, size=(num_samples, sequence_length)), 0.0, 8.0).astype(np.float32)
    hints = np.clip(0.15 + 1.8 * combined_difficulty - 0.6 * student_ability + rng.normal(0.0, 0.18, size=(num_samples, sequence_length)), 0.0, 4.0).astype(np.float32)
    speed = np.clip(4.0 + 2.2 * student_ability - 2.0 * combined_difficulty + rng.normal(0.0, 0.35, size=(num_samples, sequence_length)), 0.1, 10.0).astype(np.float32)

    behavior_cluster = _make_behavior_cluster(attempts, hints, speed)
    if config.model.num_behavior_clusters > 1:
        behavior_cluster = np.clip(behavior_cluster, 0, config.model.num_behavior_clusters - 1).astype(np.int32)
    speed_relative_student = speed - np.mean(speed, axis=1, keepdims=True).astype(np.float32)
    speed_relative_question = speed - np.mean(speed, axis=0, keepdims=True).astype(np.float32)
    behavior_soft_membership = _soft_membership_from_cluster(
        behavior_cluster,
        num_behavior_clusters=config.model.num_behavior_clusters,
    )

    logits = (
        1.1 * student_ability
        - 1.5 * combined_difficulty
        - 0.18 * attempts
        - 0.15 * hints
        + 0.12 * speed
        + 0.08 * speed_relative_student
        + 0.05 * speed_relative_question
        + rng.normal(0.0, 0.25, size=(num_samples, sequence_length))
    )
    probability = 1.0 / (1.0 + np.exp(-logits))
    responses = rng.binomial(1, probability).astype(np.int32)

    mask = np.ones((num_samples, sequence_length), dtype=np.int32)

    return SequenceBundle(
        question_ids=question_ids,
        concept_ids=concept_ids,
        responses=responses,
        question_difficulty=question_difficulty,
        concept_difficulty=concept_difficulty,
        attempts=attempts,
        hints=hints,
        speed=speed,
        speed_relative_student=speed_relative_student.astype(np.float32),
        speed_relative_question=speed_relative_question.astype(np.float32),
        behavior_cluster=behavior_cluster,
        behavior_soft_membership=behavior_soft_membership,
        mask=mask,
    )


def generate_demo_bundles(config):
    train_bundle = _generate_bundle(config.demo.train_size, config, config.seed)
    valid_bundle = _generate_bundle(config.demo.valid_size, config, config.seed + 1)
    test_bundle = _generate_bundle(config.demo.test_size, config, config.seed + 2)
    return train_bundle, valid_bundle, test_bundle
