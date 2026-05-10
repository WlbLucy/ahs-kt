import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from .dataset import SequenceBundle


def load_json(file_path):
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


@dataclass
class Assist2009Assets:
    train_records: dict
    valid_records: dict
    test_records: dict
    topic2index: dict
    kc2index: dict
    time_factor: dict
    attempts_factor: dict
    hints_factor: dict


@dataclass
class Assist2009BehaviorClusterAssets:
    scaler: StandardScaler
    cluster_model: MiniBatchKMeans
    cluster_mapping: dict
    centers: list
    num_behavior_clusters: int
    question_speed_reference: dict
    global_speed_reference: float


def load_assist2009_assets(data_dir):
    data_dir = Path(data_dir)
    return Assist2009Assets(
        train_records=load_json(data_dir / "new_train_data.json"),
        valid_records=load_json(data_dir / "new_valid_data.json"),
        test_records=load_json(data_dir / "new_test_data.json"),
        topic2index=load_json(data_dir / "topic2index.json"),
        kc2index=load_json(data_dir / "kc2index.json"),
        time_factor=load_json(data_dir / "time_factor.json"),
        attempts_factor=load_json(data_dir / "attempts_factor.json"),
        hints_factor=load_json(data_dir / "hints_factor.json"),
    )


def _kc_key(kc_value):
    if float(kc_value).is_integer():
        return str(int(kc_value))
    return str(kc_value)


def compute_difficulty_maps(train_records):
    return compute_difficulty_maps_with_smoothing(train_records=train_records)


def compute_difficulty_maps_with_smoothing(train_records, question_alpha=0.0, concept_alpha=0.0):
    topic_targets = {}
    kc_targets = {}

    for sequence in train_records.values():
        for topic_id, kc_value, correctness, *_ in sequence:
            topic_id = int(topic_id)
            kc_key = _kc_key(kc_value)
            topic_targets.setdefault(topic_id, []).append(int(correctness))
            kc_targets.setdefault(kc_key, []).append(int(correctness))

    global_targets = [target for values in topic_targets.values() for target in values]
    global_mean = float(np.mean(global_targets)) if global_targets else 0.5
    default_bin = int(global_mean * 100) + 1

    max_topic_count = max((len(targets) for targets in topic_targets.values()), default=1)
    max_kc_count = max((len(targets) for targets in kc_targets.values()), default=1)

    topic_difficulty = {}
    kc_difficulty = {}
    topic_easiness = {}
    kc_easiness = {}
    topic_confidence = {}
    kc_confidence = {}

    for topic_id, targets in topic_targets.items():
        count = len(targets)
        posterior_mean = float((np.sum(targets) + question_alpha * global_mean) / (count + question_alpha))
        topic_difficulty[topic_id] = int(np.clip(int(posterior_mean * 100) + 1, 1, 101))
        topic_easiness[topic_id] = posterior_mean
        topic_confidence[topic_id] = float(count / (count + question_alpha)) if question_alpha > 0 else 1.0

    for kc_key, targets in kc_targets.items():
        count = len(targets)
        posterior_mean = float((np.sum(targets) + concept_alpha * global_mean) / (count + concept_alpha))
        kc_difficulty[kc_key] = int(np.clip(int(posterior_mean * 100) + 1, 1, 101))
        kc_easiness[kc_key] = posterior_mean
        kc_confidence[kc_key] = float(count / (count + concept_alpha)) if concept_alpha > 0 else 1.0

    return (
        topic_difficulty,
        kc_difficulty,
        default_bin,
        topic_easiness,
        kc_easiness,
        topic_confidence,
        kc_confidence,
        {
            "global_easiness": global_mean,
            "question_alpha": float(question_alpha),
            "concept_alpha": float(concept_alpha),
            "max_question_support": int(max_topic_count),
            "max_concept_support": int(max_kc_count),
        },
    )


def _allocate_arrays(num_sequences, sequence_length):
    return {
        "question_ids": np.zeros((num_sequences, sequence_length), dtype=np.int32),
        "concept_ids": np.zeros((num_sequences, sequence_length), dtype=np.int32),
        "responses": np.zeros((num_sequences, sequence_length), dtype=np.int32),
        "question_difficulty": np.zeros((num_sequences, sequence_length), dtype=np.int32),
        "concept_difficulty": np.zeros((num_sequences, sequence_length), dtype=np.int32),
        "attempts": np.zeros((num_sequences, sequence_length), dtype=np.float32),
        "hints": np.zeros((num_sequences, sequence_length), dtype=np.float32),
        "speed": np.zeros((num_sequences, sequence_length), dtype=np.float32),
        "speed_relative_student": np.zeros((num_sequences, sequence_length), dtype=np.float32),
        "speed_relative_question": np.zeros((num_sequences, sequence_length), dtype=np.float32),
        "behavior_cluster": np.zeros((num_sequences, sequence_length), dtype=np.int32),
        "behavior_soft_membership": None,
        "mask": np.zeros((num_sequences, sequence_length), dtype=np.int32),
        "question_easiness": np.zeros((num_sequences, sequence_length), dtype=np.float32),
        "concept_easiness": np.zeros((num_sequences, sequence_length), dtype=np.float32),
        "question_confidence": np.zeros((num_sequences, sequence_length), dtype=np.float32),
        "concept_confidence": np.zeros((num_sequences, sequence_length), dtype=np.float32),
    }


def _extract_behavior_feature_rows(records, time_factor, attempts_factor, hints_factor):
    features = []
    question_speed_reference = {}

    for user_id in sorted(records, key=lambda value: int(value)):
        sequence = records[user_id]
        if not sequence:
            continue
        speed = [float(time_factor[str(user_id)][idx]) for idx in range(len(sequence))]
        attempts = [float(attempts_factor[str(user_id)][idx]) for idx in range(len(sequence))]
        hints = [float(hints_factor[str(user_id)][idx]) for idx in range(len(sequence))]
        for idx, item in enumerate(sequence):
            topic_id = int(item[0])
            features.append([attempts[idx], hints[idx], speed[idx]])
            question_speed_reference.setdefault(topic_id, []).append(speed[idx])
    return features, question_speed_reference


def _build_cluster_mapping(raw_centers):
    order = sorted(
        range(len(raw_centers)),
        key=lambda index: (
            float(raw_centers[index, 0]),
            float(raw_centers[index, 1]),
            float(-raw_centers[index, 2]),
        ),
    )
    return {int(old_label): int(new_label + 1) for new_label, old_label in enumerate(order)}


def fit_behavior_clusters(
    train_records,
    time_factor,
    attempts_factor,
    hints_factor,
    n_clusters=4,
    random_seed=2026,
    sample_size=0,
):
    features, question_speed_reference = _extract_behavior_feature_rows(
        records=train_records,
        time_factor=time_factor,
        attempts_factor=attempts_factor,
        hints_factor=hints_factor,
    )

    if not features:
        scaler = StandardScaler()
        scaler.fit(np.zeros((1, 3), dtype=np.float32))
        cluster_model = MiniBatchKMeans(
            n_clusters=1,
            random_state=random_seed,
            n_init=1,
            batch_size=1,
        )
        cluster_model.fit(np.zeros((1, 3), dtype=np.float32))
        return Assist2009BehaviorClusterAssets(
            scaler=scaler,
            cluster_model=cluster_model,
            cluster_mapping={0: 1},
            centers=[
                {
                    "cluster_id": 1,
                    "center_attempts": 0.0,
                    "center_hints": 0.0,
                    "center_speed": 0.0,
                }
            ],
            num_behavior_clusters=1,
            question_speed_reference={},
            global_speed_reference=0.0,
        )

    feature_array = np.asarray(features, dtype=np.float32)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_array)
    effective_clusters = max(1, min(int(n_clusters), len(scaled_features)))

    if sample_size and len(scaled_features) > int(sample_size):
        generator = np.random.default_rng(random_seed)
        sample_indices = generator.choice(len(scaled_features), size=int(sample_size), replace=False)
        fit_features = scaled_features[sample_indices]
    else:
        fit_features = scaled_features

    cluster_model = MiniBatchKMeans(
        n_clusters=effective_clusters,
        random_state=random_seed,
        n_init=20,
        batch_size=4096,
    )
    cluster_model.fit(fit_features)
    raw_centers = scaler.inverse_transform(cluster_model.cluster_centers_)
    cluster_mapping = _build_cluster_mapping(raw_centers)
    centers = []
    for raw_label, center in enumerate(raw_centers):
        centers.append(
            {
                "cluster_id": int(cluster_mapping[int(raw_label)]),
                "center_attempts": float(center[0]),
                "center_hints": float(center[1]),
                "center_speed": float(center[2]),
            }
        )
    centers.sort(key=lambda item: item["cluster_id"])

    return Assist2009BehaviorClusterAssets(
        scaler=scaler,
        cluster_model=cluster_model,
        cluster_mapping=cluster_mapping,
        centers=centers,
        num_behavior_clusters=effective_clusters + 1,
        question_speed_reference={
            int(topic_id): float(np.median(values)) for topic_id, values in question_speed_reference.items()
        },
        global_speed_reference=float(np.median(feature_array[:, 2])),
    )


def compute_behavior_soft_membership(features, behavior_assets):
    if behavior_assets.num_behavior_clusters <= 1:
        return np.zeros((len(features), 1), dtype=np.float32)

    scaled_features = behavior_assets.scaler.transform(features.astype(np.float32))
    centers = behavior_assets.cluster_model.cluster_centers_
    squared_distance = np.sum(
        (scaled_features[:, None, :] - centers[None, :, :]) ** 2,
        axis=-1,
    ).astype(np.float32)
    logits = -squared_distance
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights_without_padding = np.exp(logits)
    weights_without_padding /= np.maximum(np.sum(weights_without_padding, axis=-1, keepdims=True), 1e-8)
    weights = np.zeros((len(features), behavior_assets.num_behavior_clusters), dtype=np.float32)
    for raw_label in range(weights_without_padding.shape[1]):
        cluster_id = int(behavior_assets.cluster_mapping[int(raw_label)])
        weights[:, cluster_id] = weights_without_padding[:, raw_label]
    return weights


def _count_sequences(records, sequence_length, remainder_min_len):
    count = 0
    for sequence in records.values():
        count += len(sequence) // sequence_length
        if len(sequence) % sequence_length >= remainder_min_len:
            count += 1
    return count


def build_split_bundle(
    records,
    topic2index,
    kc2index,
    time_factor,
    attempts_factor,
    hints_factor,
    topic_difficulty,
    kc_difficulty,
    default_difficulty,
    topic_easiness=None,
    kc_easiness=None,
    default_easiness=0.5,
    topic_confidence=None,
    kc_confidence=None,
    default_confidence=0.0,
    sequence_length=100,
    remainder_min_len=10,
    behavior_cluster_assets=None,
):
    if behavior_cluster_assets is None:
        behavior_cluster_assets = fit_behavior_clusters(
            train_records=records,
            time_factor=time_factor,
            attempts_factor=attempts_factor,
            hints_factor=hints_factor,
        )

    topic_easiness = topic_easiness or {}
    kc_easiness = kc_easiness or {}
    topic_confidence = topic_confidence or {}
    kc_confidence = kc_confidence or {}
    num_sequences = _count_sequences(records, sequence_length, remainder_min_len)
    payload = _allocate_arrays(num_sequences, sequence_length)

    row_index = 0
    for user_id in sorted(records, key=lambda value: int(value)):
        sequence = records[user_id]
        if not sequence:
            continue

        topics = [int(topic2index[str(int(item[0]))]) + 1 for item in sequence]
        concepts = [int(kc2index[_kc_key(item[1])]) + 1 for item in sequence]
        responses = [int(item[2]) for item in sequence]
        topic_diff = [int(topic_difficulty.get(int(item[0]), default_difficulty)) for item in sequence]
        concept_diff = [int(kc_difficulty.get(_kc_key(item[1]), default_difficulty)) for item in sequence]
        topic_ease = [float(topic_easiness.get(int(item[0]), default_easiness)) for item in sequence]
        concept_ease = [float(kc_easiness.get(_kc_key(item[1]), default_easiness)) for item in sequence]
        topic_conf = [float(topic_confidence.get(int(item[0]), default_confidence)) for item in sequence]
        concept_conf = [float(kc_confidence.get(_kc_key(item[1]), default_confidence)) for item in sequence]
        speed = [float(time_factor[str(user_id)][idx]) for idx in range(len(sequence))]
        attempts = [float(attempts_factor[str(user_id)][idx]) for idx in range(len(sequence))]
        hints = [float(hints_factor[str(user_id)][idx]) for idx in range(len(sequence))]
        speed_relative_student = []
        running_speed_sum = 0.0
        for index, current_speed in enumerate(speed):
            student_baseline = (
                running_speed_sum / index if index > 0 else float(behavior_cluster_assets.global_speed_reference)
            )
            speed_relative_student.append(float(current_speed - student_baseline))
            running_speed_sum += current_speed
        speed_relative_question = [
            float(current_speed - behavior_cluster_assets.question_speed_reference.get(int(item[0]), behavior_cluster_assets.global_speed_reference))
            for current_speed, item in zip(speed, sequence)
        ]
        feature_array = np.stack([attempts, hints, speed], axis=-1).astype(np.float32)
        scaled_features = behavior_cluster_assets.scaler.transform(feature_array)
        raw_labels = behavior_cluster_assets.cluster_model.predict(scaled_features)
        clusters = [
            int(behavior_cluster_assets.cluster_mapping[int(raw_label)])
            for raw_label in raw_labels
        ]
        soft_membership = compute_behavior_soft_membership(feature_array, behavior_cluster_assets)

        full_chunks = len(sequence) // sequence_length
        for chunk_index in range(full_chunks):
            begin = chunk_index * sequence_length
            end = (chunk_index + 1) * sequence_length
            for key, values in [
                ("question_ids", topics),
                ("concept_ids", concepts),
                ("responses", responses),
                ("question_difficulty", topic_diff),
                ("concept_difficulty", concept_diff),
                ("attempts", attempts),
                ("hints", hints),
                ("speed", speed),
                ("speed_relative_student", speed_relative_student),
                ("speed_relative_question", speed_relative_question),
                ("behavior_cluster", clusters),
                ("question_easiness", topic_ease),
                ("concept_easiness", concept_ease),
                ("question_confidence", topic_conf),
                ("concept_confidence", concept_conf),
            ]:
                payload[key][row_index, :] = values[begin:end]
            if payload["behavior_soft_membership"] is None:
                payload["behavior_soft_membership"] = np.zeros(
                    (num_sequences, sequence_length, behavior_cluster_assets.num_behavior_clusters),
                    dtype=np.float32,
                )
            payload["behavior_soft_membership"][row_index, :, :] = soft_membership[begin:end]
            payload["mask"][row_index, :] = 1
            row_index += 1

        left = len(sequence) % sequence_length
        if left < remainder_min_len:
            continue
        begin = full_chunks * sequence_length
        end = len(sequence)
        for key, values in [
            ("question_ids", topics),
            ("concept_ids", concepts),
            ("responses", responses),
            ("question_difficulty", topic_diff),
            ("concept_difficulty", concept_diff),
            ("attempts", attempts),
            ("hints", hints),
            ("speed", speed),
            ("speed_relative_student", speed_relative_student),
            ("speed_relative_question", speed_relative_question),
            ("behavior_cluster", clusters),
            ("question_easiness", topic_ease),
            ("concept_easiness", concept_ease),
            ("question_confidence", topic_conf),
            ("concept_confidence", concept_conf),
        ]:
            payload[key][row_index, :left] = values[begin:end]
        if payload["behavior_soft_membership"] is None:
            payload["behavior_soft_membership"] = np.zeros(
                (num_sequences, sequence_length, behavior_cluster_assets.num_behavior_clusters),
                dtype=np.float32,
            )
        payload["behavior_soft_membership"][row_index, :left, :] = soft_membership[begin:end]
        payload["mask"][row_index, :left] = 1
        row_index += 1

    return SequenceBundle(**payload)


def build_assist2009_bundles(
    data_dir,
    sequence_length=100,
    remainder_min_len=10,
    question_alpha=0.0,
    concept_alpha=0.0,
    n_clusters=4,
    random_seed=2026,
    cluster_sample_size=0,
):
    assets = load_assist2009_assets(data_dir)
    (
        topic_difficulty,
        kc_difficulty,
        default_difficulty,
        topic_easiness,
        kc_easiness,
        topic_confidence,
        kc_confidence,
        smoothing_metadata,
    ) = compute_difficulty_maps_with_smoothing(
        train_records=assets.train_records,
        question_alpha=question_alpha,
        concept_alpha=concept_alpha,
    )
    behavior_cluster_assets = fit_behavior_clusters(
        train_records=assets.train_records,
        time_factor=assets.time_factor,
        attempts_factor=assets.attempts_factor,
        hints_factor=assets.hints_factor,
        n_clusters=n_clusters,
        random_seed=random_seed,
        sample_size=cluster_sample_size,
    )

    train_bundle = build_split_bundle(
        records=assets.train_records,
        topic2index=assets.topic2index,
        kc2index=assets.kc2index,
        time_factor=assets.time_factor,
        attempts_factor=assets.attempts_factor,
        hints_factor=assets.hints_factor,
        topic_difficulty=topic_difficulty,
        kc_difficulty=kc_difficulty,
        default_difficulty=default_difficulty,
        topic_easiness=topic_easiness,
        kc_easiness=kc_easiness,
        default_easiness=smoothing_metadata["global_easiness"],
        topic_confidence=topic_confidence,
        kc_confidence=kc_confidence,
        default_confidence=0.0,
        sequence_length=sequence_length,
        remainder_min_len=remainder_min_len,
        behavior_cluster_assets=behavior_cluster_assets,
    )
    valid_bundle = build_split_bundle(
        records=assets.valid_records,
        topic2index=assets.topic2index,
        kc2index=assets.kc2index,
        time_factor=assets.time_factor,
        attempts_factor=assets.attempts_factor,
        hints_factor=assets.hints_factor,
        topic_difficulty=topic_difficulty,
        kc_difficulty=kc_difficulty,
        default_difficulty=default_difficulty,
        topic_easiness=topic_easiness,
        kc_easiness=kc_easiness,
        default_easiness=smoothing_metadata["global_easiness"],
        topic_confidence=topic_confidence,
        kc_confidence=kc_confidence,
        default_confidence=0.0,
        sequence_length=sequence_length,
        remainder_min_len=remainder_min_len,
        behavior_cluster_assets=behavior_cluster_assets,
    )
    test_bundle = build_split_bundle(
        records=assets.test_records,
        topic2index=assets.topic2index,
        kc2index=assets.kc2index,
        time_factor=assets.time_factor,
        attempts_factor=assets.attempts_factor,
        hints_factor=assets.hints_factor,
        topic_difficulty=topic_difficulty,
        kc_difficulty=kc_difficulty,
        default_difficulty=default_difficulty,
        topic_easiness=topic_easiness,
        kc_easiness=kc_easiness,
        default_easiness=smoothing_metadata["global_easiness"],
        topic_confidence=topic_confidence,
        kc_confidence=kc_confidence,
        default_confidence=0.0,
        sequence_length=sequence_length,
        remainder_min_len=remainder_min_len,
        behavior_cluster_assets=behavior_cluster_assets,
    )

    metadata = {
        "dataset_name": "assist2009",
        "sequence_length": int(sequence_length),
        "remainder_min_len": int(remainder_min_len),
        "num_questions": int(max(assets.topic2index.values()) + 1),
        "num_concepts": int(max(assets.kc2index.values()) + 1),
        "num_question_difficulty": int(max(topic_difficulty.values())),
        "num_concept_difficulty": int(max(kc_difficulty.values())),
        "num_behavior_clusters": int(behavior_cluster_assets.num_behavior_clusters),
        "split_summary": {
            "train_users": int(len(assets.train_records)),
            "valid_users": int(len(assets.valid_records)),
            "test_users": int(len(assets.test_records)),
            "train_sequences": int(train_bundle.num_samples),
            "valid_sequences": int(valid_bundle.num_samples),
            "test_sequences": int(test_bundle.num_samples),
        },
        "default_difficulty": int(default_difficulty),
        "smoothing": smoothing_metadata,
        "behavior": {
            "n_real_clusters": int(len(behavior_cluster_assets.centers)),
            "cluster_centers": behavior_cluster_assets.centers,
            "cluster_random_seed": int(random_seed),
            "cluster_sample_size": int(cluster_sample_size),
        },
        "global_speed_reference": float(behavior_cluster_assets.global_speed_reference),
    }
    return train_bundle, valid_bundle, test_bundle, metadata


def save_bundle_npz(bundle, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **bundle.as_dict())
