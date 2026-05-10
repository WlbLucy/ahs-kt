import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .dataset import SequenceBundle


RAW_COLUMNS = [
    "user_id",
    "problem_id",
    "correct",
    "skill_id",
    "end_time",
    "attempt_count",
    "hint_count",
    "ms_first_response",
]


@dataclass
class BehaviorClusterBundle:
    clip_values: dict
    scaler: StandardScaler
    cluster_model: MiniBatchKMeans
    cluster_mapping: dict
    centers: list
    question_speed_reference: dict
    global_speed_reference: float


def load_python_dict(file_path):
    return ast.literal_eval(Path(file_path).read_text(encoding="utf-8"))


def load_dimkt_reference_assets(dimkt_data_dir):
    dimkt_data_dir = Path(dimkt_data_dir)
    return {
        "problem2id": {int(key): int(value) for key, value in load_python_dict(dimkt_data_dir / "problem2id").items()},
        "skill2id": {float(key): int(value) for key, value in load_python_dict(dimkt_data_dir / "skill2id").items()},
        "difficult2id": {int(key): int(value) for key, value in load_python_dict(dimkt_data_dir / "difficult2id").items()},
        "sdifficult2id": {float(key): int(value) for key, value in load_python_dict(dimkt_data_dir / "sdifficult2id").items()},
        "nones": {int(value) for value in np.load(dimkt_data_dir / "nones.npy", allow_pickle=True).tolist()},
        "nonesk": {float(value) for value in np.load(dimkt_data_dir / "nonesk.npy", allow_pickle=True).tolist()},
    }


def load_assist2012_interactions(csv_path):
    data_frame = pd.read_csv(
        csv_path,
        encoding="ISO-8859-1",
        usecols=RAW_COLUMNS,
        low_memory=False,
        dtype={
            "user_id": "int64",
            "problem_id": "int64",
            "correct": "float32",
            "skill_id": "float64",
            "attempt_count": "float32",
            "hint_count": "float32",
            "ms_first_response": "float32",
        },
    )
    data_frame = data_frame[data_frame["skill_id"].notna()].copy()
    data_frame["correct"] = pd.to_numeric(data_frame["correct"], errors="coerce").fillna(0.0).clip(0.0, 1.0).round().astype(np.int32)
    data_frame["attempt_count"] = pd.to_numeric(data_frame["attempt_count"], errors="coerce").fillna(1.0).clip(lower=1.0).astype(np.float32)
    data_frame["hint_count"] = pd.to_numeric(data_frame["hint_count"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(np.float32)
    median_time = float(pd.to_numeric(data_frame["ms_first_response"], errors="coerce").dropna().median())
    if not np.isfinite(median_time) or median_time <= 0:
        median_time = 30000.0
    data_frame["ms_first_response"] = pd.to_numeric(data_frame["ms_first_response"], errors="coerce").fillna(median_time).clip(lower=1000.0).astype(np.float32)
    data_frame["timestamp"] = pd.to_datetime(data_frame["end_time"].astype(str).str.slice(0, 19), errors="coerce")
    data_frame = data_frame[data_frame["timestamp"].notna()].copy()
    data_frame["timestamp"] = (data_frame["timestamp"].astype("int64") // 10**9).astype(np.int64)
    return data_frame


def attach_dimkt_fields(data_frame, reference_assets):
    mapped = data_frame.copy()
    mapped = mapped[~mapped["problem_id"].isin(reference_assets["nones"])].copy()
    mapped = mapped[~mapped["skill_id"].isin(reference_assets["nonesk"])].copy()
    mapped["question_ids"] = mapped["problem_id"].map(reference_assets["problem2id"])
    mapped["concept_ids_internal"] = mapped["skill_id"].map(reference_assets["skill2id"])
    mapped["question_difficulty"] = mapped["problem_id"].map(reference_assets["difficult2id"])
    mapped["concept_difficulty"] = mapped["skill_id"].map(reference_assets["sdifficult2id"])
    mapped = mapped.dropna(
        subset=[
            "question_ids",
            "concept_ids_internal",
            "question_difficulty",
            "concept_difficulty",
        ]
    ).copy()
    mapped["question_ids"] = mapped["question_ids"].astype(np.int32)
    mapped["concept_ids_internal"] = mapped["concept_ids_internal"].astype(np.int32)
    mapped["concept_ids"] = (mapped["concept_ids_internal"] + 1).astype(np.int32)
    mapped["question_difficulty"] = mapped["question_difficulty"].astype(np.int32)
    mapped["concept_difficulty"] = mapped["concept_difficulty"].astype(np.int32)
    return mapped


def split_user_ids(user_ids):
    user_ids = np.asarray(sorted({int(user_id) for user_id in user_ids}), dtype=np.int64)
    random_state = np.random.RandomState(100)
    random_state.shuffle(user_ids)
    train_all_ids, test_ids = train_test_split(user_ids, test_size=0.2, random_state=5)
    train_all_ids = np.asarray(train_all_ids, dtype=np.int64)
    kfold = KFold(n_splits=5, shuffle=True, random_state=5)
    train_index, valid_index = next(kfold.split(train_all_ids))
    train_ids = train_all_ids[train_index].copy()
    valid_ids = train_all_ids[valid_index].copy()
    random_state.shuffle(train_ids)
    return train_ids, valid_ids, np.asarray(test_ids, dtype=np.int64)


def compute_behavior_columns(data_frame, clip_values=None):
    attempts_raw = data_frame["attempt_count"].to_numpy(dtype=np.float32)
    hints_raw = data_frame["hint_count"].to_numpy(dtype=np.float32)
    speed_raw = (60000.0 / np.maximum(data_frame["ms_first_response"].to_numpy(dtype=np.float32), 1000.0)).astype(np.float32)

    if clip_values is None:
        clip_values = {
            "attempts": float(np.quantile(attempts_raw, 0.99)),
            "hints": float(np.quantile(hints_raw, 0.99)),
            "speed": float(np.quantile(speed_raw, 0.99)),
        }

    attempts = np.log1p(np.clip(attempts_raw, 0.0, clip_values["attempts"])).astype(np.float32)
    hints = np.log1p(np.clip(hints_raw, 0.0, clip_values["hints"])).astype(np.float32)
    speed = np.log1p(np.clip(speed_raw, 0.0, clip_values["speed"])).astype(np.float32)
    features = np.stack([attempts, hints, speed], axis=-1)
    return attempts, hints, speed, clip_values, features


def _raw_speed_qpm(data_frame):
    return (60000.0 / np.maximum(data_frame["ms_first_response"].to_numpy(dtype=np.float32), 1000.0)).astype(np.float32)


def _question_speed_reference(train_frame):
    speed_raw = _raw_speed_qpm(train_frame)
    speed_frame = pd.DataFrame(
        {
            "problem_id": train_frame["problem_id"].to_numpy(dtype=np.int64),
            "speed_raw": speed_raw,
        }
    )
    grouped = speed_frame.groupby("problem_id")["speed_raw"].median()
    return {int(problem_id): float(value) for problem_id, value in grouped.items()}, float(np.median(speed_raw))


def _student_speed_baseline(data_frame, fallback_speed):
    base_frame = pd.DataFrame(
        {
            "user_id": data_frame["user_id"].to_numpy(dtype=np.int64),
            "timestamp": data_frame["timestamp"].to_numpy(dtype=np.int64),
            "speed_raw": _raw_speed_qpm(data_frame),
            "row_id": np.arange(len(data_frame), dtype=np.int64),
        }
    )
    base_frame = base_frame.sort_values(["user_id", "timestamp", "row_id"]).copy()
    counts = base_frame.groupby("user_id").cumcount()
    cumulative_speed = base_frame.groupby("user_id")["speed_raw"].cumsum() - base_frame["speed_raw"]
    prior_speed = cumulative_speed / counts.replace(0, np.nan)
    base_frame["student_speed_baseline"] = prior_speed.fillna(fallback_speed).astype(np.float32)
    base_frame = base_frame.sort_values("row_id")
    return base_frame["student_speed_baseline"].to_numpy(dtype=np.float32)


def _soft_membership(scaled_features, behavior_bundle):
    squared_distance = np.sum(
        (scaled_features[:, None, :] - behavior_bundle.cluster_model.cluster_centers_[None, :, :]) ** 2,
        axis=-1,
    ).astype(np.float32)
    raw_logits = -squared_distance
    raw_logits -= np.max(raw_logits, axis=-1, keepdims=True)
    raw_weights = np.exp(raw_logits)
    raw_weights /= np.maximum(np.sum(raw_weights, axis=-1, keepdims=True), 1e-8)

    num_rows = scaled_features.shape[0]
    weights = np.zeros((num_rows, len(behavior_bundle.centers) + 1), dtype=np.float32)
    for raw_label in range(raw_weights.shape[1]):
        cluster_id = behavior_bundle.cluster_mapping[int(raw_label)]
        weights[:, cluster_id] = raw_weights[:, raw_label]
    return weights


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


def fit_behavior_clusters(train_frame, n_clusters, random_seed, sample_size):
    _, _, _, clip_values, transformed_features = compute_behavior_columns(train_frame)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(transformed_features)

    if sample_size and len(scaled_features) > sample_size:
        generator = np.random.default_rng(random_seed)
        sample_indices = generator.choice(len(scaled_features), size=sample_size, replace=False)
        fit_features = scaled_features[sample_indices]
    else:
        fit_features = scaled_features

    cluster_model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        n_init=20,
        batch_size=4096,
    )
    cluster_model.fit(fit_features)
    raw_centers = scaler.inverse_transform(cluster_model.cluster_centers_)
    cluster_mapping = _build_cluster_mapping(raw_centers)
    centers = []
    for old_label, center in enumerate(raw_centers):
        cluster_id = cluster_mapping[int(old_label)]
        centers.append(
            {
                "cluster_id": int(cluster_id),
                "center_attempts_log": float(center[0]),
                "center_hints_log": float(center[1]),
                "center_speed_log": float(center[2]),
                "center_attempts_raw": float(np.expm1(center[0])),
                "center_hints_raw": float(np.expm1(center[1])),
                "center_speed_qpm_raw": float(np.expm1(center[2])),
            }
        )
    centers.sort(key=lambda item: item["cluster_id"])
    question_speed_reference, global_speed_reference = _question_speed_reference(train_frame)
    return BehaviorClusterBundle(
        clip_values=clip_values,
        scaler=scaler,
        cluster_model=cluster_model,
        cluster_mapping=cluster_mapping,
        centers=centers,
        question_speed_reference=question_speed_reference,
        global_speed_reference=global_speed_reference,
    )


def attach_behavior_features(data_frame, behavior_bundle):
    attempts, hints, speed, _, transformed_features = compute_behavior_columns(
        data_frame,
        clip_values=behavior_bundle.clip_values,
    )
    scaled_features = behavior_bundle.scaler.transform(transformed_features)
    raw_labels = behavior_bundle.cluster_model.predict(scaled_features)
    soft_membership = _soft_membership(scaled_features, behavior_bundle)
    cluster_ids = np.array(
        [behavior_bundle.cluster_mapping[int(raw_label)] for raw_label in raw_labels],
        dtype=np.int32,
    )
    question_speed_raw = (
        data_frame["problem_id"]
        .map(behavior_bundle.question_speed_reference)
        .fillna(behavior_bundle.global_speed_reference)
        .to_numpy(dtype=np.float32)
    )
    question_speed_log = np.log1p(np.clip(question_speed_raw, 0.0, behavior_bundle.clip_values["speed"])).astype(np.float32)
    student_speed_raw = _student_speed_baseline(
        data_frame=data_frame,
        fallback_speed=behavior_bundle.global_speed_reference,
    )
    student_speed_log = np.log1p(np.clip(student_speed_raw, 0.0, behavior_bundle.clip_values["speed"])).astype(np.float32)
    enriched = data_frame.copy()
    enriched["attempts"] = attempts
    enriched["hints"] = hints
    enriched["speed"] = speed
    enriched["speed_relative_student"] = (speed - student_speed_log).astype(np.float32)
    enriched["speed_relative_question"] = (speed - question_speed_log).astype(np.float32)
    enriched["behavior_cluster"] = cluster_ids
    for cluster_id in range(1, soft_membership.shape[1]):
        enriched[f"behavior_soft_{cluster_id}"] = soft_membership[:, cluster_id]
    return enriched


def build_sequence_records(data_frame, user_ids, sequence_length, num_behavior_clusters=None):
    user_ids = [int(user_id) for user_id in user_ids]
    split_frame = data_frame[data_frame["user_id"].isin(user_ids)].copy()
    grouped_indices = split_frame.groupby("user_id").groups
    if num_behavior_clusters is None:
        num_behavior_clusters = max(int(split_frame["behavior_cluster"].max()) + 1, 1)
    records = []
    for user_id in user_ids:
        if user_id not in grouped_indices:
            continue
        student_frame = split_frame.loc[grouped_indices[user_id]].sort_values(by=["timestamp"])
        question_ids = student_frame["question_ids"].to_numpy(dtype=np.int32)
        question_difficulty = student_frame["question_difficulty"].to_numpy(dtype=np.int32)
        responses = student_frame["correct"].to_numpy(dtype=np.int32)
        concept_ids_internal = student_frame["concept_ids_internal"].to_numpy(dtype=np.int32)
        concept_ids = student_frame["concept_ids"].to_numpy(dtype=np.int32)
        concept_difficulty = student_frame["concept_difficulty"].to_numpy(dtype=np.int32)
        attempts = student_frame["attempts"].to_numpy(dtype=np.float32)
        hints = student_frame["hints"].to_numpy(dtype=np.float32)
        speed = student_frame["speed"].to_numpy(dtype=np.float32)
        speed_relative_student = student_frame.get("speed_relative_student", pd.Series(np.zeros(len(student_frame), dtype=np.float32))).to_numpy(dtype=np.float32)
        speed_relative_question = student_frame.get("speed_relative_question", pd.Series(np.zeros(len(student_frame), dtype=np.float32))).to_numpy(dtype=np.float32)
        behavior_cluster = student_frame["behavior_cluster"].to_numpy(dtype=np.int32)
        if num_behavior_clusters > 1 and all(
            f"behavior_soft_{cluster_id}" in student_frame.columns for cluster_id in range(1, num_behavior_clusters)
        ):
            soft_membership = np.zeros((len(student_frame), num_behavior_clusters), dtype=np.float32)
            for cluster_id in range(1, num_behavior_clusters):
                soft_membership[:, cluster_id] = student_frame[f"behavior_soft_{cluster_id}"].to_numpy(dtype=np.float32)
        else:
            soft_membership = np.zeros((len(student_frame), num_behavior_clusters), dtype=np.float32)

        for start_index in range(0, len(question_ids), sequence_length):
            end_index = min(start_index + sequence_length, len(question_ids))
            if end_index - start_index < 2:
                continue
            records.append(
                {
                    "question_ids": question_ids[start_index:end_index],
                    "question_difficulty": question_difficulty[start_index:end_index],
                    "responses": responses[start_index:end_index],
                    "concept_ids_internal": concept_ids_internal[start_index:end_index],
                    "concept_ids": concept_ids[start_index:end_index],
                    "concept_difficulty": concept_difficulty[start_index:end_index],
                    "attempts": attempts[start_index:end_index],
                    "hints": hints[start_index:end_index],
                    "speed": speed[start_index:end_index],
                    "speed_relative_student": speed_relative_student[start_index:end_index],
                    "speed_relative_question": speed_relative_question[start_index:end_index],
                    "behavior_cluster": behavior_cluster[start_index:end_index],
                    "behavior_soft_membership": soft_membership[start_index:end_index],
                }
            )
    return records


def shuffle_records(records, seed=2):
    generator = np.random.RandomState(seed)
    indices = generator.permutation(len(records))
    return [records[index] for index in indices]


def records_to_bundle(records, sequence_length):
    num_records = len(records)
    if num_records == 0:
        raise ValueError("没有可用的序列记录，无法构建 SequenceBundle。")

    question_ids = np.zeros((num_records, sequence_length), dtype=np.int32)
    concept_ids = np.zeros((num_records, sequence_length), dtype=np.int32)
    responses = np.zeros((num_records, sequence_length), dtype=np.int32)
    question_difficulty = np.zeros((num_records, sequence_length), dtype=np.int32)
    concept_difficulty = np.zeros((num_records, sequence_length), dtype=np.int32)
    attempts = np.zeros((num_records, sequence_length), dtype=np.float32)
    hints = np.zeros((num_records, sequence_length), dtype=np.float32)
    speed = np.zeros((num_records, sequence_length), dtype=np.float32)
    speed_relative_student = np.zeros((num_records, sequence_length), dtype=np.float32)
    speed_relative_question = np.zeros((num_records, sequence_length), dtype=np.float32)
    behavior_cluster = np.zeros((num_records, sequence_length), dtype=np.int32)
    num_behavior_clusters = int(records[0]["behavior_soft_membership"].shape[-1])
    behavior_soft_membership = np.zeros((num_records, sequence_length, num_behavior_clusters), dtype=np.float32)
    mask = np.zeros((num_records, sequence_length), dtype=np.int32)

    for row_index, record in enumerate(records):
        record_length = len(record["question_ids"])
        question_ids[row_index, :record_length] = record["question_ids"]
        concept_ids[row_index, :record_length] = record["concept_ids"]
        responses[row_index, :record_length] = record["responses"]
        question_difficulty[row_index, :record_length] = record["question_difficulty"]
        concept_difficulty[row_index, :record_length] = record["concept_difficulty"]
        attempts[row_index, :record_length] = record["attempts"]
        hints[row_index, :record_length] = record["hints"]
        speed[row_index, :record_length] = record["speed"]
        speed_relative_student[row_index, :record_length] = record["speed_relative_student"]
        speed_relative_question[row_index, :record_length] = record["speed_relative_question"]
        behavior_cluster[row_index, :record_length] = record["behavior_cluster"]
        behavior_soft_membership[row_index, :record_length, :] = record["behavior_soft_membership"]
        mask[row_index, :record_length] = 1

    return SequenceBundle(
        question_ids=question_ids,
        concept_ids=concept_ids,
        responses=responses,
        question_difficulty=question_difficulty,
        concept_difficulty=concept_difficulty,
        attempts=attempts,
        hints=hints,
        speed=speed,
        speed_relative_student=speed_relative_student,
        speed_relative_question=speed_relative_question,
        behavior_cluster=behavior_cluster,
        behavior_soft_membership=behavior_soft_membership,
        mask=mask,
    )


def records_to_dimkt_array(records):
    return np.array(
        [
            [
                record["question_ids"].tolist(),
                record["question_difficulty"].tolist(),
                record["responses"].tolist(),
                record["concept_ids_internal"].tolist(),
                len(record["question_ids"]),
                record["concept_difficulty"].tolist(),
            ]
            for record in records
        ],
        dtype=object,
    )


def validate_against_dimkt(records, npy_path):
    reference_array = np.load(npy_path, allow_pickle=True)
    if len(reference_array) != len(records):
        raise ValueError(f"{npy_path} 序列数不一致：{len(reference_array)} vs {len(records)}")
    reference_lengths = sorted(int(row[4]) for row in reference_array)
    current_lengths = sorted(len(record["question_ids"]) for record in records)
    if reference_lengths != current_lengths:
        raise ValueError(f"{npy_path} 的序列长度分布与当前桥接结果不一致。")


def build_assist2012_bundles(
    csv_path,
    dimkt_data_dir,
    sequence_length=100,
    n_clusters=4,
    random_seed=2026,
    cluster_sample_size=200000,
    validate_dimkt=True,
):
    reference_assets = load_dimkt_reference_assets(dimkt_data_dir)
    raw_data_frame = load_assist2012_interactions(csv_path)
    train_ids, valid_ids, test_ids = split_user_ids(raw_data_frame["user_id"].unique())
    data_frame = attach_dimkt_fields(raw_data_frame, reference_assets)

    train_interactions = data_frame[data_frame["user_id"].isin(train_ids)].copy()
    behavior_bundle = fit_behavior_clusters(
        train_frame=train_interactions,
        n_clusters=n_clusters,
        random_seed=random_seed,
        sample_size=cluster_sample_size,
    )
    data_frame = attach_behavior_features(data_frame, behavior_bundle)

    train_records = build_sequence_records(data_frame, train_ids, sequence_length)
    valid_records = build_sequence_records(
        data_frame,
        valid_ids,
        sequence_length,
        num_behavior_clusters=int(n_clusters + 1),
    )
    test_records = build_sequence_records(
        data_frame,
        test_ids,
        sequence_length,
        num_behavior_clusters=int(n_clusters + 1),
    )

    train_records = shuffle_records(train_records, seed=2)
    valid_records = shuffle_records(valid_records, seed=2)

    if validate_dimkt:
        dimkt_data_dir = Path(dimkt_data_dir)
        validate_against_dimkt(train_records, dimkt_data_dir / "train0.npy")
        validate_against_dimkt(valid_records, dimkt_data_dir / "valid0.npy")
        validate_against_dimkt(test_records, dimkt_data_dir / "test.npy")

    train_bundle = records_to_bundle(train_records, sequence_length=sequence_length)
    valid_bundle = records_to_bundle(valid_records, sequence_length=sequence_length)
    test_bundle = records_to_bundle(test_records, sequence_length=sequence_length)

    full_cluster_counts = data_frame["behavior_cluster"].value_counts().sort_index()
    metadata = {
        "dataset_name": "assist2012",
        "sequence_length": int(sequence_length),
        "num_questions": int(data_frame["question_ids"].max()),
        "num_concepts": int(data_frame["concept_ids"].max()),
        "num_question_difficulty": int(data_frame["question_difficulty"].max()),
        "num_concept_difficulty": int(data_frame["concept_difficulty"].max()),
        "num_behavior_clusters": int(n_clusters + 1),
        "num_filtered_interactions": int(len(data_frame)),
        "num_users": int(data_frame["user_id"].nunique()),
        "clip_values": {key: float(value) for key, value in behavior_bundle.clip_values.items()},
        "behavior_centers": behavior_bundle.centers,
        "cluster_counts": {str(int(cluster_id)): int(count) for cluster_id, count in full_cluster_counts.items()},
        "split_summary": {
            "train_users": int(len(train_ids)),
            "valid_users": int(len(valid_ids)),
            "test_users": int(len(test_ids)),
            "train_sequences": int(train_bundle.num_samples),
            "valid_sequences": int(valid_bundle.num_samples),
            "test_sequences": int(test_bundle.num_samples),
        },
    }
    return train_bundle, valid_bundle, test_bundle, metadata


def save_bundle_npz(bundle, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **bundle.as_dict())
