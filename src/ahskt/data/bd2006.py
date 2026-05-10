from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from .dataset import SequenceBundle


BD2006_USECOLS = [
    "Row",
    "Anon Student Id",
    "Problem Name",
    "Step Name",
    "Step Start Time",
    "Correct First Attempt",
    "Incorrects",
    "Hints",
    "Corrects",
    "Step Duration (sec)",
    "KC(SubSkills)",
]
NO_KC_TOKEN = "__NO_KC__"


@dataclass
class BD2006PreparedData:
    train_bundle: SequenceBundle
    valid_bundle: SequenceBundle
    test_bundle: SequenceBundle
    metadata: Dict


def normalize_kc_combo(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return NO_KC_TOKEN
    parts = [part.strip() for part in str(value).split("~~") if part and part.strip()]
    if not parts:
        return NO_KC_TOKEN
    return "~~".join(sorted(set(parts)))


def assign_tail_validation_targets(student_lengths: Dict[str, int], valid_points_per_user: int) -> Dict[str, int]:
    assignments = {}
    for uid, length in student_lengths.items():
        if length <= 1:
            assignments[uid] = 0
            continue
        assignments[uid] = min(int(valid_points_per_user), int(length) - 1)
    return assignments


def prepare_bd2006_bundles(
    data_dir,
    sequence_length: int = 200,
    valid_points_per_user: int = 5,
    n_behavior_clusters: int = 4,
    question_alpha: float = 5.0,
    concept_alpha: float = 20.0,
) -> BD2006PreparedData:
    data_dir = Path(data_dir)
    train_path = data_dir / "bridge_to_algebra_2006_2007_train.txt"
    master_path = data_dir / "bridge_to_algebra_2006_2007_master.txt"
    test_path = data_dir / "bridge_to_algebra_2006_2007_test.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"missing file: {train_path}")
    if not master_path.exists():
        raise FileNotFoundError(f"missing file: {master_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"missing file: {test_path}")

    train_frame = _load_question_level_frame(train_path)
    master_frame = _load_question_level_frame(master_path)
    _validate_public_test_alignment(test_path, master_frame)

    question_id_map, concept_id_map = _build_id_maps(train_frame, master_frame)
    _attach_id_columns(train_frame, question_id_map, concept_id_map)
    _attach_id_columns(master_frame, question_id_map, concept_id_map)

    train_frame = _sort_student_timeline(train_frame)
    master_frame = _sort_student_timeline(master_frame)

    student_lengths = {uid: int(size) for uid, size in train_frame.groupby("uid", sort=False).size().items()}
    valid_tail_counts = assign_tail_validation_targets(student_lengths, valid_points_per_user)
    train_visible_frame, valid_target_frame = _split_train_and_valid(train_frame, valid_tail_counts)

    (
        q_diff_map,
        q_ease_map,
        q_conf_map,
        default_q_diff,
        default_q_ease,
        c_diff_map,
        c_ease_map,
        c_conf_map,
        default_c_diff,
        default_c_ease,
        global_mean,
    ) = _fit_difficulty_maps(train_visible_frame, question_alpha=question_alpha, concept_alpha=concept_alpha)

    (
        behavior_clip_values,
        behavior_scaler,
        behavior_cluster_model,
        behavior_cluster_mapping,
        behavior_centers,
    ) = _fit_behavior_clusters(train_visible_frame, n_behavior_clusters=n_behavior_clusters)

    train_visible_enriched = _attach_feature_columns(
        train_visible_frame,
        q_diff_map=q_diff_map,
        c_diff_map=c_diff_map,
        default_q_diff=default_q_diff,
        default_c_diff=default_c_diff,
        q_ease_map=q_ease_map,
        c_ease_map=c_ease_map,
        q_conf_map=q_conf_map,
        c_conf_map=c_conf_map,
        default_q_ease=default_q_ease,
        default_c_ease=default_c_ease,
        behavior_clip_values=behavior_clip_values,
        behavior_scaler=behavior_scaler,
        behavior_cluster_model=behavior_cluster_model,
        behavior_cluster_mapping=behavior_cluster_mapping,
    )
    full_train_enriched = _attach_feature_columns(
        train_frame,
        q_diff_map=q_diff_map,
        c_diff_map=c_diff_map,
        default_q_diff=default_q_diff,
        default_c_diff=default_c_diff,
        q_ease_map=q_ease_map,
        c_ease_map=c_ease_map,
        q_conf_map=q_conf_map,
        c_conf_map=c_conf_map,
        default_q_ease=default_q_ease,
        default_c_ease=default_c_ease,
        behavior_clip_values=behavior_clip_values,
        behavior_scaler=behavior_scaler,
        behavior_cluster_model=behavior_cluster_model,
        behavior_cluster_mapping=behavior_cluster_mapping,
    )
    valid_target_enriched = _attach_feature_columns(
        valid_target_frame,
        q_diff_map=q_diff_map,
        c_diff_map=c_diff_map,
        default_q_diff=default_q_diff,
        default_c_diff=default_c_diff,
        q_ease_map=q_ease_map,
        c_ease_map=c_ease_map,
        q_conf_map=q_conf_map,
        c_conf_map=c_conf_map,
        default_q_ease=default_q_ease,
        default_c_ease=default_c_ease,
        behavior_clip_values=behavior_clip_values,
        behavior_scaler=behavior_scaler,
        behavior_cluster_model=behavior_cluster_model,
        behavior_cluster_mapping=behavior_cluster_mapping,
    )
    master_enriched = _attach_feature_columns(
        master_frame,
        q_diff_map=q_diff_map,
        c_diff_map=c_diff_map,
        default_q_diff=default_q_diff,
        default_c_diff=default_c_diff,
        q_ease_map=q_ease_map,
        c_ease_map=c_ease_map,
        q_conf_map=q_conf_map,
        c_conf_map=c_conf_map,
        default_q_ease=default_q_ease,
        default_c_ease=default_c_ease,
        behavior_clip_values=behavior_clip_values,
        behavior_scaler=behavior_scaler,
        behavior_cluster_model=behavior_cluster_model,
        behavior_cluster_mapping=behavior_cluster_mapping,
    )

    train_bundle = _build_training_bundle(train_visible_enriched, sequence_length=sequence_length)
    valid_bundle, valid_stats = _build_validation_bundle(
        history_frame=train_visible_enriched,
        target_frame=valid_target_enriched,
        sequence_length=sequence_length,
    )
    test_bundle, test_stats = _build_test_bundle(
        history_frame=full_train_enriched,
        target_frame=master_enriched,
        sequence_length=sequence_length,
    )

    metadata = {
        "dataset_name": "BD2006",
        "representation": "question_level_combo_concept_no_leak",
        "sequence_length": int(sequence_length),
        "valid_points_per_user": int(valid_points_per_user),
        "n_behavior_clusters": int(n_behavior_clusters),
        "num_questions": int(max(question_id_map.values()) if question_id_map else 0),
        "num_concepts": int(max(concept_id_map.values()) if concept_id_map else 0),
        "num_behavior_clusters_with_padding": int(n_behavior_clusters + 1),
        "question_global_easiness": float(global_mean),
        "concept_global_easiness": float(global_mean),
        "default_question_difficulty": int(default_q_diff),
        "default_concept_difficulty": int(default_c_diff),
        "behavior_centers": behavior_centers,
        "behavior_clip_values": {key: float(value) for key, value in behavior_clip_values.items()},
        "split_summary": {
            "train_visible_interactions": int(len(train_visible_enriched)),
            "valid_targets_total": int(len(valid_target_enriched)),
            "test_targets_total": int(len(master_enriched)),
            "train_sequences": int(train_bundle.num_samples),
            "valid_sequences": int(valid_bundle.num_samples),
            "test_sequences": int(test_bundle.num_samples),
            "valid_target_points_used": int(valid_stats["used_targets"]),
            "valid_target_points_dropped_no_history": int(valid_stats["dropped_no_history"]),
            "test_target_points_used": int(test_stats["used_targets"]),
            "test_target_points_dropped_no_history": int(test_stats["dropped_no_history"]),
            "students": int(train_frame["uid"].nunique()),
        },
        "notes": [
            "KC(SubSkills) 为空时保留为 __NO_KC__，避免丢掉大量样本。",
            "validation 只取每个学生 train 时间轴最后若干个点，并且不把这些 held-out 点当作可观察历史。",
            "test 使用官方 master 标签，但每个测试点只允许看到更早的官方 train 交互。",
        ],
    }

    return BD2006PreparedData(
        train_bundle=train_bundle,
        valid_bundle=valid_bundle,
        test_bundle=test_bundle,
        metadata=metadata,
    )


def _load_question_level_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        sep="\t",
        usecols=BD2006_USECOLS,
        dtype="string",
        low_memory=False,
        encoding="latin1",
    )
    raw = raw[raw["Correct First Attempt"].isin(["0", "1"])].copy()

    duration = pd.to_numeric(raw["Step Duration (sec)"], errors="coerce")
    median_duration = float(duration.dropna().median()) if duration.notna().any() else 15.0
    if not np.isfinite(median_duration) or median_duration <= 0:
        median_duration = 15.0

    question_key = raw["Problem Name"].fillna("") + "----" + raw["Step Name"].fillna("")
    concept_key = raw["KC(SubSkills)"].map(normalize_kc_combo)
    step_start_ts = pd.to_datetime(raw["Step Start Time"], errors="coerce")
    sort_ts = step_start_ts.fillna(pd.Timestamp.max)
    sort_ts_ns = sort_ts.astype("int64")

    frame = pd.DataFrame(
        {
            "uid": raw["Anon Student Id"].fillna("").astype(str),
            "row_id": pd.to_numeric(raw["Row"], errors="coerce").fillna(0).astype(np.int64),
            "question_key": question_key.astype(str),
            "concept_key": concept_key.astype(str),
            "response": raw["Correct First Attempt"].astype(np.int32),
            "attempt_count_raw": (
                pd.to_numeric(raw["Incorrects"], errors="coerce").fillna(0).clip(lower=0)
                + pd.to_numeric(raw["Corrects"], errors="coerce").fillna(1).clip(lower=0)
            ).clip(lower=1.0).astype(np.float32),
            "hint_count_raw": pd.to_numeric(raw["Hints"], errors="coerce").fillna(0).clip(lower=0).astype(np.float32),
            "duration_sec": duration.fillna(median_duration).clip(lower=1.0).astype(np.float32),
            "step_start_ts": step_start_ts,
            "sort_ts_ns": sort_ts_ns.astype(np.int64),
        }
    )
    frame["speed_raw"] = (60.0 / frame["duration_sec"]).astype(np.float32)
    return frame


def _validate_public_test_alignment(test_path: Path, master_frame: pd.DataFrame) -> None:
    public_test = pd.read_csv(
        test_path,
        sep="\t",
        usecols=["Row", "Correct First Attempt"],
        dtype="string",
        low_memory=False,
        encoding="latin1",
    )
    public_test_rows = set(pd.to_numeric(public_test["Row"], errors="coerce").fillna(0).astype(np.int64).tolist())
    master_rows = set(master_frame["row_id"].astype(np.int64).tolist())
    if public_test_rows != master_rows:
        raise ValueError("public test rows and master rows do not align")
    if public_test["Correct First Attempt"].notna().any():
        raise ValueError("public test file is expected to be unlabeled")


def _build_id_maps(train_frame: pd.DataFrame, master_frame: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    question_values = pd.Index(
        pd.concat([train_frame["question_key"], master_frame["question_key"]], ignore_index=True).unique()
    ).sort_values()
    concept_values = pd.Index(
        pd.concat([train_frame["concept_key"], master_frame["concept_key"]], ignore_index=True).unique()
    ).sort_values()
    question_id_map = {value: idx + 1 for idx, value in enumerate(question_values.tolist())}
    concept_id_map = {value: idx + 1 for idx, value in enumerate(concept_values.tolist())}
    return question_id_map, concept_id_map


def _attach_id_columns(frame: pd.DataFrame, question_id_map: Dict[str, int], concept_id_map: Dict[str, int]) -> None:
    frame["question_id"] = frame["question_key"].map(question_id_map).astype(np.int32)
    frame["concept_id"] = frame["concept_key"].map(concept_id_map).astype(np.int32)
    frame.drop(columns=["question_key", "concept_key"], inplace=True)


def _sort_student_timeline(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["uid", "sort_ts_ns", "row_id"], kind="mergesort").reset_index(drop=True)


def _split_train_and_valid(
    train_frame: pd.DataFrame,
    valid_tail_counts: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_visible_mask = np.ones(len(train_frame), dtype=np.int8)
    valid_target_mask = np.zeros(len(train_frame), dtype=np.int8)

    for uid, positions in train_frame.groupby("uid", sort=False).indices.items():
        position_array = np.asarray(positions, dtype=np.int64)
        holdout_count = int(valid_tail_counts.get(uid, 0))
        if holdout_count <= 0:
            continue
        tail_positions = position_array[-holdout_count:]
        train_visible_mask[tail_positions] = 0
        valid_target_mask[tail_positions] = 1

    train_visible = train_frame.loc[train_visible_mask == 1].copy().reset_index(drop=True)
    valid_targets = train_frame.loc[valid_target_mask == 1].copy().reset_index(drop=True)
    return train_visible, valid_targets


def _fit_difficulty_maps(
    train_visible_frame: pd.DataFrame,
    question_alpha: float,
    concept_alpha: float,
):
    q_diff_map, q_ease_map, q_conf_map, default_q_diff, default_q_ease = _compute_smoothed_maps(
        train_visible_frame,
        id_col="question_id",
        response_col="response",
        alpha=question_alpha,
    )
    c_diff_map, c_ease_map, c_conf_map, default_c_diff, default_c_ease = _compute_smoothed_maps(
        train_visible_frame,
        id_col="concept_id",
        response_col="response",
        alpha=concept_alpha,
    )
    global_mean = float(train_visible_frame["response"].mean()) if len(train_visible_frame) else 0.5
    return (
        q_diff_map,
        q_ease_map,
        q_conf_map,
        default_q_diff,
        default_q_ease,
        c_diff_map,
        c_ease_map,
        c_conf_map,
        default_c_diff,
        default_c_ease,
        global_mean,
    )


def _compute_smoothed_maps(frame: pd.DataFrame, id_col: str, response_col: str, alpha: float):
    grouped = frame.groupby(id_col)[response_col].agg(["sum", "count"])
    global_mean = float(frame[response_col].mean()) if len(frame) else 0.5
    default_bin = int(np.clip(int(global_mean * 100) + 1, 1, 101))
    posterior = (grouped["sum"] + alpha * global_mean) / (grouped["count"] + alpha)
    bins = np.clip((posterior * 100).astype(int) + 1, 1, 101).astype(int)
    confidence = grouped["count"] / (grouped["count"] + alpha) if alpha > 0 else pd.Series(1.0, index=grouped.index)
    return (
        bins.to_dict(),
        posterior.astype(float).to_dict(),
        confidence.astype(float).to_dict(),
        default_bin,
        float(global_mean),
    )


def _fit_behavior_clusters(train_visible_frame: pd.DataFrame, n_behavior_clusters: int):
    _, _, _, clip_values, features = _compute_behavior_columns(train_visible_frame)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    cluster_model = MiniBatchKMeans(
        n_clusters=n_behavior_clusters,
        random_state=2026,
        n_init=20,
        batch_size=4096,
    )
    cluster_model.fit(scaled_features)
    raw_centers = scaler.inverse_transform(cluster_model.cluster_centers_)
    cluster_mapping = _build_cluster_mapping(raw_centers)
    centers = []
    for raw_label, center in enumerate(raw_centers):
        cluster_id = cluster_mapping[int(raw_label)]
        centers.append(
            {
                "cluster_id": int(cluster_id),
                "attempts_log_center": float(center[0]),
                "hints_log_center": float(center[1]),
                "speed_log_center": float(center[2]),
                "attempts_raw_center": float(np.expm1(center[0])),
                "hints_raw_center": float(np.expm1(center[1])),
                "speed_raw_center": float(np.expm1(center[2])),
            }
        )
    centers = sorted(centers, key=lambda item: item["cluster_id"])
    return clip_values, scaler, cluster_model, cluster_mapping, centers


def _compute_behavior_columns(frame: pd.DataFrame, clip_values=None):
    attempts_raw = frame["attempt_count_raw"].to_numpy(dtype=np.float32)
    hints_raw = frame["hint_count_raw"].to_numpy(dtype=np.float32)
    speed_raw = frame["speed_raw"].to_numpy(dtype=np.float32)
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


def _build_cluster_mapping(raw_centers: np.ndarray):
    order = sorted(
        range(len(raw_centers)),
        key=lambda idx: (
            float(raw_centers[idx, 0]),
            float(raw_centers[idx, 1]),
            float(-raw_centers[idx, 2]),
        ),
    )
    return {int(old_label): int(new_label + 1) for new_label, old_label in enumerate(order)}


def _attach_feature_columns(
    frame: pd.DataFrame,
    *,
    q_diff_map: Dict[int, int],
    c_diff_map: Dict[int, int],
    default_q_diff: int,
    default_c_diff: int,
    q_ease_map: Dict[int, float],
    c_ease_map: Dict[int, float],
    q_conf_map: Dict[int, float],
    c_conf_map: Dict[int, float],
    default_q_ease: float,
    default_c_ease: float,
    behavior_clip_values: Dict[str, float],
    behavior_scaler: StandardScaler,
    behavior_cluster_model: MiniBatchKMeans,
    behavior_cluster_mapping: Dict[int, int],
) -> pd.DataFrame:
    attempts, hints, speed, _, features = _compute_behavior_columns(frame, clip_values=behavior_clip_values)
    scaled_features = behavior_scaler.transform(features)
    raw_labels = behavior_cluster_model.predict(scaled_features)
    cluster_ids = np.array([behavior_cluster_mapping[int(raw_label)] for raw_label in raw_labels], dtype=np.int32)

    enriched = frame.copy()
    enriched["question_difficulty"] = (
        enriched["question_id"].map(q_diff_map).fillna(default_q_diff).astype(np.int32)
    )
    enriched["concept_difficulty"] = (
        enriched["concept_id"].map(c_diff_map).fillna(default_c_diff).astype(np.int32)
    )
    enriched["question_easiness"] = (
        enriched["question_id"].map(q_ease_map).fillna(default_q_ease).astype(np.float32)
    )
    enriched["concept_easiness"] = (
        enriched["concept_id"].map(c_ease_map).fillna(default_c_ease).astype(np.float32)
    )
    enriched["question_confidence"] = enriched["question_id"].map(q_conf_map).fillna(0.0).astype(np.float32)
    enriched["concept_confidence"] = enriched["concept_id"].map(c_conf_map).fillna(0.0).astype(np.float32)
    enriched["attempts"] = attempts
    enriched["hints"] = hints
    enriched["speed"] = speed
    enriched["behavior_cluster"] = cluster_ids
    return enriched


def _build_training_bundle(frame: pd.DataFrame, sequence_length: int) -> SequenceBundle:
    records = []
    for _, student in frame.groupby("uid", sort=False):
        records.extend(_chunk_student_rows(student, sequence_length=sequence_length))
    return _records_to_bundle(records, sequence_length=sequence_length)


def _build_validation_bundle(
    history_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    sequence_length: int,
):
    history_by_user = {uid: student.reset_index(drop=True) for uid, student in history_frame.groupby("uid", sort=False)}
    records = []
    dropped_no_history = 0
    used_targets = 0
    for uid, targets in target_frame.groupby("uid", sort=False):
        history_student = history_by_user.get(uid)
        if history_student is None or history_student.empty:
            dropped_no_history += int(len(targets))
            continue
        history_tail = history_student.tail(sequence_length - 1)
        for _, target_row in targets.iterrows():
            record = _single_target_record(history_tail, target_row, sequence_length=sequence_length)
            if record is None:
                dropped_no_history += 1
                continue
            records.append(record)
            used_targets += 1
    return _records_to_bundle(records, sequence_length=sequence_length), {
        "used_targets": used_targets,
        "dropped_no_history": dropped_no_history,
    }


def _build_test_bundle(
    history_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    sequence_length: int,
):
    history_by_user = {uid: student.reset_index(drop=True) for uid, student in history_frame.groupby("uid", sort=False)}
    records = []
    dropped_no_history = 0
    used_targets = 0

    for uid, targets in target_frame.groupby("uid", sort=False):
        history_student = history_by_user.get(uid)
        if history_student is None or history_student.empty:
            dropped_no_history += int(len(targets))
            continue

        history_keys = np.rec.fromarrays(
            [
                history_student["sort_ts_ns"].to_numpy(dtype=np.int64),
                history_student["row_id"].to_numpy(dtype=np.int64),
            ],
            names=["ts_ns", "row_id"],
        )
        key_dtype = history_keys.dtype

        for _, target_row in targets.iterrows():
            target_key = np.array((int(target_row["sort_ts_ns"]), int(target_row["row_id"])), dtype=key_dtype)
            history_end = int(np.searchsorted(history_keys, target_key, side="left"))
            if history_end <= 0:
                dropped_no_history += 1
                continue
            history_tail = history_student.iloc[max(0, history_end - (sequence_length - 1)) : history_end]
            record = _single_target_record(history_tail, target_row, sequence_length=sequence_length)
            if record is None:
                dropped_no_history += 1
                continue
            records.append(record)
            used_targets += 1

    return _records_to_bundle(records, sequence_length=sequence_length), {
        "used_targets": used_targets,
        "dropped_no_history": dropped_no_history,
    }


def _chunk_student_rows(student: pd.DataFrame, sequence_length: int) -> List[Dict[str, np.ndarray]]:
    student = student.reset_index(drop=True)
    records = []
    total_len = len(student)
    for start in range(0, total_len, sequence_length):
        end = min(start + sequence_length, total_len)
        current = student.iloc[start:end]
        if len(current) < 2:
            continue
        mask = np.zeros(sequence_length, dtype=np.int32)
        mask[: len(current)] = 1
        records.append(_frame_to_record(current, mask, sequence_length=sequence_length))
    return records


def _single_target_record(history_tail: pd.DataFrame, target_row: pd.Series, sequence_length: int):
    if history_tail is None or history_tail.empty:
        return None
    target_frame = pd.DataFrame([target_row])
    current = pd.concat([history_tail.tail(sequence_length - 1), target_frame], ignore_index=True)
    if len(current) < 2:
        return None
    mask = np.zeros(sequence_length, dtype=np.int32)
    mask[len(current) - 1] = 1
    return _frame_to_record(current, mask, sequence_length=sequence_length)


def _frame_to_record(frame: pd.DataFrame, mask: np.ndarray, sequence_length: int) -> Dict[str, np.ndarray]:
    frame = frame.reset_index(drop=True)
    current_len = len(frame)
    record = {
        "question_ids": np.zeros(sequence_length, dtype=np.int32),
        "concept_ids": np.zeros(sequence_length, dtype=np.int32),
        "responses": np.zeros(sequence_length, dtype=np.int32),
        "question_difficulty": np.zeros(sequence_length, dtype=np.int32),
        "concept_difficulty": np.zeros(sequence_length, dtype=np.int32),
        "attempts": np.zeros(sequence_length, dtype=np.float32),
        "hints": np.zeros(sequence_length, dtype=np.float32),
        "speed": np.zeros(sequence_length, dtype=np.float32),
        "behavior_cluster": np.zeros(sequence_length, dtype=np.int32),
        "mask": mask.astype(np.int32),
        "question_easiness": np.zeros(sequence_length, dtype=np.float32),
        "concept_easiness": np.zeros(sequence_length, dtype=np.float32),
        "question_confidence": np.zeros(sequence_length, dtype=np.float32),
        "concept_confidence": np.zeros(sequence_length, dtype=np.float32),
    }

    copy_map = {
        "question_ids": "question_id",
        "concept_ids": "concept_id",
        "responses": "response",
        "question_difficulty": "question_difficulty",
        "concept_difficulty": "concept_difficulty",
        "attempts": "attempts",
        "hints": "hints",
        "speed": "speed",
        "behavior_cluster": "behavior_cluster",
        "question_easiness": "question_easiness",
        "concept_easiness": "concept_easiness",
        "question_confidence": "question_confidence",
        "concept_confidence": "concept_confidence",
    }
    for target_name, source_name in copy_map.items():
        record[target_name][:current_len] = frame[source_name].to_numpy(dtype=record[target_name].dtype)

    return record


def _records_to_bundle(records: List[Dict[str, np.ndarray]], sequence_length: int) -> SequenceBundle:
    if not records:
        raise ValueError("no sequence records were generated")
    payload = {}
    for key in records[0]:
        payload[key] = np.stack([record[key] for record in records], axis=0)
    num_behavior_cluster_channels = max(int(np.max(payload["behavior_cluster"])) + 1, 1)
    return SequenceBundle(
        question_ids=payload["question_ids"],
        concept_ids=payload["concept_ids"],
        responses=payload["responses"],
        question_difficulty=payload["question_difficulty"],
        concept_difficulty=payload["concept_difficulty"],
        attempts=payload["attempts"],
        hints=payload["hints"],
        speed=payload["speed"],
        behavior_cluster=payload["behavior_cluster"],
        mask=payload["mask"],
        question_easiness=payload["question_easiness"],
        concept_easiness=payload["concept_easiness"],
        question_confidence=payload["question_confidence"],
        concept_confidence=payload["concept_confidence"],
        speed_relative_student=np.zeros((len(records), sequence_length), dtype=np.float32),
        speed_relative_question=np.zeros((len(records), sequence_length), dtype=np.float32),
        behavior_soft_membership=np.zeros(
            (len(records), sequence_length, num_behavior_cluster_channels),
            dtype=np.float32,
        ),
    )
