import numpy as np
import pandas as pd

from ahskt.data.bd2006 import (
    NO_KC_TOKEN,
    _build_test_bundle,
    assign_tail_validation_targets,
    normalize_kc_combo,
)


def _row(uid, ts, row_id, question_id, response):
    return {
        "uid": uid,
        "sort_ts_ns": np.int64(ts),
        "row_id": np.int64(row_id),
        "question_id": np.int32(question_id),
        "concept_id": np.int32(1),
        "response": np.int32(response),
        "question_difficulty": np.int32(50),
        "concept_difficulty": np.int32(50),
        "attempts": np.float32(0.1),
        "hints": np.float32(0.2),
        "speed": np.float32(0.3),
        "behavior_cluster": np.int32(1),
        "question_easiness": np.float32(0.5),
        "concept_easiness": np.float32(0.5),
        "question_confidence": np.float32(0.8),
        "concept_confidence": np.float32(0.8),
    }


def test_normalize_kc_combo_handles_blank_and_dedup():
    assert normalize_kc_combo(None) == NO_KC_TOKEN
    assert normalize_kc_combo("") == NO_KC_TOKEN
    assert normalize_kc_combo("b~~a~~b") == "a~~b"


def test_assign_tail_validation_targets_keeps_prefix():
    assignments = assign_tail_validation_targets({"u1": 10, "u2": 2, "u3": 1}, valid_points_per_user=5)
    assert assignments == {"u1": 5, "u2": 1, "u3": 0}


def test_build_test_bundle_uses_only_earlier_train_history():
    history_frame = pd.DataFrame(
        [
            _row("u1", 10, 1, 101, 1),
            _row("u1", 20, 2, 102, 0),
        ]
    )
    target_frame = pd.DataFrame(
        [
            _row("u1", 5,  0, 201, 1),
            _row("u1", 15, 3, 202, 1),
            _row("u1", 25, 4, 203, 0),
        ]
    )

    bundle, stats = _build_test_bundle(history_frame, target_frame, sequence_length=4)

    assert stats == {"used_targets": 2, "dropped_no_history": 1}
    assert bundle.num_samples == 2
    assert bundle.mask[0].tolist() == [0, 1, 0, 0]
    assert bundle.mask[1].tolist() == [0, 0, 1, 0]
    assert bundle.question_ids[0, :2].tolist() == [101, 202]
    assert bundle.question_ids[1, :3].tolist() == [101, 102, 203]
