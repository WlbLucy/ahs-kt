DIMKT_FIELDS = {
    "question_ids": "problem_id 或 question_id",
    "concept_ids": "skill_id / kc_id",
    "responses": "correctness",
    "question_difficulty": "problem difficulty bin",
    "concept_difficulty": "skill difficulty bin",
}


LBKT_FIELDS = {
    "attempts": "尝试次数或 attempts_factor 对应的原始次数",
    "hints": "提示使用次数或 hints_factor 对应的原始次数",
    "speed": "作答速度或 time-based speed 特征",
    "speed_relative_student": "相对学生个人历史速度的偏移量",
    "speed_relative_question": "相对题目典型速度的偏移量",
    "behavior_cluster": "基于 attempts / hints / speed 的聚类编号",
    "behavior_soft_membership": "对各行为原型的软权重",
    "mask": "有效位置掩码",
}


AHSKT_TARGET_FIELDS = [
    "question_ids",
    "concept_ids",
    "responses",
    "question_difficulty",
    "concept_difficulty",
    "attempts",
    "hints",
    "speed",
    "speed_relative_student",
    "speed_relative_question",
    "behavior_cluster",
    "behavior_soft_membership",
    "mask",
]


def describe_alignment():
    return {
        "from_dimkt": DIMKT_FIELDS,
        "from_lbkt": LBKT_FIELDS,
        "target_fields": AHSKT_TARGET_FIELDS,
    }
