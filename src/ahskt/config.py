from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class DatasetConfig:
    mode: str
    train_path: str
    valid_path: str
    test_path: str


@dataclass
class ModelConfig:
    num_questions: int
    num_concepts: int
    num_question_difficulty: int
    num_concept_difficulty: int
    num_behavior_clusters: int
    sequence_length: int
    embedding_dim: int
    difficulty_dim: int
    behavior_dim: int
    hidden_dim: int
    dropout: float
    use_behavior_cluster: bool
    use_difficulty_features: bool = True
    use_behavior_features: bool = True
    use_target_interaction: bool = False
    fusion_mode: str = "early"
    behavior_condition_on_difficulty: bool = True
    aux_residual_scale: float = 0.25
    difficulty_mode: str = "embedding"
    difficulty_bias_scale: float = 0.1
    difficulty_feature_source: str = "question_concept"
    question_global_easiness: float = 0.5
    concept_global_easiness: float = 0.5
    use_relative_speed: bool = True
    use_soft_behavior_prototypes: bool = True


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int


@dataclass
class DemoConfig:
    train_size: int
    valid_size: int
    test_size: int


@dataclass
class OutputConfig:
    root_dir: str


@dataclass
class AHSKTConfig:
    project_name: str
    task_name: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    demo: DemoConfig
    outputs: OutputConfig
    project_root: Path
    output_root: Path


def load_config(config_path, project_root):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    payload = json.loads(config_path.read_text(encoding="utf-8"))

    dataset = DatasetConfig(**payload["dataset"])
    model = ModelConfig(**payload["model"])
    training = TrainingConfig(**payload["training"])
    demo = DemoConfig(**payload["demo"])
    outputs = OutputConfig(**payload["outputs"])

    output_root = Path(outputs.root_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    return AHSKTConfig(
        project_name=payload["project_name"],
        task_name=payload["task_name"],
        seed=int(payload["seed"]),
        dataset=dataset,
        model=model,
        training=training,
        demo=demo,
        outputs=outputs,
        project_root=Path(project_root),
        output_root=output_root,
    )
