import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ahskt.config import load_config  # noqa: E402
from ahskt.data.bd2006 import BD2006PreparedData, prepare_bd2006_bundles  # noqa: E402
from ahskt.models.ahs_kt import AHSKTModel  # noqa: E402
from ahskt.training.engine import fit_and_evaluate  # noqa: E402


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
}


@dataclass
class BD2006RunConfig:
    data_dir: str = "/root/autodl-tmp/ahs-kt/data/BD06"
    task_name: str = "ahskt_bd2006_notebook"
    sequence_length: int = 200
    valid_points_per_user: int = 5
    n_behavior_clusters: int = 4
    question_alpha: float = 5.0
    concept_alpha: float = 20.0
    embedding_dim: int = 64
    difficulty_dim: int = 32
    behavior_dim: int = 32
    hidden_dim: int = 96
    dropout: float = 0.2
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 2
    seed: int = 2026
    output_root: str = "outputs/bd2006_notebook"
    data_output_root: str = "data/bd2006_notebook"
    config_output_path: str = "configs/ahskt_bd2006_notebook.json"
    cpu_only: bool = False


def run_bd2006_experiment(run_config: BD2006RunConfig):
    _apply_runtime_environment(run_config.cpu_only)

    import tensorflow as tf

    if not run_config.cpu_only:
        for gpu_device in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu_device, True)
            except RuntimeError:
                pass

    np.random.seed(run_config.seed)
    tf.random.set_seed(run_config.seed)

    prepared = prepare_bd2006_bundles(
        data_dir=run_config.data_dir,
        sequence_length=run_config.sequence_length,
        valid_points_per_user=run_config.valid_points_per_user,
        n_behavior_clusters=run_config.n_behavior_clusters,
        question_alpha=run_config.question_alpha,
        concept_alpha=run_config.concept_alpha,
    )

    data_output_root = PROJECT_ROOT / run_config.data_output_root
    output_root = PROJECT_ROOT / run_config.output_root
    data_output_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    saved_paths = _save_prepared_bundles(prepared, data_output_root, run_config.task_name)
    config_payload = _build_config_payload(run_config, prepared, saved_paths)
    config_path = PROJECT_ROOT / run_config.config_output_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    config = load_config(config_path, project_root=PROJECT_ROOT)
    model = AHSKTModel(config.model)
    metrics_summary = fit_and_evaluate(
        model=model,
        train_bundle=prepared.train_bundle,
        valid_bundle=prepared.valid_bundle,
        test_bundle=prepared.test_bundle,
        config=config,
    )

    metrics_path = config.output_root / f"{config.task_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "run_config": asdict(run_config),
        "prepared_metadata": prepared.metadata,
        "saved_paths": saved_paths,
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
    }
    manifest_path = output_root / f"{run_config.task_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "saved_paths": saved_paths,
        "prepared_metadata": prepared.metadata,
        "metrics_summary": metrics_summary,
    }


def _apply_runtime_environment(cpu_only: bool) -> None:
    for key, value in THREAD_ENV.items():
        os.environ[key] = value
    if cpu_only or os.environ.get("AHSKT_FORCE_CPU", "0") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _save_prepared_bundles(prepared: BD2006PreparedData, data_output_root: Path, task_name: str):
    train_path = data_output_root / f"{task_name}_train_ahskt.npz"
    valid_path = data_output_root / f"{task_name}_valid_ahskt.npz"
    test_path = data_output_root / f"{task_name}_test_ahskt.npz"
    metadata_path = data_output_root / f"{task_name}_metadata.json"

    np.savez_compressed(train_path, **prepared.train_bundle.as_dict())
    np.savez_compressed(valid_path, **prepared.valid_bundle.as_dict())
    np.savez_compressed(test_path, **prepared.test_bundle.as_dict())
    metadata_path.write_text(json.dumps(prepared.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "train_npz": str(train_path.relative_to(PROJECT_ROOT)),
        "valid_npz": str(valid_path.relative_to(PROJECT_ROOT)),
        "test_npz": str(test_path.relative_to(PROJECT_ROOT)),
        "metadata_json": str(metadata_path.relative_to(PROJECT_ROOT)),
    }


def _build_config_payload(run_config: BD2006RunConfig, prepared: BD2006PreparedData, saved_paths):
    metadata = prepared.metadata
    return {
        "project_name": "ahs-kt",
        "task_name": run_config.task_name,
        "seed": run_config.seed,
        "dataset": {
            "mode": "real",
            "train_path": saved_paths["train_npz"],
            "valid_path": saved_paths["valid_npz"],
            "test_path": saved_paths["test_npz"],
        },
        "model": {
            "num_questions": metadata["num_questions"],
            "num_concepts": metadata["num_concepts"],
            "num_question_difficulty": 101,
            "num_concept_difficulty": 101,
            "num_behavior_clusters": metadata["num_behavior_clusters_with_padding"],
            "sequence_length": run_config.sequence_length,
            "embedding_dim": run_config.embedding_dim,
            "difficulty_dim": run_config.difficulty_dim,
            "behavior_dim": run_config.behavior_dim,
            "hidden_dim": run_config.hidden_dim,
            "dropout": run_config.dropout,
            "use_behavior_cluster": True,
            "use_difficulty_features": True,
            "use_behavior_features": True,
            "use_target_interaction": True,
            "use_relative_speed": False,
            "use_soft_behavior_prototypes": False,
            "question_global_easiness": metadata["question_global_easiness"],
            "concept_global_easiness": metadata["concept_global_easiness"],
            "fusion_mode": "late_residual",
            "behavior_condition_on_difficulty": False,
            "aux_residual_scale": 0.1,
            "difficulty_mode": "smoothed_target_calibration",
            "difficulty_bias_scale": 0.05,
            "difficulty_feature_source": "question_only",
        },
        "training": {
            "epochs": run_config.epochs,
            "batch_size": run_config.batch_size,
            "learning_rate": run_config.learning_rate,
            "patience": run_config.patience,
        },
        "demo": {
            "train_size": 0,
            "valid_size": 0,
            "test_size": 0,
        },
        "outputs": {
            "root_dir": run_config.output_root,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run AHS-KT on BD2006 and export metrics.")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU execution.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--task-name", type=str, default="ahskt_bd2006_notebook", help="Task name prefix.")
    args = parser.parse_args()

    run_config = BD2006RunConfig(
        cpu_only=bool(args.cpu_only),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        task_name=str(args.task_name),
        output_root=f"outputs/{args.task_name}",
        data_output_root=f"data/{args.task_name}",
        config_output_path=f"configs/{args.task_name}.json",
    )
    result = run_bd2006_experiment(run_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
