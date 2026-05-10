import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_config_payload(metadata, train_path, valid_path, test_path, task_name, output_root):
    return {
        "project_name": "ahs-kt",
        "task_name": task_name,
        "seed": 2026,
        "dataset": {
            "mode": "real_npz",
            "train_path": train_path,
            "valid_path": valid_path,
            "test_path": test_path,
        },
        "model": {
            "num_questions": int(metadata["num_questions"]),
            "num_concepts": int(metadata["num_concepts"]),
            "num_question_difficulty": int(metadata["num_question_difficulty"]),
            "num_concept_difficulty": int(metadata["num_concept_difficulty"]),
            "num_behavior_clusters": int(metadata["num_behavior_clusters"]),
            "sequence_length": int(metadata["sequence_length"]),
            "embedding_dim": 64,
            "difficulty_dim": 32,
            "behavior_dim": 32,
            "hidden_dim": 96,
            "dropout": 0.2,
            "use_behavior_cluster": True,
            "use_difficulty_features": True,
            "use_behavior_features": True,
            "use_target_interaction": True,
            "use_relative_speed": True,
            "use_soft_behavior_prototypes": True,
            "question_global_easiness": float(metadata.get("smoothing", {}).get("global_easiness", 0.5)),
            "concept_global_easiness": float(metadata.get("smoothing", {}).get("global_easiness", 0.5)),
        },
        "training": {
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 0.001,
            "patience": 2,
        },
        "demo": {
            "train_size": 0,
            "valid_size": 0,
            "test_size": 0,
        },
        "outputs": {
            "root_dir": output_root,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build AHS-KT assist2009 bundles from LBKT data.")
    parser.add_argument(
        "--data-dir",
        default="/root/autodl-tmp/LBKT/data2",
        help="Path to LBKT assist2009 data directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data"),
        help="Directory used to save generated NPZ bundles and metadata.",
    )
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--remainder-min-len", type=int, default=10)
    parser.add_argument(
        "--config-output",
        default=str(PROJECT_ROOT / "configs" / "ahskt_assist2009_v1.json"),
        help="Output path for generated training config.",
    )
    parser.add_argument(
        "--task-name",
        default="ahskt_assist2009_v1",
        help="Task name written into the generated config.",
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs/assist2009_v1_run",
        help="Relative output root written into the generated config.",
    )
    parser.add_argument(
        "--file-prefix",
        default="assist2009",
        help="Prefix used in generated NPZ and metadata file names.",
    )
    parser.add_argument("--question-alpha", type=float, default=0.0)
    parser.add_argument("--concept-alpha", type=float, default=0.0)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--cluster-sample-size", type=int, default=0)
    args = parser.parse_args()

    from ahskt.data.assist2009 import build_assist2009_bundles, save_bundle_npz

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_bundle, valid_bundle, test_bundle, metadata = build_assist2009_bundles(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        remainder_min_len=args.remainder_min_len,
        question_alpha=args.question_alpha,
        concept_alpha=args.concept_alpha,
        n_clusters=args.n_clusters,
        random_seed=args.random_seed,
        cluster_sample_size=args.cluster_sample_size,
    )

    train_path = output_dir / f"{args.file_prefix}_train_ahskt.npz"
    valid_path = output_dir / f"{args.file_prefix}_valid_ahskt.npz"
    test_path = output_dir / f"{args.file_prefix}_test_ahskt.npz"
    metadata_path = output_dir / f"{args.file_prefix}_metadata.json"

    save_bundle_npz(train_bundle, train_path)
    save_bundle_npz(valid_bundle, valid_path)
    save_bundle_npz(test_bundle, test_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    config_payload = build_config_payload(
        metadata=metadata,
        train_path=str(train_path.relative_to(PROJECT_ROOT)),
        valid_path=str(valid_path.relative_to(PROJECT_ROOT)),
        test_path=str(test_path.relative_to(PROJECT_ROOT)),
        task_name=args.task_name,
        output_root=args.outputs_root,
    )
    config_output = Path(args.config_output)
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "train_path": str(train_path),
                "valid_path": str(valid_path),
                "test_path": str(test_path),
                "metadata_path": str(metadata_path),
                "config_output": str(config_output),
                "split_summary": metadata["split_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
