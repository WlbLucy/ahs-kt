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
            "use_relative_speed": True,
            "use_soft_behavior_prototypes": True,
        },
        "training": {
            "epochs": 5,
            "batch_size": 64,
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
    parser = argparse.ArgumentParser(description="Build AHS-KT assist2012 bundles from DIMKT raw data.")
    parser.add_argument(
        "--csv-path",
        default="/root/autodl-tmp/DIMKT/data/2012-2013-data-with-predictions-4-final.csv",
        help="Path to the ASSIST2012 raw CSV.",
    )
    parser.add_argument(
        "--dimkt-data-dir",
        default="/root/autodl-tmp/DIMKT/data",
        help="Path to DIMKT data directory containing mappings and splits.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data"),
        help="Directory used to save generated NPZ bundles and metadata.",
    )
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--cluster-sample-size", type=int, default=200000)
    parser.add_argument("--skip-validate-dimkt", action="store_true")
    parser.add_argument(
        "--config-output",
        default=str(PROJECT_ROOT / "configs" / "ahskt_assist2012_v1.json"),
        help="Output path for generated training config.",
    )
    parser.add_argument(
        "--task-name",
        default="ahskt_assist2012_v1",
        help="Task name written into the generated config.",
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs/assist2012_v1_run",
        help="Relative output root written into the generated config.",
    )
    args = parser.parse_args()

    from ahskt.data.assist2012 import build_assist2012_bundles, save_bundle_npz

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_bundle, valid_bundle, test_bundle, metadata = build_assist2012_bundles(
        csv_path=args.csv_path,
        dimkt_data_dir=args.dimkt_data_dir,
        sequence_length=args.sequence_length,
        n_clusters=args.n_clusters,
        random_seed=args.random_seed,
        cluster_sample_size=args.cluster_sample_size,
        validate_dimkt=not args.skip_validate_dimkt,
    )

    train_path = output_dir / "assist2012_train_ahskt.npz"
    valid_path = output_dir / "assist2012_valid_ahskt.npz"
    test_path = output_dir / "assist2012_test_ahskt.npz"
    metadata_path = output_dir / "assist2012_metadata.json"

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
