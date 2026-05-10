import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Train AHS-KT.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument("--demo", action="store_true", help="Force synthetic demo mode.")
    parser.add_argument("--cpu-only", action="store_true", help="Run on CPU only.")
    args = parser.parse_args()

    if args.demo or args.cpu_only or os.environ.get("AHSKT_FORCE_CPU", "0") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow as tf

    if not (args.demo or args.cpu_only or os.environ.get("AHSKT_FORCE_CPU", "0") == "1"):
        for gpu_device in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu_device, True)
            except RuntimeError:
                pass

    from ahskt.config import load_config
    from ahskt.data.synthetic import generate_demo_bundles
    from ahskt.data.dataset import load_bundle_from_config
    from ahskt.models.ahs_kt import AHSKTModel
    from ahskt.training.engine import fit_and_evaluate

    config = load_config(args.config, project_root=PROJECT_ROOT)
    tf.random.set_seed(config.seed)

    if args.demo or config.dataset.mode == "demo":
        train_bundle, valid_bundle, test_bundle = generate_demo_bundles(config)
    else:
        train_bundle, valid_bundle, test_bundle = load_bundle_from_config(config)

    model = AHSKTModel(config.model)
    metrics_summary = fit_and_evaluate(
        model=model,
        train_bundle=train_bundle,
        valid_bundle=valid_bundle,
        test_bundle=test_bundle,
        config=config,
    )

    metrics_path = config.output_root / f"{config.task_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved metrics to {metrics_path}")
    print(json.dumps(metrics_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
