import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


EXPERIMENT_PROFILES = {
    "legacy": [
        {
            "name": "target_only",
            "config_path": PROJECT_ROOT / "configs" / "ahskt_assist2012_ablation_target_only.json",
        },
        {
            "name": "target_difficulty",
            "config_path": PROJECT_ROOT / "configs" / "ahskt_assist2012_ablation_target_difficulty.json",
        },
        {
            "name": "target_difficulty_behavior_cluster",
            "config_path": PROJECT_ROOT / "configs" / "ahskt_assist2012_v2.json",
        },
    ],
    "formal": [
        {
            "name": "target_only",
            "config_path": PROJECT_ROOT / "configs" / "ahskt_assist2012_formal_target_only.json",
        },
        {
            "name": "target_difficulty",
            "config_path": PROJECT_ROOT / "configs" / "ahskt_assist2012_formal_target_difficulty.json",
        },
        {
            "name": "target_difficulty_behavior_cluster",
            "config_path": PROJECT_ROOT / "configs" / "ahskt_assist2012_formal_full.json",
        },
    ],
}


def load_json(file_path):
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


def save_json(file_path, payload):
    Path(file_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_seeds(seed_text):
    return [int(item.strip()) for item in str(seed_text).split(",") if item.strip()]


def build_run_config(experiment, seed, generated_config_dir, run_root_base):
    base_config = load_json(experiment["config_path"])
    run_task_name = f'{base_config["task_name"]}_s{seed}'
    run_output_root = run_root_base / experiment["name"] / f"s{seed}"

    base_config["seed"] = int(seed)
    base_config["task_name"] = run_task_name
    base_config["outputs"]["root_dir"] = str(run_output_root.relative_to(PROJECT_ROOT))

    generated_config_dir.mkdir(parents=True, exist_ok=True)
    config_output_path = generated_config_dir / f'{experiment["name"]}_s{seed}.json'
    save_json(config_output_path, base_config)
    return base_config, config_output_path, run_output_root


def run_experiment(experiment, seed, generated_config_dir, run_root_base, reuse_existing=False):
    config, config_path, output_root = build_run_config(
        experiment=experiment,
        seed=seed,
        generated_config_dir=generated_config_dir,
        run_root_base=run_root_base,
    )
    metrics_path = output_root / f'{config["task_name"]}_metrics.json'

    if not reuse_existing:
        shutil.rmtree(output_root, ignore_errors=True)

    if reuse_existing and metrics_path.exists():
        metrics = load_json(metrics_path)
        return config, metrics, metrics_path

    subprocess.run(
        [sys.executable, "scripts/train_ahskt.py", "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    metrics = load_json(metrics_path)
    return config, metrics, metrics_path


def make_raw_row(experiment_name, seed, metrics, metrics_path):
    return {
        "experiment": experiment_name,
        "seed": int(seed),
        "task_name": metrics["task_name"],
        "best_epoch": int(metrics["best_epoch"]),
        "best_valid_auc": float(metrics["best_valid_auc"]),
        "test_auc": float(metrics["test_metrics"]["auc"]),
        "test_acc": float(metrics["test_metrics"]["acc"]),
        "test_rmse": float(metrics["test_metrics"]["rmse"]),
        "test_loss": float(metrics["test_metrics"]["loss"]),
        "metrics_path": str(metrics_path),
    }


def _metric_stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "var": float(np.var(array, ddof=0)),
    }


def aggregate_experiment_rows(rows):
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row["experiment"]].append(row)

    aggregate_rows = []
    for experiment_name, experiment_rows in grouped_rows.items():
        experiment_rows = sorted(experiment_rows, key=lambda item: item["seed"])
        best_valid_auc_stats = _metric_stats([row["best_valid_auc"] for row in experiment_rows])
        test_auc_stats = _metric_stats([row["test_auc"] for row in experiment_rows])
        test_acc_stats = _metric_stats([row["test_acc"] for row in experiment_rows])
        test_rmse_stats = _metric_stats([row["test_rmse"] for row in experiment_rows])
        test_loss_stats = _metric_stats([row["test_loss"] for row in experiment_rows])

        aggregate_rows.append(
            {
                "experiment": experiment_name,
                "seeds": ",".join(str(row["seed"]) for row in experiment_rows),
                "runs": len(experiment_rows),
                "best_valid_auc_mean": best_valid_auc_stats["mean"],
                "best_valid_auc_std": best_valid_auc_stats["std"],
                "best_valid_auc_var": best_valid_auc_stats["var"],
                "best_valid_auc_mean_pm_std": f'{best_valid_auc_stats["mean"]:.6f}±{best_valid_auc_stats["std"]:.6f}',
                "test_auc_mean": test_auc_stats["mean"],
                "test_auc_std": test_auc_stats["std"],
                "test_auc_var": test_auc_stats["var"],
                "test_auc_mean_pm_std": f'{test_auc_stats["mean"]:.6f}±{test_auc_stats["std"]:.6f}',
                "test_acc_mean": test_acc_stats["mean"],
                "test_acc_std": test_acc_stats["std"],
                "test_acc_var": test_acc_stats["var"],
                "test_acc_mean_pm_std": f'{test_acc_stats["mean"]:.6f}±{test_acc_stats["std"]:.6f}',
                "test_rmse_mean": test_rmse_stats["mean"],
                "test_rmse_std": test_rmse_stats["std"],
                "test_rmse_var": test_rmse_stats["var"],
                "test_rmse_mean_pm_std": f'{test_rmse_stats["mean"]:.6f}±{test_rmse_stats["std"]:.6f}',
                "test_loss_mean": test_loss_stats["mean"],
                "test_loss_std": test_loss_stats["std"],
                "test_loss_var": test_loss_stats["var"],
                "test_loss_mean_pm_std": f'{test_loss_stats["mean"]:.6f}±{test_loss_stats["std"]:.6f}',
            }
        )

    return sorted(aggregate_rows, key=lambda item: item["experiment"])


def write_csv(file_path, rows, fieldnames):
    with Path(file_path).open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summaries(raw_rows, aggregate_rows, output_prefix):
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    raw_json_path = outputs_dir / f"{output_prefix}_raw.json"
    raw_csv_path = outputs_dir / f"{output_prefix}_raw.csv"
    agg_json_path = outputs_dir / f"{output_prefix}_aggregate.json"
    agg_csv_path = outputs_dir / f"{output_prefix}_aggregate.csv"

    save_json(raw_json_path, raw_rows)
    save_json(agg_json_path, aggregate_rows)
    write_csv(
        raw_csv_path,
        raw_rows,
        fieldnames=[
            "experiment",
            "seed",
            "task_name",
            "best_epoch",
            "best_valid_auc",
            "test_auc",
            "test_acc",
            "test_rmse",
            "test_loss",
            "metrics_path",
        ],
    )
    write_csv(
        agg_csv_path,
        aggregate_rows,
        fieldnames=[
            "experiment",
            "seeds",
            "runs",
            "best_valid_auc_mean",
            "best_valid_auc_std",
            "best_valid_auc_var",
            "best_valid_auc_mean_pm_std",
            "test_auc_mean",
            "test_auc_std",
            "test_auc_var",
            "test_auc_mean_pm_std",
            "test_acc_mean",
            "test_acc_std",
            "test_acc_var",
            "test_acc_mean_pm_std",
            "test_rmse_mean",
            "test_rmse_std",
            "test_rmse_var",
            "test_rmse_mean_pm_std",
            "test_loss_mean",
            "test_loss_std",
            "test_loss_var",
            "test_loss_mean_pm_std",
        ],
    )
    return raw_json_path, raw_csv_path, agg_json_path, agg_csv_path


def main():
    parser = argparse.ArgumentParser(description="Run ASSIST2012 ablation experiments with multiple seeds.")
    parser.add_argument("--seeds", default="2024,2025,2026", help="Comma-separated random seeds.")
    parser.add_argument(
        "--profile",
        default="formal",
        choices=sorted(EXPERIMENT_PROFILES.keys()),
        help="Config profile to run. Use 'formal' for the final late_residual thesis setup, or 'legacy' for the older setup.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix used for raw/aggregate summary files under outputs/.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing metrics if the run-specific metrics file already exists.",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    experiments = EXPERIMENT_PROFILES[args.profile]
    output_prefix = args.output_prefix
    if not output_prefix:
        if args.profile == "formal":
            output_prefix = "assist2012_ablation_formal_seed3"
        else:
            output_prefix = "assist2012_ablation_seed3"
    generated_config_dir = PROJECT_ROOT / "outputs" / "generated_configs" / output_prefix
    run_root_base = PROJECT_ROOT / "outputs" / output_prefix

    raw_rows = []
    for experiment in experiments:
        for seed in seeds:
            _, metrics, metrics_path = run_experiment(
                experiment=experiment,
                seed=seed,
                generated_config_dir=generated_config_dir,
                run_root_base=run_root_base,
                reuse_existing=args.reuse_existing,
            )
            raw_rows.append(make_raw_row(experiment["name"], seed, metrics, metrics_path))

    aggregate_rows = aggregate_experiment_rows(raw_rows)
    raw_json_path, raw_csv_path, agg_json_path, agg_csv_path = save_summaries(
        raw_rows=raw_rows,
        aggregate_rows=aggregate_rows,
        output_prefix=output_prefix,
    )

    print(json.dumps(aggregate_rows, ensure_ascii=False, indent=2))
    print(f"saved raw summary to {raw_json_path}")
    print(f"saved raw summary to {raw_csv_path}")
    print(f"saved aggregate summary to {agg_json_path}")
    print(f"saved aggregate summary to {agg_csv_path}")


if __name__ == "__main__":
    main()
