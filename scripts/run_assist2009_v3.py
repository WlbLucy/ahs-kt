import argparse
from pathlib import Path

import run_assist2009_ablation


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run ASSIST2009 AHS-KT v3 with multiple seeds.")
    parser.add_argument("--seeds", default="2023,2024,2025")
    parser.add_argument("--output-prefix", default="assist2009_v3_seed3")
    parser.add_argument("--config", default="configs/ahskt_assist2009_v3.json")
    parser.add_argument("--experiment-name", default="target_difficulty_behavior_cluster_v3")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    experiment = {
        "name": args.experiment_name,
        "config_path": PROJECT_ROOT / args.config,
    }
    seeds = run_assist2009_ablation.parse_seeds(args.seeds)
    generated_config_dir = PROJECT_ROOT / "outputs" / "generated_configs" / args.output_prefix
    run_root_base = PROJECT_ROOT / "outputs" / args.output_prefix
    raw_rows = []

    for seed in seeds:
        _, metrics, metrics_path = run_assist2009_ablation.run_experiment(
            experiment=experiment,
            seed=seed,
            generated_config_dir=generated_config_dir,
            run_root_base=run_root_base,
            reuse_existing=args.reuse_existing,
        )
        raw_rows.append(
            run_assist2009_ablation.make_raw_row(
                experiment_name=experiment["name"],
                seed=seed,
                metrics=metrics,
                metrics_path=metrics_path,
            )
        )

    aggregate_rows = run_assist2009_ablation.aggregate_experiment_rows(raw_rows)
    run_assist2009_ablation.save_summaries(raw_rows, aggregate_rows, args.output_prefix)
    print(aggregate_rows)


if __name__ == "__main__":
    main()
