import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    file_prefix = "assist2009_v5"
    config_output = PROJECT_ROOT / "configs" / "ahskt_assist2009_v5.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_assist2009_ahskt.py",
            "--file-prefix",
            file_prefix,
            "--question-alpha",
            "5",
            "--concept-alpha",
            "20",
            "--config-output",
            str(config_output),
            "--task-name",
            "ahskt_assist2009_v5",
            "--outputs-root",
            "outputs/assist2009_v5_run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    config = json.loads(config_output.read_text(encoding="utf-8"))
    config["model"].update(
        {
            "use_difficulty_features": True,
            "use_behavior_features": True,
            "use_target_interaction": True,
            "fusion_mode": "late_residual",
            "behavior_condition_on_difficulty": False,
            "aux_residual_scale": 0.1,
            "difficulty_mode": "smoothed_target_calibration",
            "difficulty_bias_scale": 0.05,
            "difficulty_feature_source": "question_only",
        }
    )
    config_output.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"config_output": str(config_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
