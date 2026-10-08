"""Run all model-level statistics scripts for one saved training run."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation, significance tests, and sanity checks for one model."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Run directory containing model/, or the model directory itself.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    model_dir = Path(args.model_dir).expanduser().resolve()
    scripts = ("evaluate.py", "significance.py", "sanity_checks.py")
    for name in scripts:
        command = [
            sys.executable,
            str(root / name),
            "--model-dir",
            str(model_dir),
        ]
        print(f"\nRunning {name}...", flush=True)
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
