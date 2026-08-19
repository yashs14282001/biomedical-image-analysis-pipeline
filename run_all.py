from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(command))
    print("=" * 80)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tasks 1-4 end-to-end.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-llm", action="store_true", help="Debug only; marked submission should run LLM steps.")
    args = parser.parse_args()

    if not args.skip_download:
        run("00_download_dataset.py")
    run("01_prepare_eda.py")
    if not args.skip_llm:
        run("02_task1_vlm.py")
        run("03_task2_classical_llm.py")
    else:
        run("03_task2_classical_llm.py", "--no-llm")
    run("04_task3_train_unet.py", "--epochs", str(args.epochs), "--loss", "bce_dice")
    hybrid_args = ["--no-llm"] if args.skip_llm else []
    run("05_task4_hybrid_pipeline.py", *hybrid_args)
    print("\nPipeline complete. Check the outputs/ directory for report-ready figures, JSON, CSV, and metrics.")


if __name__ == "__main__":
    main()
