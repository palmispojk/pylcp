#!/bin/bash
#SBATCH --job-name=bench_gpu
#SBATCH --partition=qist-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bench_gpu_%j.out
#SBATCH --error=logs/bench_gpu_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)"

cd "${REPO_ROOT}"
uv sync --extra cuda

cd "${SLURM_SUBMIT_DIR}"
nvidia-smi || true

# Run each transition in a fresh Python process. Avoids cumulative ptxas /
# XLA cache state that previously crashed the second transition mid-run.
TRANSITIONS=(F0_F1 F0p5_F1p5 F1_F2 F2_F3)
for t in "${TRANSITIONS[@]}"; do
    echo "==================== transition: ${t} ===================="
    uv run --extra cuda python benchmark_gpu.py --transition "${t}" || {
        echo "WARNING: transition ${t} failed; continuing with next."
    }
done
