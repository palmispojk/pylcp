#!/bin/bash
#SBATCH --job-name=bench_cpu
#SBATCH --partition=qist-fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bench_cpu_%j.out
#SBATCH --error=logs/bench_cpu_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)"

cd "${REPO_ROOT}"
uv sync

cd "${SLURM_SUBMIT_DIR}"
uv run python benchmark_cpu.py
