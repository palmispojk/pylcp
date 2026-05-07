#!/bin/bash
#SBATCH --job-name=sf_red_mot
#SBATCH --partition=qist-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sf_red_mot_%j.out
#SBATCH --error=logs/sf_red_mot_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)"

cd "${REPO_ROOT}"
uv sync --extra cuda

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
uv run --extra cuda python -u sf_red_mot_sim.py
