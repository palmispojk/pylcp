#!/bin/bash
#SBATCH --job-name=blue_mot
#SBATCH --partition=qist-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/blue_mot_%j.out
#SBATCH --error=logs/blue_mot_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)"

export PYTHONUNBUFFERED=1
"${REPO_ROOT}/.venv/bin/python" -u blue_mot_sim.py
