#!/bin/bash
#SBATCH --job-name=sf_red_mot
#SBATCH --partition=qist-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=logs/sf_red_mot_%j.out
#SBATCH --error=logs/sf_red_mot_%j.err

set -euo pipefail

# Upstream stage that feeds this MOT. Point at any <stage>_final_state.pkl.
UPSTREAM="../bb_red_mot/no_weak_blue_run/bb_red_mot_final_state.pkl"

cd "${SLURM_SUBMIT_DIR}"

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)"

export PYTHONUNBUFFERED=1
"${REPO_ROOT}/.venv/bin/python" -u sf_red_mot_sim.py --upstream "${UPSTREAM}"
