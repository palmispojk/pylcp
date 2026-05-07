#!/bin/bash
#SBATCH --job-name=bb_red_mot
#SBATCH --partition=qist-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=logs/bb_red_mot_%j.out
#SBATCH --error=logs/bb_red_mot_%j.err

set -euo pipefail

# Upstream stage that feeds this MOT. Point at any <stage>_final_state.pkl;
# the script will auto-load the sibling constants.py for unit rescaling.
#   Full pipeline:   ../low_power_blue_mot/low_power_blue_mot_final_state.pkl
#   Skip low-power:  ../blue_mot/blue_mot_final_state.pkl
UPSTREAM="../blue_mot/blue_mot_final_state.pkl"

cd "${SLURM_SUBMIT_DIR}"

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)"

export PYTHONUNBUFFERED=1
"${REPO_ROOT}/.venv/bin/python" -u bb_red_mot_sim.py --upstream "${UPSTREAM}"
