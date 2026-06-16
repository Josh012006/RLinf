#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=256G
#SBATCH --time=120:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --no-requeue
#SBATCH --exclude=cn-g011,cn-g008,cn-g012,cn-g026,cn-g025,cn-g015,cn-g017

# Job script for LIBERO-Long GRPO fine-tuning with OpenVLA-OFT on a single A100L
# Requires: RLinf installed in .venv-libero, Openvla-oft-SFT-libero10-traj1 model downloaded
# Adapt model paths in examples/embodiment/config/libero_10_grpo_openvlaoft.yaml before running
# Submit with: sbatch job-libero-long.sh

set -e
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cleanup loop: automatically deletes old checkpoints, keeping only the 2 most recent
# Adapt the path below to match your project directory and experiment name
(
  while true; do
    sleep 1800
    ls -dt ~/projects/libero_rl/RLinf/logs/*/libero_10_grpo_openvlaoft/checkpoints/global_step_* 2>/dev/null | tail -n +3 | xargs rm -rf
  done
) &
CLEANUP_PID=$!

trap "kill $CLEANUP_PID 2>/dev/null" EXIT

# Adapt the path below to your project directory
cd ~/projects/libero_rl/RLinf
source .venv-libero/bin/activate
unset RAY_ADDRESS
exec srun bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft
