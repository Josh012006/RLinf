#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=256G
#SBATCH --time=120:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com
#SBATCH --no-requeue

# Job script for ManiSkill StackCube GRPO fine-tuning with OpenVLA on a single A100L
# Requires: RLinf installed in .venv-maniskill, openvla-7b-rlvla-warmup model downloaded
# Adapt model paths in examples/embodiment/config/maniskill_grpo_openvla_stackcube.yaml before running
# Submit with: sbatch job-maniskill-stackcube-grpo.sh

set -e
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cleanup loop: automatically deletes old checkpoints, keeping only the 2 most recent
# Adapt the path below to match your project directory and experiment name
(
  while true; do
    sleep 1800
    ls -dt /path/to/RLinf/logs/*/maniskill_grpo_openvla_stackcube/checkpoints/global_step_* 2>/dev/null | tail -n +3 | xargs rm -rf
  done
) &
CLEANUP_PID=$!

trap "kill $CLEANUP_PID 2>/dev/null; rm -rf /tmp/ray/session_* 2>/dev/null || true" EXIT

# Adapt the path below to your project directory
cd /path/to/RLinf
source .venv-maniskill/bin/activate
unset RAY_ADDRESS

# Adapt the paths here too
# Install libvulkan if not already present
VULKAN_DIR=/path/to/vulkan
if [ ! -f "$VULKAN_DIR/usr/lib/x86_64-linux-gnu/libvulkan.so.1" ]; then
    mkdir -p $VULKAN_DIR
    cd $VULKAN_DIR
    apt-get download libvulkan1
    dpkg -x libvulkan1_*.deb $VULKAN_DIR
    cd /path/to/RLinf
fi
export LD_LIBRARY_PATH=$VULKAN_DIR/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

exec srun bash examples/embodiment/run_embodiment.sh maniskill_grpo_openvla_stackcube
