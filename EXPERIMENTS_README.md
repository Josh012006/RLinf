# VLA RL Fine-Tuning Experiments on Mila Cluster

This README documents the steps to reproduce RL fine-tuning experiments using the [RLinf](https://github.com/montrealrobotics/RLinf) framework on a single A100L GPU (80GB) on the Mila SLURM cluster.

> ⚠️ **Important**: LIBERO, RoboCasa, and ManiSkill have conflicting dependencies and **cannot share the same virtual environment**. Each experiment uses its own dedicated venv:
> - `.venv-libero` — OpenVLA-OFT + LIBERO
> - `.venv-robocasa` — π0 + RoboCasa
> - `.venv-maniskill` — OpenVLA + ManiSkill

---

## Cluster Notes

### Nodes with known issues

The following nodes have exhibited issues at some point (Vulkan failures, kernel restrictions, or instability). Consider adding them to `--exclude` in your SLURM job script if you encounter problems — the list may evolve over time:

```
cn-g001,cn-g007,cn-g008,cn-g010,cn-g011,cn-g012,cn-g013,cn-g014,cn-g015,cn-g017,cn-g018,cn-g024,cn-g025,cn-g026,cn-d003,cn-i001
```

### Critical job script requirements

Always include the following in your job script:

```bash
# Before exec srun — cleans up stale Ray sessions from previous jobs on the same node
unset RAY_ADDRESS
rm -rf /tmp/ray/session_* 2>/dev/null || true

# In the trap EXIT handler — cleans up after the current job ends
trap "kill $CLEANUP_PID 2>/dev/null; rm -rf /tmp/ray/session_* 2>/dev/null || true" EXIT
```

Use `--no-requeue` in your SLURM directives — RLinf does not support automatic checkpoint resume detection.

---

## Experiment 1: OpenVLA-OFT + LIBERO-Spatial (GRPO)

### Overview

| | |
|---|---|
| Model | OpenVLA-OFT with LoRA |
| Environment | LIBERO-Spatial (10 tasks) |
| Algorithm | GRPO |
| Hardware | 1 × A100L (80GB) |
| Virtual env | `.venv-libero` |
| Config file | `examples/embodiment/config/libero_spatial_grpo_openvlaoft.yaml` |
| Job script | `job-libero-spatial.sh` |

### Step 1 — Clone the repository

```bash
git clone https://github.com/montrealrobotics/RLinf.git
cd RLinf
```

### Step 2 — Install dependencies

```bash
bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero --no-root --venv .venv-libero
source .venv-libero/bin/activate
```

Verify:

```bash
python -c "from libero.libero.envs import OffScreenRenderEnv; print('libero ok')"
```

### Step 3 — Download the model

```bash
pip install huggingface-hub
hf download Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
    --local-dir /path/to/RLinf/Openvla-oft-SFT-libero-spatial-traj1
```

### Step 4 — Configure

In `examples/embodiment/config/libero_spatial_grpo_openvlaoft.yaml`, set model paths and key parameters:

```yaml
rollout:
  model:
    model_path: "/path/to/RLinf/Openvla-oft-SFT-libero-spatial-traj1"

actor:
  model:
    model_path: "/path/to/RLinf/Openvla-oft-SFT-libero-spatial-traj1"
    is_lora: True
    lora_rank: 32
    unnorm_key: libero_spatial_no_noops
    max_prompt_length: 128
```

Key hyperparameters for a single A100L:

```yaml
algorithm:
  group_size: 4
  rollout_epoch: 4
  reward_coef: 5.0
  sampling_params:
    temperature_train: 1.6
    temperature_eval: 1.6
    top_k: -1

env:
  train:
    total_num_envs: 16
    max_episode_steps: 416
    max_steps_per_rollout_epoch: 416

actor:
  micro_batch_size: 4
  global_batch_size: 256
  enable_offload: True
  optim:
    lr: 1.0e-6

rollout:
  enable_offload: True

runner:
  max_epochs: 75
  save_interval: 5
```

### Step 5 — Submit

```bash
sbatch job-libero-spatial.sh
```

### Step 6 — Monitor

```bash
tail -f slurm-JOBID.out
ssh -L 6006:localhost:6006 mila  # then open http://localhost:6006
```

On the compute node:
```bash
tensorboard --logdir ./logs --port 6006
```

Key metric: `env/success_once` — sustained improvement over training steps.

### Resuming from a checkpoint

```yaml
runner:
  resume_dir: "/path/to/RLinf/logs/TIMESTAMP-libero_spatial_grpo_openvlaoft/libero_spatial_grpo_openvlaoft/checkpoints/global_step_X"
```

---

## Experiment 2: π0 + RoboCasa CloseDrawer (PPO)

### Overview

| | |
|---|---|
| Model | π0 (PaliGemma + flow matching action head) with LoRA |
| Environment | RoboCasa — CloseDrawer (single task) |
| Algorithm | PPO |
| Hardware | 1 × A100L (80GB) |
| Virtual env | `.venv-robocasa` |
| Config file | `examples/embodiment/config/robocasa_closedrawer_ppo_openpi.yaml` |
| Job script | `job-robocasa.sh` |

### Step 1 — Clone the repository

```bash
git clone https://github.com/montrealrobotics/RLinf.git
cd RLinf
```

### Step 2 — Install dependencies

```bash
bash requirements/install.sh embodied --model openpi --env robocasa --no-root --venv .venv-robocasa
source .venv-robocasa/bin/activate
```

Verify:

```bash
python -c "import robocasa; print('robocasa ok')"
```

### Step 3 — Download kitchen assets

```bash
python -m robocasa.scripts.download_kitchen_assets
```

> Downloads ~5GB of kitchen assets. Run on a compute node, not the login node.

### Step 4 — Download the model

```bash
pip install huggingface-hub
hf download RLinf/RLinf-Pi0-RoboCasa \
    --local-dir /path/to/RLinf/RLinf-Pi0-RoboCasa
```

### Step 5 — Configure

In `examples/embodiment/config/robocasa_closedrawer_ppo_openpi.yaml`, set model paths and key parameters:

```yaml
rollout:
  model:
    model_path: "/path/to/RLinf/RLinf-Pi0-RoboCasa"

actor:
  model:
    model_path: "/path/to/RLinf/RLinf-Pi0-RoboCasa"
    is_lora: True
    lora_rank: 32
```

Key hyperparameters for a single A100L:

```yaml
algorithm:
  group_size: 1
  rollout_epoch: 2
  update_epoch: 2
  sampling_params:
    temperature_train: 1.0
    temperature_eval: 0.6
    top_k: 50

env:
  train:
    total_num_envs: 8
    max_episode_steps: 240
    max_steps_per_rollout_epoch: 240
  enable_offload: True

actor:
  micro_batch_size: 4      # keep at 4 to avoid OOM with LoRA rank 32
  global_batch_size: 64
  enable_offload: True
  optim:
    lr: 1.0e-6

rollout:
  enable_offload: True

runner:
  max_epochs: 75
  save_interval: 5
```

### Step 6 — Submit

```bash
sbatch job-robocasa.sh
```

### Step 7 — Monitor

```bash
tail -f slurm-JOBID.out
```

Key metrics: `env/success_once`, `actor/approx_kl` (should stay below 0.05).

### Resuming from a checkpoint

```yaml
runner:
  resume_dir: "/path/to/RLinf/logs/TIMESTAMP-robocasa_closedrawer_ppo_openpi/robocasa_closedrawer_ppo_openpi/checkpoints/global_step_X"
```

---

## Experiment 3: OpenVLA + ManiSkill StackCube (GRPO & PPO)

### Overview

| | |
|---|---|
| Model | OpenVLA base (`openvla-7b-rlvla-warmup`) — no task-specific SFT |
| Environment | ManiSkill — StackCube-v1 |
| Algorithms | GRPO and PPO (run in parallel) |
| Hardware | 1 × A100L (80GB) per run |
| Virtual env | `.venv-maniskill` |
| Config files | `examples/embodiment/config/maniskill_grpo_openvla_stackcube.yaml` |
| | `examples/embodiment/config/maniskill_ppo_openvla_stackcube.yaml` |
| Job scripts | `job-maniskill-stackcube-grpo.sh` |
| | `job-maniskill-stackcube-ppo.sh` |

> **Research context**: This experiment tests whether RL fine-tuning alone (no SFT demos) can teach OpenVLA a task completely absent from its pretraining distribution (OXE Magic Soup++). StackCube is not present in any OXE dataset, making it a true OOD test. The base model starts with ~0% success rate.

### Step 1 — Clone the repository

```bash
git clone https://github.com/montrealrobotics/RLinf.git
cd RLinf
```

### Step 2 — Install dependencies

```bash
bash requirements/install.sh embodied --model openvla --env maniskill --no-root --venv .venv-maniskill
source .venv-maniskill/bin/activate
```

Verify:

```bash
python -c "import mani_skill; print('maniskill ok')"
```

### Step 3 — Vulkan setup (required for ManiSkill GPU rendering)

ManiSkill requires Vulkan for GPU-accelerated rendering. Since `libvulkan.so` is not installed on Mila compute nodes, install it locally without sudo:

```bash
VULKAN_DIR=~/projects/libero_rl/vulkan
mkdir -p $VULKAN_DIR
cd $VULKAN_DIR
apt-get download libvulkan1
dpkg -x libvulkan1_*.deb $VULKAN_DIR
cd -
```

Add the following to your job script before `exec srun`:

```bash
export LD_LIBRARY_PATH=~/projects/libero_rl/vulkan/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### Step 4 — Download the model

```bash
pip install huggingface-hub
hf download gen-robot/openvla-7b-rlvla-warmup \
    --local-dir /path/to/RLinf/openvla-7b-rlvla-warmup
```

### Step 5 — Configure (GRPO)

In `examples/embodiment/config/maniskill_grpo_openvla_stackcube.yaml`, set model paths and key parameters:

```yaml
rollout:
  model:
    model_path: "/path/to/RLinf/openvla-7b-rlvla-warmup"

actor:
  model:
    model_path: "/path/to/RLinf/openvla-7b-rlvla-warmup"
```

Key hyperparameters for a single A100L:

```yaml
algorithm:
  group_size: 8
  rollout_epoch: 1
  sampling_params:
    temperature_train: 1.0
    temperature_eval: 0.6
    top_k: 50

env:
  train:
    total_num_envs: 16
    max_episode_steps: 320
    max_steps_per_rollout_epoch: 320  # 16 × 320 = 5120, 5120 % 128 = 0 ✓
    reward_mode: raw
    enable_offload: False

actor:
  micro_batch_size: 4
  global_batch_size: 128             # 5120 % 128 = 0 ✓
  enable_offload: True
  optim:
    lr: 1.0e-5

rollout:
  enable_offload: True

runner:
  max_epochs: 250
  save_interval: 5
```

### Step 6 — Configure (PPO)

In `examples/embodiment/config/maniskill_ppo_openvla_stackcube.yaml`:

```yaml
algorithm:
  group_size: 1
  rollout_epoch: 1
  sampling_params:
    temperature_train: 1.0
    temperature_eval: 0.6
    top_k: 0

env:
  train:
    total_num_envs: 16
    max_episode_steps: 300
    max_steps_per_rollout_epoch: 300  # 16 × 300 = 4800, 4800 % 160 = 0 ✓
    reward_mode: raw
    enable_offload: False

actor:
  micro_batch_size: 4
  global_batch_size: 160             # 4800 % 160 = 0, 160 % 4 = 0 ✓
  add_value_head: True
  is_lora: True
  enable_offload: True
  optim:
    lr: 1.0e-5                       # ⚠️ do NOT use 1e-4 — causes divergence

rollout:
  enable_offload: False              # enable_offload in env must match rollout

runner:
  max_epochs: 250
  save_interval: 5
```

> ⚠️ **PPO warning**: Using `lr > 1e-5` causes catastrophic divergence (`approx_kl → 20+`, `ratio → 0`) within 15 steps. Always use `lr: 1.0e-5`.

### Step 7 — Submit

```bash
# GRPO
sbatch job-maniskill-stackcube-grpo.sh

# PPO (separate node)
sbatch job-maniskill-stackcube-ppo.sh
```

### Step 8 — Monitor

```bash
tail -f slurm-JOBID.out
```

Key metrics:
- `env/success_once` — primary metric (starts at 0.0, target > 0)
- `env/return` — with `use_rel_reward: True`, values ~0.04-0.06 = random policy, >0.10 = robot approaching cube
- `actor/approx_kl` — `inf` is normal for GRPO; for PPO should stay below 0.2
- `critic/explained_variance` (PPO only) — starts at -100, should reach 0.5+ by step 10

### Resuming from a checkpoint

```yaml
runner:
  resume_dir: "/path/to/RLinf/logs/TIMESTAMP-maniskill_grpo_openvla_stackcube/maniskill_grpo_openvla_stackcube/checkpoints/global_step_X"
```

---

## Known Issues

| Issue | Cause | Fix |
|---|---|---|
| `KeyError: '3rd_view_camera'` | ManiSkill native tasks use `base_camera`, not `3rd_view_camera` | Set `wrap_obs_mode: simple` in env yaml |
| `TypeError: 'NoneType' object is not iterable` | `StackCubeEnv` has no `get_language_instruction()` | Set `task_description` in `init_params` of env yaml |
| `TypeError: unexpected keyword argument 'task_description'` | `task_description` passed to `gym.make()` | Fixed in `maniskill_env.py` — `env_args.pop('task_description', None)` |
| `RuntimeError: pidfd_getfd: Operation not permitted` | `enable_offload: True` in env on restricted nodes | Set `enable_offload: False` in `env.train` and `env.eval` |
| `AssertionError: 4800 is not divisible by 128` | `total_num_envs × max_steps` not divisible by `global_batch_size` | Ensure `(total_num_envs × max_steps_per_rollout_epoch) % global_batch_size == 0` |
| Ray hang on resubmit | `RAY_ADDRESS` still set from previous session | Always `unset RAY_ADDRESS` and `rm -rf /tmp/ray/session_*` before launch |
| PPO divergence | `lr` too high | Always use `lr: 1.0e-5` for PPO with OpenVLA |