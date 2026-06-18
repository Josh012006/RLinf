# Adding a New ManiSkill Native Task to RLinf

This guide documents how to integrate a new ManiSkill native task (e.g. `StackCube-v1`, `PickClutter-v1`) into the RLinf pipeline with OpenVLA. It is distinct from RLinf's custom tasks (e.g. `PutOnPlateInScene25Main-v3`) which have their own infrastructure.

---

## Overview of required changes

Adding a new ManiSkill native task requires:

1. A new **env config yaml** (`examples/embodiment/config/env/maniskill_<task>.yaml`)
2. A new **training config yaml** for each algorithm (GRPO, PPO)
3. No code modifications needed — the pipeline is now generic

---

## Step 1 — Check the task's available cameras and info keys

Before configuring, verify what the task exposes:

```python
import gymnasium as gym
import mani_skill.envs

env = gym.make('YourTask-v1', obs_mode='rgb', render_mode='all', num_envs=1, sim_backend='gpu')
obs, info = env.reset()

# Camera names available
print('cameras:', list(obs['sensor_data'].keys()))

# Info keys (for reward shaping)
action = env.action_space.sample()
obs, reward, term, trunc, info = env.step(action)
print('info keys:', list(info.keys()))
print('native reward:', reward)

# Language instruction support
print('has get_language_instruction:', hasattr(env.unwrapped, 'get_language_instruction'))
```

---

## Step 2 — Create the env config yaml

Create `examples/embodiment/config/env/maniskill_<task>.yaml`:

```yaml
env_type: maniskill
wrap_obs_mode: simple          # required for ManiSkill native tasks
reward_mode: raw               # uses ManiSkill's native dense reward
total_num_envs: null
auto_reset: False
ignore_terminations: False
use_rel_reward: True
seed: 0
group_size: 1
use_fixed_reset_state_ids: False
max_steps_per_rollout_epoch: 300
max_episode_steps: 300

video_cfg:
  save_video: True
  info_on_video: True
  video_base_dir: ${runner.logger.log_path}/video/train

enable_offload: False          # keep False — enable_offload: True causes pidfd_getfd errors on some nodes

init_params:
  id: "YourTask-v1"
  task_description: "your task description here"   # required if task has no get_language_instruction()
  num_envs: null
  obs_mode: "rgb"              # do NOT use rgb+segmentation — not compatible with wrap_obs_mode: simple
  control_mode: pd_ee_delta_pose
  sim_backend: "gpu"
  sim_config:
    sim_freq: 500
    control_freq: 5
  max_episode_steps: ${env.train.max_episode_steps}
  sensor_configs:
    shader_pack: "default"
  render_mode: all
```

> **`task_description`**: required if the task does not implement `get_language_instruction()`. Most ManiSkill native tasks do not. Check with `hasattr(env.unwrapped, 'get_language_instruction')`.

> **`control_mode`**: `pd_ee_delta_pose` (delta end-effector) is the recommended action space for OpenVLA — easiest to learn and consistent with the base model's pretraining.

> **`max_episode_steps`**: allow enough steps for the task. A rule of thumb: `steps × (control_freq / sim_freq)` should be at least 30-60 seconds of simulation. For tabletop manipulation, 200-320 steps is typical.

---

## Step 3 — Verify divisibility constraints

RLinf requires:

```
(total_num_envs × max_steps_per_rollout_epoch) % global_batch_size == 0
global_batch_size % (micro_batch_size × world_size) == 0
```

For a single A100L (world_size=1), common valid combinations:

| total_num_envs | max_steps | rollout_size | global_batch_size (GRPO) | global_batch_size (PPO) |
|---|---|---|---|---|
| 16 | 320 | 5120 | 128 | 160 |
| 16 | 300 | 4800 | 160 | 160 |
| 16 | 256 | 4096 | 128 | 128 |

---

## Step 4 — Create the GRPO training config

Create `examples/embodiment/config/maniskill_grpo_openvla_<task>.yaml`:

```yaml
defaults:
  - env/maniskill_<task>@env.train
  - env/maniskill_<task>@env.eval
  - model/openvla@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all

runner:
  task_type: embodied
  logger:
    log_path: "./logs"
    project_name: rlinf
    experiment_name: "maniskill_grpo_openvla_<task>"
    logger_backends: ["tensorboard"]
  max_epochs: 250
  save_interval: 5
  resume_dir: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  group_size: 8
  rollout_epoch: 1
  eval_rollout_epoch: 1
  reward_type: action_level
  logprob_type: token_level
  entropy_type: token_level
  adv_type: grpo
  loss_type: actor
  loss_agg_func: "token-mean"
  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.28
  clip_ratio_low: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  sampling_params:
    do_sample: True
    temperature_train: 1.0
    temperature_eval: 0.6
    top_k: 50
    top_p: 1.0
  length_params:
    max_new_token: 7
    max_length: 1024
    min_length: 1

env:
  group_name: "EnvGroup"
  train:
    total_num_envs: 16
    group_size: ${algorithm.group_size}
    max_episode_steps: 320
    max_steps_per_rollout_epoch: 320
  eval:
    total_num_envs: 16
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 320
    max_steps_per_rollout_epoch: 320
    group_size: 1
    video_cfg:
      save_video: True
      video_base_dir: ${runner.logger.log_path}/video/eval

rollout:
  group_name: "RolloutGroup"
  generation_backend: "huggingface"
  enable_offload: True
  pipeline_stage_num: 1
  model:
    model_path: "/path/to/RLinf/openvla-7b-rlvla-warmup"
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 4
  global_batch_size: 128
  seed: 1234
  enable_offload: True
  model:
    model_path: "/path/to/RLinf/openvla-7b-rlvla-warmup"
    max_prompt_length: 30
  optim:
    lr: 1.0e-5
    clip_grad: 1.0
  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: True

reward:
  use_reward_model: False
critic:
  use_critic_model: False
```

---

## Step 5 — Create the PPO training config

Create `examples/embodiment/config/maniskill_ppo_openvla_<task>.yaml` — same structure as GRPO but with these differences:

```yaml
algorithm:
  adv_type: gae
  loss_type: actor_critic
  group_size: 1
  bootstrap_type: always
  logprob_type: action_level
  entropy_type: action_level
  reward_type: action_level
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2

env:
  train:
    max_episode_steps: 300
    max_steps_per_rollout_epoch: 300

actor:
  global_batch_size: 160      # (16 × 300) % 160 = 0 ✓
  model:
    add_value_head: True
    is_lora: True
  optim:
    lr: 1.0e-5                # ⚠️ never use lr > 1e-5 for PPO — causes divergence
    clip_grad: 10.0
    critic_warmup_steps: 0
```

---

## Step 6 — Create the job script

Copy an existing job script and adapt:

```bash
cp job-maniskill-stackcube-grpo.sh job-maniskill-<task>-grpo.sh
```

Key sections to update:

```bash
# Vulkan (required for all ManiSkill tasks)
VULKAN_DIR=~/projects/libero_rl/vulkan
if [ ! -f "$VULKAN_DIR/usr/lib/x86_64-linux-gnu/libvulkan.so.1" ]; then
    mkdir -p $VULKAN_DIR && cd $VULKAN_DIR
    apt-get download libvulkan1
    dpkg -x libvulkan1_*.deb $VULKAN_DIR
    cd ~/projects/libero_rl/RLinf
fi
export LD_LIBRARY_PATH=$VULKAN_DIR/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Always before srun
unset RAY_ADDRESS
rm -rf /tmp/ray/session_* 2>/dev/null || true

exec srun python examples/embodiment/train_embodied_agent.py \
    --config-name maniskill_grpo_openvla_<task>
```

---

## Reward modes

The env yaml supports three `reward_mode` values:

| `reward_mode` | Description | When to use |
|---|---|---|
| `raw` | Native ManiSkill dense reward (distance-based) | Default — works for all tasks |
| `only_success` | Sparse reward: 0 or 1 | When dense reward is noisy or misleading |
| default (omit) | RLinf custom reward using `is_src_obj_grasped` keys | Only for RLinf custom tasks (PutOnPlate etc.) |

For ManiSkill native tasks, always use `reward_mode: raw` — the native reward is already dense and well-calibrated.

---

## Checklist before submitting

- [ ] Task camera is `base_camera` (check with the inspection script in Step 1)
- [ ] `task_description` is set in `init_params` if task has no `get_language_instruction()`
- [ ] `obs_mode: rgb` (not `rgb+segmentation`)
- [ ] `wrap_obs_mode: simple`
- [ ] `enable_offload: False` in env yaml
- [ ] `(total_num_envs × max_steps) % global_batch_size == 0`
- [ ] `global_batch_size % 4 == 0` (micro_batch_size=4)
- [ ] `lr: 1.0e-5` for PPO
- [ ] Vulkan env vars set in job script
- [ ] `unset RAY_ADDRESS` in job script
- [ ] Node exclusion list in `--exclude`