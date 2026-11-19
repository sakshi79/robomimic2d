# RoboMimic2D

A lightweight 2D simulation benchmark for robot manipulation research, built on [Gymnasium](https://gymnasium.farama.org/).

RoboMimic2D provides fast-iteration 2D environments that mirror the task structure of [RoboMimic](https://robomimic.github.io/) — without the overhead of full 3D physics simulators. Built with **Pymunk** for rigid-body physics and **Pygame** for rendering, it enables rapid prototyping and evaluation of manipulation policy architectures (imitation learning, diffusion policies, reinforcement learning) in seconds rather than hours.

> **Status:** Active development. Policy learning baselines and pip-installable package coming soon.

<p align="center">
  <img src="assets/lift2d_demo.gif" width="400" alt="Lift2D environment demo"/>
</p>

---

## Why 2D?

Training and evaluating manipulation policies in 3D simulators (MuJoCo, Isaac Sim) is slow — environment resets, rendering, and physics all add up. RoboMimic2D strips the problem to its core: **can your policy architecture learn contact-rich manipulation from demonstrations?**

This makes it ideal for:
- **Rapid architecture search** — test a new visual encoder or policy head in minutes, not hours
- **Debugging policy learning** — 2D state is easy to visualize end-to-end
- **Ablation studies** — isolate whether failures stem from perception, planning, or control
- **Benchmarking generalization** — lightweight domain randomization over object size, mass, and friction

## Environments

| Environment | Task | Observation Space | Action Space |
|---|---|---|---|
| **Lift2D** | Reach, grasp, and lift a block above a height threshold | `image` (3×96×96 float), `agent_pos` (2,) | Target x, target y, grip ∈ [-1, 1] |

*More environments (Push2D, Stack2D, PickPlace2D) in development.*

### Lift2D Details

The reward structure matches [RoboMimic's Lift task](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html#lift): reward = 1.0 when the block is lifted above threshold, 0.0 otherwise.

**Physics:**
- Rigid-body dynamics via [Pymunk](http://www.pymunk.org/) (Chipmunk2D) with tuned solver iterations for stiff contact
- Parallel-jaw gripper modeled with kinematic fingers, velocity-motor drive, and force limits
- **Grip force is computed as:** `F = safety_factor × m×g / (2μ)`, where μ = finger_friction × block_friction (Pymunk's product rule). The gripper can only hold the block if friction forces balance gravity
- Grasp state machine: both fingers must stall against the block with grip closed → `grasped=True`. Grasp breaks on release
- Coefficients, solver parameters and constraints tuned for realistic contact behavior

All physics parameters are exposed via `Lift2DConfig`. This enables user to modify gravity, friction, block mass, gripper geometry, control frequency, etc. as per your use case.

## Quick Start

```bash
git clone https://github.com/sakshi79/robomimic2d.git
cd robomimic2d
pip install -r requirements.txt

# Interactive teleoperation
python -m envs2d.lift2d
```

**Controls:**
| Input | Action |
|---|---|
| Left-click drag | Move gripper toward cursor |
| Right-click / Space | Toggle gripper open/closed |
| Arrow keys | Move gripper (additive) |
| `S` / Enter | End episode (saved when recording) and start the next |
| `R` | Discard current episode and restart |
| `Q` / Escape | Quit (current episode is not saved) |

### Record Demonstrations

```bash
python -m envs2d.lift2d -o data/lift.zarr
```

An episode ends automatically once the block has stayed lifted for `success_hold_steps` consecutive control steps (default 30, i.e. 3 s at 10 Hz), or when you press `S` / Enter. Ended episodes are appended to the zarr replay buffer; `R` discards the current episode and `Q` quits without saving it. Recording resumes from the existing episode count, and episode *i* is seeded with `i` so its initial layout is reproducible. Each step stores `img` (render, without the HUD), `state` (gripper xy + block x, y, θ), `gripper` (finger gap) and `action` (target x, target y, grip). Options: `--render-size` (default 96), `--window` (default 512).


### Gymnasium API

```python
import gymnasium as gym
import envs2d    # registers all env

env = gym.make("Lift2D-v0")   # or gym.make("Lift2D-v0", render_size=84, block_mass=2.0)
obs, info = env.reset(seed=0) # obs['image']: (3,96,96), obs['agent_pos']: (2,)

for step in range(1000):
    action = policy(obs)      # [target_x, target_y, grip]
    obs, reward, terminated, truncated, info = env.step(action)
    # info: pos_agent, vel_agent, block_pose, grasped, grip, gripper_gap, n_contacts
    if terminated or truncated:
        obs, info = env.reset()
```
`terminated` is True once the block has been held lifted for `success_hold_steps`; `truncated` is True after `max_episode_steps`. Actions are in pixel coordinates; for RL, wrap with `gymnasium.wrappers.RescaleAction(env, -1.0, 1.0)`.

### Replay Buffer

Demonstrations are stored in a zarr-backed `ReplayBuffer` supporting episodic storage, chunked compression, and efficient random access — compatible with standard imitation learning data pipelines (e.g., Diffusion Policy).

```python
from envs2d.replay_buffer import ReplayBuffer

buffer = ReplayBuffer.create_from_path("data/lift.zarr", mode="r")
print(f"{buffer.n_episodes} episodes, {buffer.n_steps} total steps")
episode = buffer.get_episode(0)  # dict of numpy arrays
```

## Policy Benchmarking

Policies are trained with [robomimic](https://robomimic.github.io/) v0.5.0 and evaluated in Lift2D on held-out seeds.

| Policy | Low-dim | Image |
|---|---|---|
| BC (MLP) | ✓ | ✓ |
| BC-RNN (GMM) | ✓ | ✓ |
| BC-Transformer (GMM) | ✓ | ✓ |
| Diffusion Policy (U-Net) | ✓ | ✓ |

**Observations:** low-dim uses `agent_pos` (2), `block_pose` (3) and `gripper` (1, finger gap). Image uses `image` (96×96 RGB, random-cropped to 84×84, ResNet-18 + spatial softmax), `agent_pos` and `gripper`.
**Actions:** pixel-space targets, min-max normalized to [-1, 1] by robomimic and unnormalized at rollout time.

### Setup

```bash
pip install "robomimic @ git+https://github.com/ARISE-Initiative/robomimic@v0.5.0" h5py
```

### Pipeline

```bash
# 1. Convert demos to robomimic hdf5 and split train/valid
python -m scripts.zarr_to_robomimic -i data/lift.zarr -o data/lift2d.hdf5
python -m robomimic.scripts.split_train_val --dataset data/lift2d.hdf5 --ratio 0.1

# 2. Generate configs (4 policies × {low_dim, image})
python -m scripts.make_configs --dataset data/lift2d.hdf5 --out configs/lift2d --output-dir runs/lift2d

# 3. Train
python -m robomimic.scripts.train --config configs/lift2d/diffusion_policy_image.json

# 4. Evaluate every checkpoint on 50 held-out seeds and print a summary table
SDL_VIDEODRIVER=dummy python -m scripts.eval_policy --run runs/lift2d --summary --episodes 50
```

Training runs `--num-epochs` × 100 gradient steps (default 300 → 30k steps). robomimic's built-in rollouts are disabled, so `scripts/eval_policy.py` evaluates checkpoints in the Gymnasium env instead.

**Evaluation protocol:** evaluation seeds start at 100000, so they never overlap the demo layouts (seeds 0…N−1). An episode succeeds when `terminated` is True, meaning the block was held lifted for `success_hold_steps`; it fails when truncated at `max_episode_steps`. We report the best success rate across checkpoints and the success rate at the final checkpoint. Per-checkpoint results are saved to `eval.json` in each run directory. Add `--render` to watch rollouts.

### Results

*Coming soon.*


## Architecture

```
envs2d/
├── __init__.py
├── lift2d/
│   ├── env.py          # Lift2DEnv (gymnasium.Env) — step, reset, obs, reward
│   ├── physics.py      # Pymunk space, walls, block, contact counting
│   ├── gripper.py      # Parallel-jaw gripper: kinematic base, finger motor, grasp state machine
│   ├── config.py       # Lift2DConfig — all tunable physics/rendering/control params
│   ├── render.py       # Pygame renderer (human + rgb_array modes)
│   ├── teleop.py       # Mouse/keyboard teleoperation controller
│   └── demo.py         # CLI entry point for interactive + recording
├── replay_buffer.py    # Zarr-backed episodic replay buffer with chunking/compression
└── pymunk_override.py  # Pymunk rendering utilities
scripts/
├── zarr_to_robomimic.py  # zarr replay buffer → robomimic hdf5
├── make_configs.py       # robomimic configs for each policy × observation type
└── eval_policy.py        # Gymnasium rollouts on held-out seeds, success-rate table
```

## Roadmap

- [x] Lift2D environment with Pymunk rigid-body physics
- [x] Physics-based grip force computation (no cheating)
- [x] Gripper state machine with contact-based grasp detection
- [x] Interactive teleoperation with mouse/keyboard
- [x] Zarr replay buffer for demonstration recording
- [x] Gymnasium-compatible API
- [x] Policy benchmark pipeline via robomimic (BC, BC-RNN, BC-Transformer, Diffusion Policy)
- [ ] Benchmark results
- [ ] Push2D, Stack2D, PickPlace2D environments
- [ ] pip-installable package (`pip install robomimic2d`)
- [ ] Domain randomization (object size, mass, friction)
- [x] Gymnasium `register()` integration

## Tech Stack

**Physics:** Pymunk 6.2.1 (Chipmunk2D)  
**Rendering:** Pygame  
**Data:** NumPy, zarr (chunked, compressed episodic replay)  
**API:** Gymnasium 1.3.0  
**Policy learning:** robomimic 0.5.0 (BC, BC-RNN, BC-Transformer, Diffusion Policy), PyTorch


## Related Work

This project is part of ongoing research on visual generalization for manipulation policies in the [Helping Hands Lab](https://www2.ccs.neu.edu/research/helpinghands/) at Northeastern University, advised by Dr. Robert Platt.

- [RoboMimic](https://robomimic.github.io/) — the 3D benchmark this project mirrors
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) — a key policy architecture we evaluate on these environments

## License

MIT

## Citation

```bibtex
@software{bhatia2025robomimic2d,
  author = {Bhatia, Sakshi},
  title = {RoboMimic2D: Lightweight 2D Manipulation Benchmarks for Policy Learning},
  year = {2026},
  url = {https://github.com/sakshi79/robomimic2d}
}
```