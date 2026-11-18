"""Evaluate robomimic checkpoints on Lift2D with held-out seeds.

Usage:
    python -m scripts.eval_policy --run runs/lift2d/lift2d_diffusion_policy_image/<timestamp>
    python -m scripts.eval_policy --run runs/lift2d --summary        # every run under runs/lift2d
"""
import glob
import json
import os
import re
from collections import deque

import click
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
from envs2d.lift2d.env import Lift2DEnv

EVAL_SEED_OFFSET = 100_000   # demos use seeds 0..N-1; never evaluate on those layouts


def make_obs(obs, info, keys):
    """robomimic raw obs dict (same keys/format as the converted hdf5), limited to the policy's keys."""
    full = {
        "agent_pos": info["pos_agent"].astype(np.float32),
        "block_pose": info["block_pose"].astype(np.float32),
        "gripper": np.float32([info["gripper_gap"]]),
        "image": np.round(np.moveaxis(obs["image"], 0, -1) * 255).astype(np.uint8),  # HWC uint8
    }
    return {k: full[k] for k in keys}


def rollout(policy, env, seed, frame_stack, obs_keys, render):
    obs, info = env.reset(seed=seed)
    policy.start_episode()
    history = deque([make_obs(obs, info, obs_keys)] * frame_stack, maxlen=frame_stack)
    while True:
        ob = history[0] if frame_stack == 1 else {k: np.stack([h[k] for h in history]) for k in history[0]}
        act = policy(ob=ob)
        obs, reward, terminated, truncated, info = env.step(act)
        if render:
            env.render_frame("human")
        history.append(make_obs(obs, info, obs_keys))
        if terminated or truncated:
            return bool(terminated), env.unwrapped._steps


def eval_checkpoint(ckpt, n_episodes, render, device):
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=device, verbose=False)
    frame_stack = json.loads(ckpt_dict["config"])["train"]["frame_stack"]
    obs_keys = ckpt_dict["shape_metadata"]["all_obs_keys"]
    env = Lift2DEnv()
    try:
        results = [rollout(policy, env, EVAL_SEED_OFFSET + i, frame_stack, obs_keys, render) for i in range(n_episodes)]
    finally:
        env.close()
    successes = [s for s, _ in results]
    return {"success_rate": float(np.mean(successes)),
            "mean_steps": float(np.mean([t for _, t in results]))}


def epoch_of(path):
    m = re.search(r"epoch_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def eval_run(run_dir, n_episodes, render, device):
    ckpts = sorted(glob.glob(os.path.join(run_dir, "models", "*.pth")), key=epoch_of)
    if not ckpts:
        ckpts = [os.path.join(run_dir, "last.pth")]
    per_ckpt = {}
    for ckpt in ckpts:
        per_ckpt[os.path.basename(ckpt)] = r = eval_checkpoint(ckpt, n_episodes, render, device)
        print(f"  {os.path.basename(ckpt):28s} success={r['success_rate']:.2f}  steps={r['mean_steps']:.0f}")
    rates = [r["success_rate"] for r in per_ckpt.values()]
    summary = {"run": run_dir, "n_episodes": n_episodes, "max": max(rates), "last": rates[-1], "per_ckpt": per_ckpt}
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


@click.command()
@click.option("--run", "run_path", required=True, help="a run dir (contains models/), or a parent dir with --summary")
@click.option("--episodes", "n_episodes", default=50, type=int, show_default=True)
@click.option("--summary", is_flag=True, help="evaluate every run under --run and print a table")
@click.option("--render", is_flag=True, help="show rollouts in a window")
def main(run_path, n_episodes, summary, render):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not summary:
        eval_run(run_path, n_episodes, render, device)
        return
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(run_path, "*", "*"))):
        if not os.path.isdir(os.path.join(run_dir, "models")):
            continue
        print(run_dir)
        s = eval_run(run_dir, n_episodes, render, device)
        rows.append((os.path.basename(os.path.dirname(run_dir)), s["max"], s["last"]))
    print(f"\n{'experiment':40s} {'max':>6s} {'last':>6s}")
    for name, best, last in rows:
        print(f"{name:40s} {best:6.2f} {last:6.2f}")


if __name__ == "__main__":
    main()
