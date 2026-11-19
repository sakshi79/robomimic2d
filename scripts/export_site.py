"""Export benchmark results and assets for the RoboMimic2D project site.

Usage:
    python -m scripts.export_site --runs runs/lift2d --dataset data/lift2d.hdf5 --site ../robomimic2d-site
    python -m scripts.export_site ... --rollouts      # also record policy rollout videos (needs GPU)
"""
import glob
import json
import os
import re
from collections import deque

import click
import h5py
import numpy as np
from PIL import Image

ALGO_NAMES = {
    "diffusion_policy": "Diffusion Policy (U-Net)",
    "bc_rnn": "BC-RNN (GMM)",
    "bc_transformer": "BC-Transformer (GMM)",
    "bc": "BC (MLP)",
}


def epoch_of(name):
    m = re.search(r"epoch_(\d+)", name)
    return int(m.group(1)) if m else -1


def parse_experiment(name):
    """lift2d_<algo>_<modality> -> (algo, modality)"""
    body = name.removeprefix("lift2d_")
    for modality in ("low_dim", "image"):
        if body.endswith("_" + modality):
            return body[: -len(modality) - 1], modality
    return body, "unknown"


def export_results(runs_root, max_epoch=None):
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_root, "*", "*", "eval.json"))):
        run_dir = os.path.dirname(path)
        experiment = os.path.basename(os.path.dirname(run_dir))
        algo, modality = parse_experiment(experiment)
        ev = json.load(open(path))
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        ckpts = sorted(ev["per_ckpt"].items(), key=lambda kv: epoch_of(kv[0]))
        if max_epoch is not None:
            ckpts = [kv for kv in ckpts if epoch_of(kv[0]) <= max_epoch]
        runs.append({
            "experiment": experiment,
            "algo": algo,
            "name": ALGO_NAMES.get(algo, algo),
            "modality": modality,
            "timestamp": os.path.basename(run_dir),
            "train_seed": cfg["train"]["seed"],
            "num_epochs": cfg["train"]["num_epochs"] if max_epoch is None else min(max_epoch, cfg["train"]["num_epochs"]),
            "steps_per_epoch": cfg["experiment"]["epoch_every_n_steps"],
            "n_episodes": ev["n_episodes"],
            "epochs": [epoch_of(k) for k, _ in ckpts],
            "success": [v["success_rate"] for _, v in ckpts],
            "mean_steps": [v["mean_steps"] for _, v in ckpts],
        })
    return runs


def export_dataset(dataset, assets_dir):
    with h5py.File(dataset, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        lengths = [int(f["data"][k].attrs["num_samples"]) for k in demos]
        masks = {k: len(v) for k, v in f["mask"].items()} if "mask" in f else {}

        # observation strip from the first demo: start, approach, grasp, lift
        g = f["data"][demos[0]]
        imgs, act = g["obs/image"][:], g["actions"][:]
        close = int(np.argmax(act[:, 2] > 0)) if (act[:, 2] > 0).any() else len(act) // 2
        picks = {"start": 0, "approach": close // 2, "grasp": close, "lift": len(act) - 1}
        frames = []
        for label, t in picks.items():
            fname = f"obs_{label}.png"
            Image.fromarray(imgs[t]).save(os.path.join(assets_dir, fname))
            frames.append({"label": label, "step": t, "src": f"assets/{fname}"})

    return {
        "path": os.path.basename(dataset),
        "n_demos": len(demos),
        "n_steps": int(sum(lengths)),
        "mean_length": float(np.mean(lengths)),
        "min_length": int(min(lengths)),
        "max_length": int(max(lengths)),
        "splits": masks,
        "obs_frames": frames,
    }


def record_rollout(ckpt, seed, out_path, fps=10):
    """Roll out one checkpoint on one eval seed and save a full-resolution mp4."""
    import imageio
    import torch
    import robomimic.utils.file_utils as FileUtils
    from envs2d.lift2d.env import Lift2DEnv
    from scripts.eval_policy import make_obs

    policy, cd = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=torch.device("cuda"), verbose=False)
    frame_stack = json.loads(cd["config"])["train"]["frame_stack"]
    keys = cd["shape_metadata"]["all_obs_keys"]
    env = Lift2DEnv()
    obs_size = env.cfg.render_size

    def full_frame():
        env.cfg.render_size = env.cfg.window_size      # render at window resolution for the video only
        img = env.render_frame("rgb_array")
        env.cfg.render_size = obs_size
        return img

    torch.manual_seed(seed)                            # GMM / diffusion sampling is stochastic; fix it per video
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    policy.start_episode()
    hist = deque([make_obs(obs, info, keys)] * frame_stack, maxlen=frame_stack)
    frames = [full_frame()]
    while True:
        ob = hist[0] if frame_stack == 1 else {k: np.stack([h[k] for h in hist]) for k in hist[0]}
        obs, _, terminated, truncated, info = env.step(policy(ob=ob))
        hist.append(make_obs(obs, info, keys))
        frames.append(full_frame())
        if terminated or truncated:
            break
    steps = env._steps
    env.close()
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=16,
                     ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    return {"success": bool(terminated), "steps": int(steps)}


@click.command()
@click.option("--runs", "runs_root", default="runs/lift2d", show_default=True)
@click.option("--dataset", default="data/lift2d.hdf5", show_default=True)
@click.option("--site", "site_dir", required=True, help="site repo root (writes data/ and assets/)")
@click.option("--rollouts", is_flag=True, help="record rollout videos for each image run's last checkpoint")
@click.option("--seed", "seeds", default=[100_000], type=int, multiple=True, show_default=True,
              help="eval seed(s) for rollout videos; repeat the flag for several")
@click.option("--max-epoch", default=None, type=int, help="only report checkpoints up to this epoch")
def main(runs_root, dataset, site_dir, rollouts, seeds, max_epoch):
    data_dir, assets_dir = os.path.join(site_dir, "data"), os.path.join(site_dir, "assets")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    for old in glob.glob(os.path.join(assets_dir, "rollout_*.mp4")):
        os.remove(old)                                 # drop videos from earlier exports

    results = {"runs": export_results(runs_root, max_epoch), "dataset": export_dataset(dataset, assets_dir),
               "max_epoch": max_epoch, "rollouts": []}

    if rollouts:
        for run in results["runs"]:
            run_dir = os.path.join(runs_root, run["experiment"], run["timestamp"])
            ckpt = os.path.join(run_dir, "models", f"model_epoch_{run['epochs'][-1]}.pth")
            for seed in seeds:
                fname = f"rollout_{run['experiment']}_seed{seed}.mp4"
                r = record_rollout(ckpt, seed, os.path.join(assets_dir, fname))
                results["rollouts"].append({"experiment": run["experiment"], "name": run["name"], "modality": run["modality"],
                                            "epoch": run["epochs"][-1], "seed": seed, "src": f"assets/{fname}", **r})
                print(f"{fname}: success={r['success']} steps={r['steps']}")

    with open(os.path.join(data_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {data_dir}/results.json ({len(results['runs'])} runs)")


if __name__ == "__main__":
    main()
