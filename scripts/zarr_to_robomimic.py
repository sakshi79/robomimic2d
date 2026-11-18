"""Convert a Lift2D zarr replay buffer into a robomimic-format hdf5 dataset.

Usage:
    python -m scripts.zarr_to_robomimic -i data/lift2.zarr -o data/lift2d.hdf5
    python -m robomimic.scripts.split_train_val --dataset data/lift2d.hdf5 --ratio 0.1
"""
import json
import click
import h5py
import numpy as np

from envs2d.replay_buffer import ReplayBuffer
from envs2d.lift2d.config import Lift2DConfig


@click.command()
@click.option("-i", "--input", "zarr_path", required=True)
@click.option("-o", "--output", "hdf5_path", required=True)
def main(zarr_path, hdf5_path):
    cfg = Lift2DConfig()
    buf = ReplayBuffer.create_from_path(zarr_path, mode="r")
    floor_rest_y = cfg.window_size - cfg.wall_thickness - cfg.block_size / 2

    with h5py.File(hdf5_path, "w") as f:
        data = f.create_group("data")
        total = 0
        for i in range(buf.n_episodes):
            ep = buf.get_episode(i)
            T = len(ep["action"])
            state = ep["state"].astype(np.float32)

            obs = {
                "agent_pos": state[:, 0:2],
                "block_pose": state[:, 2:5],
                "gripper": ep["gripper"].astype(np.float32),
                "image": ep["img"],                       # (T, H, W, 3) uint8, robomimic converts to CHW
            }
            # next_obs[t] = obs[t+1]; last step repeats
            next_obs = {k: np.concatenate([v[1:], v[-1:]]) for k, v in obs.items()}

            # reward for transition t is whether the block is lifted after the action
            lifted_next = next_obs["block_pose"][:, 1] < floor_rest_y - cfg.lift_threshold
            dones = np.zeros(T, dtype=np.int64)
            dones[-1] = 1

            g = data.create_group(f"demo_{i}")
            g.attrs["num_samples"] = T
            g.create_dataset("actions", data=ep["action"].astype(np.float32))
            g.create_dataset("rewards", data=lifted_next.astype(np.float32))
            g.create_dataset("dones", data=dones)
            for name, group in (("obs", obs), ("next_obs", next_obs)):
                for k, v in group.items():
                    g.create_dataset(f"{name}/{k}", data=v,
                                     compression="gzip" if k == "image" else None)
            total += T

        data.attrs["total"] = total
        data.attrs["env_args"] = json.dumps({
            "env_name": "Lift2D-v0",
            "type": 2,  # robomimic EnvType.GYM_TYPE; only read if rollouts are enabled (we disable them)
            "env_kwargs": {"render_size": cfg.render_size, "window_size": cfg.window_size},
        })
    print(f"Wrote {buf.n_episodes} demos ({total} steps) to {hdf5_path}")


if __name__ == "__main__":
    main()
