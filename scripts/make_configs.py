"""Generate robomimic training configs for the Lift2D benchmark.

Usage:
    python -m scripts.make_configs --dataset data/lift2d.hdf5 --out configs/lift2d
    python -m robomimic.scripts.train --config configs/lift2d/bc_rnn_image.json
"""
import os
import click
from robomimic.config import config_factory

ALGOS = ["bc", "bc_rnn", "bc_transformer", "diffusion_policy"]
MODALITIES = ["low_dim", "image"]

# robomimic paper image settings (ResNet-18 + spatial softmax); crop 96 -> 84 (~88%, paper uses 84 -> 76)
RGB_CORE_KWARGS = {
    "feature_dimension": 64,
    "backbone_class": "ResNet18Conv",
    "backbone_kwargs": {"pretrained": False, "input_coord_conv": False},
    "pool_class": "SpatialSoftmax",
    "pool_kwargs": {"num_kp": 32, "learnable_temperature": False, "temperature": 1.0, "noise_std": 0.0},
}
CROP = 84


def make_config(algo, modality, dataset, output_dir, num_epochs):
    config = config_factory("diffusion_policy" if algo == "diffusion_policy" else "bc")
    with config.values_unlocked():
        config.experiment.name = f"lift2d_{algo}_{modality}"
        config.experiment.validate = True
        config.experiment.rollout.enabled = False      # robomimic's EnvGym is old-gym; we evaluate separately
        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = 25
        config.experiment.save.on_best_rollout_success_rate = False

        config.train.data = [{"path": os.path.abspath(dataset)}]
        config.train.output_dir = os.path.abspath(output_dir)
        config.train.hdf5_filter_key = "train"
        config.train.hdf5_validation_filter_key = "valid"
        config.train.action_config = {"actions": {"normalization": "min_max"}}  # pixel actions -> [-1, 1]
        config.train.num_epochs = num_epochs           # 1 epoch = experiment.epoch_every_n_steps (100) grad steps

        obs = config.observation.modalities.obs
        if modality == "low_dim":
            obs.low_dim = ["agent_pos", "block_pose", "gripper"]
            obs.rgb = []
        else:
            obs.low_dim = ["agent_pos", "gripper"]
            obs.rgb = ["image"]
            config.observation.encoder.rgb.core_kwargs = RGB_CORE_KWARGS
            config.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"
            config.observation.encoder.rgb.obs_randomizer_kwargs = {
                "crop_height": CROP, "crop_width": CROP, "num_crops": 1, "pos_enc": False}
            config.train.num_data_workers = 2

        if algo == "bc_rnn":
            config.algo.rnn.enabled = True
            config.algo.rnn.horizon = 10
            config.algo.rnn.hidden_dim = 400 if modality == "low_dim" else 1000
            config.algo.gmm.enabled = True
            config.algo.actor_layer_dims = []
            config.train.seq_length = 10
        elif algo == "bc_transformer":
            config.algo.transformer.enabled = True
            config.algo.transformer.context_length = 10
            config.algo.gmm.enabled = True
            config.algo.actor_layer_dims = []
            config.train.frame_stack = 10
            config.train.seq_length = 1
        elif algo == "diffusion_policy":
            config.train.frame_stack = config.algo.horizon.observation_horizon   # 2
            config.train.seq_length = config.algo.horizon.prediction_horizon     # 16
    return config


@click.command()
@click.option("--dataset", required=True, help="robomimic hdf5 with mask/train and mask/valid")
@click.option("--out", "out_dir", default="configs/lift2d", show_default=True)
@click.option("--output-dir", default="runs/lift2d", show_default=True, help="where robomimic writes logs/checkpoints")
@click.option("--num-epochs", default=300, type=int, show_default=True)
def main(dataset, out_dir, output_dir, num_epochs):
    os.makedirs(out_dir, exist_ok=True)
    for algo in ALGOS:
        for modality in MODALITIES:
            path = os.path.join(out_dir, f"{algo}_{modality}.json")
            make_config(algo, modality, dataset, output_dir, num_epochs).dump(path)
            print("wrote", path)


if __name__ == "__main__":
    main()
