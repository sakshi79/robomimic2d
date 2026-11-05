"""RoboMimic2D environments. Importing this package registers all envs with Gymnasium."""
import gymnasium as gym

gym.register(id="Lift2D-v0", entry_point="envs2d.lift2d.env:Lift2DEnv")