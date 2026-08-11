2D simulation environments for robot learning in OpenAI Gym style.

Run "python -m envs2d.lift2d" to play the simulation env.

![Demo](assets/lift2d.gif)

## Demo + recording

```
python -m envs2d.lift2d                    # interactive, no recording
python -m envs2d.lift2d -o data/lift.zarr   # interactive + record episodes
```

Controls:
- Left-click drag: move gripper toward cursor
- Right-click: toggle gripper open / closed
- Arrow keys (additive): also move gripper
- SPACE (toggle): also open / close gripper
- `R`: retry episode
- `Q` / Escape: quit

Each completed episode (not retried, with at least one step) is appended to the zarr
replay buffer at the `-o`/`--output` path. Other options: `--render-size` (default 96),
`--window` (default 512).

Coming soon..
1. More environments
2. Pip installable package