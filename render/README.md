# render

Synthetic broadcast video with exact labels, for training the detector and for measuring the
pipeline's error (spec section 6.10).

This is its own environment, not a member of the main uv workspace: it imports Blender as the
`bpy` module, and each bpy release pins one Python version (3.11 for bpy 4.5 LTS, 3.13 for
bpy 5.2). It reads `scene.yaml` and episode Parquet files and writes frames, labels and an
episode. It talks to the main environment through files only, and runs from
`docker/render.Dockerfile`.

Because it imports `bpy`, this folder is planned to be GPL-3.0 with its own `LICENSE` file,
separate from the Apache-2.0 packages (spec section 10).
