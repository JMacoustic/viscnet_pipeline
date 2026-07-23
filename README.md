# ViscNet Synthetic Data Generator

Minimal reference implementation of the synthetic-data pipeline used for ViscNet:

1. SPlisHSPlasH produces fluid-particle VTK files and impeller OBJ files.
2. Splashsurf reconstructs a fluid-surface OBJ sequence.
3. Blender renders the fluid and impeller sequences against the retained background patterns.

This repository contains ViscNet configuration, rig meshes, fluid properties, and thin execution scripts. It does not vendor SPlisHSPlasH, Splashsurf, Blender, or generated datasets.

## Requirements

- Linux
- SPlisHSPlasH 2.14.0 with Python bindings
- Python packages listed in `splishsplash/requirements.txt` and `splashsurf/requirements.txt`
- Blender 4.4.3
- FFmpeg, only when encoding the rendered PNG sequence

## 1. Particle simulation

Run one fluid-property row at one impeller speed:

```bash
python splishsplash/run_simulation.py \
  --property-index 1 \
  --rpm 270 \
  --output outputs/sim_0101
```

The dataset grid uses 50 property rows and 10 speeds from 270 to 450 RPM in 20 RPM steps. Each run lasts 15 s and exports at 10 fps. The motor is powered until 11.5 s, stops at 12.5 s, and then decays freely.

## 2. Surface reconstruction

```bash
python splashsurf/reconstruct.py \
  --input outputs/sim_0101/vtk \
  --output outputs/sim_0101/mesh
```

Frames 101–150 are reconstructed by default.

## 3. Blender rendering

```bash
blender -b --python blender/render.py -- \
  --fluid-dir outputs/sim_0101/mesh \
  --impeller-dir outputs/sim_0101/obj \
  --background blender/backgrounds/checkerboard/checkerboard_1900x1000_15px_00_15px_1p07mm.png \
  --output outputs/sim_0101/render \
  --device OPTIX
```

Use `--device CPU` when a supported Cycles GPU is unavailable. The script writes 50 PNG frames at 224 × 224. Encode the video with:

```bash
ffmpeg -framerate 10 -i outputs/sim_0101/render/%04d.png \
  -c:v libx264 -pix_fmt yuv420p outputs/sim_0101.mp4
```

Repeat the render for the five checkerboard and five white-noise images in `blender/backgrounds/`, and vary `--seed` from 0 to 9 for the ten lighting conditions.

## Configuration

- `splishsplash/scene.json`: DFSPH scene and solver settings.
- `splishsplash/properties.csv`: executed 50-row glycerol–water property table.
- `splashsurf/config.json`: surface-reconstruction settings.
- `blender/config.json`: camera, material, lighting, and output settings.

The included executed property table spans 0.8927–300 cP.
