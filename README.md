# ViscNet Pipeline

ViscNet estimates the **dynamic viscosity (cP)** of a stirred fluid from a short video of
its free surface — no probe in the liquid, just a camera looking at the swirl and how it
settles. This repository holds everything needed to *use* the trained model and to
*reproduce the training videos*, split into two self-contained sections:

| Section | Path | What it does |
|---|---|---|
| **Inference** | [`inference/`](inference/) | Run the trained ViViT estimator on videos to predict viscosity (with an optional uncertainty variant). Ships the weights and 100 sample clips. |
| **Data generation** | [`data_generation/`](data_generation/) | Reproduce the synthetic training videos: SPlisHSPlasH fluid simulation → Splashsurf surface reconstruction → Blender rendering. |

The two sections are independent — you can run inference without touching the data
pipeline, and vice-versa. Each folder has its own README and requirements.

---

## Quick start — inference

Run the model on the 100 bundled sample clips:

```bash
cd inference
pip install -r requirements.txt
python infer.py --model cp      # viscosity point estimate
python infer.py --model gmm     # viscosity + uncertainty (K=5 mixture)
```

It prints a per-clip table (true vs predicted cP) and a summary. Two trained checkpoints
are included:

| Variant | Output |
|---|---|
| `cp` — viscosity regression | a single cP estimate |
| `gmm` — K=5 mixture | a cP estimate **+ uncertainty** |

See [`inference/README.md`](inference/README.md) for the model description, the five-window
evaluation scheme, and how to run on your own clips.

## Quick start — data generation

Generate one synthetic clip (one fluid property × one impeller speed):

```bash
cd data_generation
python splishsplash/run_simulation.py --property-index 1 --rpm 270 --output outputs/sim_0101
python splashsurf/reconstruct.py     --input outputs/sim_0101/vtk  --output outputs/sim_0101/mesh
blender -b --python blender/render.py -- \
  --fluid-dir outputs/sim_0101/mesh --impeller-dir outputs/sim_0101/obj \
  --background blender/backgrounds/checkerboard/checkerboard_1900x1000_15px_00_15px_1p07mm.png \
  --output outputs/sim_0101/render
```

Requires SPlisHSPlasH 2.14.0, Splashsurf, and Blender 4.4.3. The full grid (50 fluid
properties × 10 speeds, × 10 backgrounds × 10 lighting seeds) and all parameters are
documented in [`data_generation/README.md`](data_generation/README.md).

---

## How ViscNet works (short version)

A **ViViT** video transformer (~9.2M params) is trained in two stages: it first regresses
four dimensionless flow groups (Reynolds, Capillary, Weber, Froude) to build a
physics-structured prior, then a viscosity head is fine-tuned end-to-end to predict
`log10(cP)`. Each 60-frame clip is scored with five 30-frame windows (offsets
{5,10,15,20,25}) whose predictions are averaged. The `gmm` variant adds a Gaussian-mixture
head for calibrated uncertainty. Full details and provenance are in the
[inference README](inference/README.md).

