# ViscNet Pipeline

End-to-end resources for **ViscNet**, a computer-vision model that estimates the
**dynamic viscosity (cP)** of a stirred fluid from a short video of its free
surface. The repository is split into two self-contained sections:

| Section | Path | What it does |
|---|---|---|
| 🧪 **Data generation** | [`data_generation/`](data_generation/) | Produces the synthetic training videos: SPlisHSPlasH fluid simulation → Splashsurf surface reconstruction → Blender rendering. |
| 🔮 **Inference** | [`inference/`](inference/) | Runs the trained ViViT estimator on videos to predict cP (with an optional uncertainty variant). Ships weights + 50 sample clips. |

The two sections are independent — you can run inference without the data
pipeline, and vice-versa. Each folder has its own README and requirements.

---

## Quick start — inference

Predict viscosity on the 50 bundled sample clips:

```bash
cd inference
pip install -r requirements.txt
python infer.py --model cp      # cP point estimate
python infer.py --model gmm     # cP + uncertainty (K=5 mixture)
```

Two trained checkpoints are included (each the best across three training seeds):

| Variant | Output | Test cP MAE |
|---|---|---|
| `cp` — cP regression | a single cP estimate | ~4.06 cP |
| `gmm` — GMM (K=5) | cP estimate **+ uncertainty** | ~2.39 cP |

See [`inference/README.md`](inference/README.md) for the model description, the
five-window evaluation scheme, and how to run on your own clips.

## Quick start — data generation

Generate one synthetic clip (one fluid property × one impeller speed):

```bash
cd data_generation
python splishsplash/run_simulation.py --property-index 1 --rpm 270 --output outputs/sim_0101
python splashsurf/reconstruct.py     --input outputs/sim_0101/vtk --output outputs/sim_0101/mesh
python blender/render.py             --mesh  outputs/sim_0101/mesh --output outputs/sim_0101/render
```

Requires SPlisHSPlasH 2.14.0, Splashsurf, and Blender 4.4.3. The full grid (50
fluid properties × 10 speeds, ×10 backgrounds ×10 lighting = 50,000 clips) and
all parameters are documented in [`data_generation/README.md`](data_generation/README.md).

---

## How ViscNet works (short version)

A **ViViT** video transformer (~9.2M params) is trained in two stages: it first
regresses four dimensionless flow groups (Reynolds, Capillary, Weber, Froude) to
build a physics-structured prior, then a viscosity head is fine-tuned end-to-end
to predict `log10(cP)`. Each 60-frame clip is scored with five 30-frame windows
(offsets {5,10,15,20,25}) whose predictions are averaged. The `gmm` variant adds
a Gaussian-mixture head for calibrated uncertainty. Full details and provenance
are in the [inference README](inference/README.md).

## Citation

If you use ViscNet, please cite the ViscNet paper. License: TBD by the authors.
