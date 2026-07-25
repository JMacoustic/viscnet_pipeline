# ViscNet — Minimal Inference

Standalone, minimal-dependency inference for **ViscNet**, a computer-vision
model that estimates the **dynamic viscosity (cP)** of a stirred fluid from a
short video of its free surface. This repo bundles the trained weights, a
small public sample set, and just enough code to reproduce per-clip predictions.

Two model variants are provided:

| Variant | File | Output | Test cP MAE\* |
|---|---|---|---|
| **cP regression** (`cp`) | `weights/cp_regression_seed1206.pth` | a single cP estimate | ~4.06 cP |
| **GMM uncertainty** (`gmm`, K=5) | `weights/gmm_k5_seed1206.pth` | cP estimate **+ uncertainty** | ~2.39 cP |

\* Video-level MAE on the 1000-clip held-out test split of the seed-1206 partition.
Each checkpoint is the **best-performing weight across the three training seeds
(1205 / 1206 / 1207)** for its variant.

## How it works

ViscNet uses a **ViViT** (video vision transformer, ~9.2M params) encoder trained
in two stages:

1. **Dimensionless transfer pretraining** — the encoder first regresses four
   dimensionless groups (Reynolds, Capillary, Weber, Froude) from the **real**
   videos, giving it a physics-structured prior on the flow regime.
2. **cP regression fine-tuning** — the dimensionless head is replaced by a
   viscosity head and the whole network is fine-tuned end-to-end to predict
   `standardized log10(cP)`. The `gmm` variant instead attaches a K=5 Gaussian
   mixture head trained with a β-NLL objective, so it also reports uncertainty.

**Five-window evaluation.** Each 60-frame clip is cut into five 30-frame windows
at frame offsets **{5, 10, 15, 20, 25}** (the window slides by 5). Every window
is run independently; the five standardized predictions are **averaged in
standardized space** and then mapped back to cP:

```
cP = 10 ** (mean_over_windows(model_output) * std + mean)
```

with `mean`, `std` read from `weights/standardizer_seed1206.json`.

## Install

```bash
pip install -r requirements.txt
```

Dependencies are intentionally small: `torch`, `transformers` (only the ViViT
transformer layers are used), `numpy`, and `opencv-python-headless` for mp4
decoding. A GPU is optional — CPU inference of the 50 bundled clips takes a few
minutes. Pinned to `torch==2.5.1` / `transformers==4.39.0` (the training env);
newer versions generally work but are untested.

## Run

```bash
python infer.py --model cp            # cP point estimate
python infer.py --model gmm           # cP + uncertainty (K=5 mixture)
python infer.py --model cp --out preds.csv   # also dump a CSV
python infer.py --model gmm --device cpu     # force CPU
```

Output is a per-clip table (true vs predicted cP, absolute percentage error) and
a summary MAE / MAPE over the sample set. Point your own clips at it with
`--data <folder-of-mp4>` (60-frame clips) and a matching `--labels <json>`.

## Sample data

`data/clips/` holds **50 real test clips** — the 10 viscosity classes × 5 clips
each, with varied stirring RPM and background pattern. `data/labels.json` carries
the ground-truth `cP` and metadata (RPM, pattern, density, surface tension,
kinematic viscosity, Galilei number) per clip.

## Layout

```
viscnet-inference/
├── infer.py                     # CLI entry point
├── requirements.txt
├── viscnet_infer/
│   ├── vivit_embed.py           # VivitEmbed encoder + regression/GMM head
│   ├── vivit/                   # ViViT transformer layers (HF fork)
│   ├── dataset.py               # 5-window decode + preprocessing
│   ├── standardizer.py          # log10(cP) (un)standardization
│   ├── gmm.py                   # mixture -> (mean, std)
│   └── predict.py               # build model, load ckpt, 5-window predict
├── weights/                     # 2 checkpoints + standardizer
├── data/                        # 50 sample clips + labels.json
└── scripts/stage_from_pod.py    # (provenance) how weights/data were assembled
```

## Notes on provenance

- The bundled weights use **dimensionless-transfer pretraining on real videos**
  (not synthetic pretraining). Synthetic pretraining exists in the wider project
  only as a data-efficiency ablation and is **not** the basis of these weights.
- Preprocessing is a plain rescale `(x / 127.5) - 1` to `[-1, 1]` — **no**
  ImageNet normalization. Inputs are RGB, 224×224, 30 frames per window.

## Citation

If you use this model, please cite the ViscNet paper (see the main project /
manuscript). License: TBD by the authors.
