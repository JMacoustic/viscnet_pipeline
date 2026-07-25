"""Build the ViViT model, load a checkpoint, and run 5-window inference.

Per-clip prediction:
  1. cut the 5 windows {5,10,15,20,25};
  2. run each through the model -> a standardized log10(cP) point per window
     (for the GMM model, the point is the mixture mean);
  3. AVERAGE the 5 standardized points, then un-standardize:
         cP = 10 ** (mean_std * std + mean);
  4. for the GMM model, also report a predictive sigma (mean per-window mixture
     std, in log10(cP) units).
"""

from __future__ import annotations

import numpy as np
import torch

from .dataset import IMAGE_SIZE, five_window_batch, normalize_clip_batch
from .gmm import gmm_moments, split_gmm_params
from .standardizer import Standardizer
from .vivit_embed import VivitEmbed


def build_model(gmm_k: int = 0) -> VivitEmbed:
    """The 9.2M no-RPM / no-classification ViViT. output_size = 1 (cP) or 3K (GMM)."""
    output_size = 1 if not gmm_k else 3 * int(gmm_k)
    return VivitEmbed(
        dropout=0.0,
        output_size=output_size,
        class_bool=False,
        visc_class=10,
        gmm_num=3,
        rpm_bool=False,
        pat_bool=False,
        num_frames=30,
        image_size=224,
        hidden_size=256,
        num_hidden_layers=10,
        num_attention_heads=8,
        intermediate_size=1024,
    )


def _load_state_dict(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device)


class ViscNetPredictor:
    def __init__(self, checkpoint: str, standardizer, gmm_k: int = 0, device=None):
        self.gmm_k = int(gmm_k or 0)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.std = (
            standardizer
            if isinstance(standardizer, Standardizer)
            else Standardizer.from_json(standardizer)
        )
        self.model = build_model(self.gmm_k).to(self.device).eval()
        self.model.load_state_dict(_load_state_dict(checkpoint, self.device), strict=True)

    @torch.no_grad()
    def predict_clip(self, mp4_path: str) -> dict:
        windows = five_window_batch(mp4_path)                       # uint8 [5,T,H,W,3]
        frames = normalize_clip_batch(windows).to(self.device)     # float [5,T,3,H,W]
        n = frames.shape[0]
        rpm = torch.zeros(n, dtype=torch.long, device=self.device)
        pattern = torch.zeros(n, IMAGE_SIZE, IMAGE_SIZE, 3, dtype=torch.float32, device=self.device)
        out = self.model(frames, rpm, pattern).detach().cpu()      # [5, output_size]

        if self.gmm_k:
            mu, sigma, pi = split_gmm_params(out, self.gmm_k)
            win_point, win_sigma = gmm_moments(mu.numpy(), sigma.numpy(), pi.numpy())  # [5],[5]
        else:
            win_point = out[:, 0].numpy().astype(np.float64)       # [5] standardized point
            win_sigma = None

        mean_std = float(win_point.mean())                          # average in standardized space
        pred_log10 = mean_std * self.std.std + self.std.mean
        result = {
            "pred_cP": float(10.0 ** pred_log10),
            "pred_log10_cP": float(pred_log10),
            "window_pred_cP": [float(10.0 ** (p * self.std.std + self.std.mean)) for p in win_point],
        }
        if self.gmm_k:
            # per-window mixture std -> averaged, expressed in log10(cP) units
            sigma_log10 = float(win_sigma.mean()) * self.std.std
            result["pred_log10_sigma"] = sigma_log10
            # approximate +/-1 sigma band in cP
            result["cP_lo"] = float(10.0 ** (pred_log10 - sigma_log10))
            result["cP_hi"] = float(10.0 ** (pred_log10 + sigma_log10))
        return result


def load_predictor(kind: str, weights_dir: str = "weights", device=None) -> ViscNetPredictor:
    """kind = 'cp' (regression) or 'gmm' (K=5 uncertainty)."""
    std_path = f"{weights_dir}/standardizer_seed1206.json"
    if kind == "cp":
        return ViscNetPredictor(f"{weights_dir}/cp_regression_seed1206.pth", std_path, gmm_k=0, device=device)
    if kind == "gmm":
        return ViscNetPredictor(f"{weights_dir}/gmm_k5_seed1206.pth", std_path, gmm_k=5, device=device)
    raise ValueError(f"unknown kind {kind!r} (use 'cp' or 'gmm')")
