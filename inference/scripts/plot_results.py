#!/usr/bin/env python3
"""Make the parity figure for the bundled 100-clip sample.

Reads `data/labels.json` (ground truth) and the two prediction CSVs written by
`infer.py --out` (`results/predictions_cp.csv`, `results/predictions_gmm.csv`),
and draws predicted-vs-true viscosity on log-log axes: dots on the diagonal are
perfect predictions. The right panel (GMM) adds +/-1 sigma error bars.

    python scripts/plot_results.py            # -> results/parity.png

Only needs matplotlib + numpy (no torch).
"""

from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # inference/
RESULTS = os.path.join(HERE, "results")


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def load():
    labels = {r["name"]: r for r in json.load(open(os.path.join(HERE, "data/labels.json")))}
    cp = read_csv(os.path.join(RESULTS, "predictions_cp.csv"))
    gmm = read_csv(os.path.join(RESULTS, "predictions_gmm.csv"))
    return labels, cp, gmm


def _arr(rows, key):
    return np.array([float(r[key]) for r in rows], dtype=float)


def _mae_mape(true, pred):
    ae = np.abs(pred - true)
    return ae.mean(), (ae / true * 100.0).mean()


def panel(ax, true, pred, cls, title, sigma=None):
    lo = min(true.min(), pred.min()) * 0.6
    hi = max(true.max(), pred.max()) * 1.6
    ax.plot([lo, hi], [lo, hi], "--", color="0.4", lw=1.0, zorder=1, label="perfect (y = x)")
    if sigma is not None:
        ax.errorbar(true, pred, yerr=sigma, fmt="none", ecolor="0.7",
                    elinewidth=0.7, capsize=1.5, zorder=2)
    sc = ax.scatter(true, pred, c=cls, cmap="viridis", s=26, edgecolor="white",
                    linewidth=0.4, zorder=3, vmin=0, vmax=9)
    mae, mape = _mae_mape(true, pred)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("true viscosity  (cP)")
    ax.set_ylabel("predicted viscosity  (cP)")
    ax.set_title(title, fontsize=11)
    ax.text(0.04, 0.96, f"MAE = {mae:.2f} cP\nMAPE = {mape:.1f} %",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.6))
    ax.grid(True, which="both", ls="--", lw=0.35, alpha=0.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    return sc


def main():
    labels, cp, gmm = load()

    cp_true = _arr(cp, "true_cP"); cp_pred = _arr(cp, "pred_cP")
    cp_cls = np.array([int(labels[r["name"]]["viscosity_class"]) for r in cp])

    gm_true = _arr(gmm, "true_cP"); gm_pred = _arr(gmm, "pred_cP")
    gm_cls = np.array([int(labels[r["name"]]["viscosity_class"]) for r in gmm])
    gm_lo = _arr(gmm, "cP_lo"); gm_hi = _arr(gmm, "cP_hi")
    gm_sig = np.vstack([gm_pred - gm_lo, gm_hi - gm_pred])  # asymmetric in cP space

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    panel(axes[0], cp_true, cp_pred, cp_cls,
          "(a) Point estimate  —  cp model")
    sc = panel(axes[1], gm_true, gm_pred, gm_cls,
               "(b) With uncertainty  —  gmm model (K = 5)", sigma=gm_sig)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.035, pad=0.02, ticks=range(0, 10))
    cbar.set_label("viscosity class (0 = thinnest → 9 = thickest)", fontsize=9)
    fig.suptitle("ViscNet viscosity predictions on 100 held-out sample videos",
                 fontsize=12, y=0.98)
    out = os.path.join(RESULTS, "parity.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
    print(f"cp : MAE {_mae_mape(cp_true, cp_pred)[0]:.3f} cP  MAPE {_mae_mape(cp_true, cp_pred)[1]:.2f} %")
    print(f"gmm: MAE {_mae_mape(gm_true, gm_pred)[0]:.3f} cP  MAPE {_mae_mape(gm_true, gm_pred)[1]:.2f} %")


if __name__ == "__main__":
    main()
