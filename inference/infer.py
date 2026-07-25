#!/usr/bin/env python3
"""Run ViscNet 5-window inference over the bundled 50-clip sample set.

Examples
--------
    python infer.py --model cp                 # cP point-estimate model
    python infer.py --model gmm                # GMM (K=5) with uncertainty
    python infer.py --model cp --out preds.csv # also write a CSV

Prints a per-clip table (true vs predicted cP) and a summary MAE / MAPE.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from viscnet_infer.predict import load_predictor


def main() -> None:
    ap = argparse.ArgumentParser(description="ViscNet minimal 5-window inference")
    ap.add_argument("--model", choices=["cp", "gmm"], default="cp",
                    help="cp = dimensionless-transfer cP regression; gmm = GMM (K=5) uncertainty")
    ap.add_argument("--data", default="data/clips", help="folder of .mp4 clips")
    ap.add_argument("--labels", default="data/labels.json", help="ground-truth labels JSON")
    ap.add_argument("--weights", default="weights", help="weights folder")
    ap.add_argument("--out", default=None, help="optional CSV output path")
    ap.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    args = ap.parse_args()

    predictor = load_predictor(args.model, weights_dir=args.weights, device=args.device)
    labels = {row["name"]: row for row in json.load(open(args.labels))}
    files = sorted(f for f in os.listdir(args.data) if f.endswith(".mp4"))

    rows, abs_errs, apes = [], [], []
    is_gmm = args.model == "gmm"
    header = f"{'clip':38s} {'true_cP':>9s} {'pred_cP':>9s} {'APE%':>7s}"
    if is_gmm:
        header += f" {'+/-sig(cP)':>12s}"
    print(header)
    print("-" * len(header))

    for fn in files:
        name = fn[:-4]
        res = predictor.predict_clip(os.path.join(args.data, fn))
        true = labels.get(name, {}).get("cP")
        row = {"name": name, "pred_cP": round(res["pred_cP"], 4)}
        line = f"{name:38s} "
        if true is not None:
            true = float(true)
            ae = abs(res["pred_cP"] - true)
            ape = ae / true * 100.0
            abs_errs.append(ae)
            apes.append(ape)
            row.update(true_cP=round(true, 4), abs_err_cP=round(ae, 4), APE_percent=round(ape, 2))
            line += f"{true:9.3f} {res['pred_cP']:9.3f} {ape:7.2f}"
        else:
            line += f"{'--':>9s} {res['pred_cP']:9.3f} {'--':>7s}"
        if is_gmm:
            row["pred_log10_sigma"] = round(res["pred_log10_sigma"], 4)
            row["cP_lo"] = round(res["cP_lo"], 4)
            row["cP_hi"] = round(res["cP_hi"], 4)
            band = (res["cP_hi"] - res["cP_lo"]) / 2.0
            line += f" {band:12.3f}"
        print(line)
        rows.append(row)

    if abs_errs:
        mae = sum(abs_errs) / len(abs_errs)
        mape = sum(apes) / len(apes)
        print("-" * len(header))
        print(f"n={len(abs_errs)}   cP MAE = {mae:.3f}   MAPE = {mape:.2f}%   (model: {args.model})")

    if args.out:
        keys = sorted({k for r in rows for k in r})
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["name"] + [k for k in keys if k != "name"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
