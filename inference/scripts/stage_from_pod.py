#!/usr/bin/env python3
"""Populate the package with model files, weights, and 50 sample clips.

Runs ON the ViscNet pod (needs /root/Viscnet). Copies the 3 ViViT model files
(with imports rewritten to the package layout and timm made optional), the two
seed-1206 checkpoints + their standardizer, and 50 real test clips
(10 viscosity classes x 5 clips with varied pattern/RPM) with a labels.json.

Idempotent: safe to re-run.
"""

import collections
import json
import os
import shutil

REPO = "/root/Viscnet"
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # package root
TEST_MANIFEST = f"{REPO}/dataset/realdata_auto_260630_less_bubble/real_test_seed1206/manifest.json"


def _patch(src, dst, replacements, lazy_timm):
    text = open(src).read()
    for a, b in replacements:
        text = text.replace(a, b)
    if lazy_timm:
        text = text.replace(
            "import timm\n",
            "try:\n    import timm\nexcept Exception:  # optional; only used by the (disabled) pattern branch\n    timm = None\n",
            1,
        )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write(text)


def copy_model_files():
    _patch(
        f"{REPO}/src/models/VivitEmbed.py",
        f"{PKG}/viscnet_infer/vivit_embed.py",
        [
            ("from models.vivit.configuration_vivit import VivitConfig",
             "from .vivit.configuration_vivit import VivitConfig"),
            ("from models.vivit.modeling_vivit import VivitModel",
             "from .vivit.modeling_vivit import VivitModel"),
        ],
        lazy_timm=True,
    )
    _patch(
        f"{REPO}/src/models/vivit/modeling_vivit.py",
        f"{PKG}/viscnet_infer/vivit/modeling_vivit.py",
        [("from models.vivit.configuration_vivit import VivitConfig",
          "from .configuration_vivit import VivitConfig")],
        lazy_timm=True,
    )
    _patch(
        f"{REPO}/src/models/vivit/configuration_vivit.py",
        f"{PKG}/viscnet_infer/vivit/configuration_vivit.py",
        [],
        lazy_timm=False,
    )
    print("copied + patched 3 model files")


def copy_weights():
    os.makedirs(f"{PKG}/weights", exist_ok=True)
    pairs = [
        (f"{REPO}/outputs/transfer/checkpoints/gooddim_cp_s1206_lr1.5e-04.pth",
         f"{PKG}/weights/cp_regression_seed1206.pth"),
        (f"{REPO}/outputs/transfer/checkpoints/g3s_k5_s1206.pth",
         f"{PKG}/weights/gmm_k5_seed1206.pth"),
        (f"{REPO}/outputs/transfer/gooddim_cp_s1206_lr1.5e-04/target_standardizer.json",
         f"{PKG}/weights/standardizer_seed1206.json"),
    ]
    for src, dst in pairs:
        shutil.copy(src, dst)
    print("copied 2 checkpoints + 1 standardizer")


def _meta(name):
    parts = name.split("_")
    return (
        int(parts[0].replace("class", "")),
        int(parts[2].replace("pattern", "")),
        int(parts[3].replace("RPM", "")),
    )


def select_clips(per_class=5):
    samples = json.load(open(TEST_MANIFEST))
    by_class = collections.defaultdict(list)
    for s in samples:
        cls, pat, rpm = _meta(s["name"])
        by_class[cls].append((pat, rpm, s))

    selected = []
    for cls in sorted(by_class):
        by_pat = collections.defaultdict(list)
        for pat, rpm, s in by_class[cls]:
            by_pat[pat].append((rpm, s))
        pats = sorted(by_pat)
        picked = []
        # one clip per pattern (up to per_class patterns), varying RPM across patterns
        for i, pat in enumerate(pats[:per_class]):
            lst = sorted(by_pat[pat], key=lambda t: t[0])  # by RPM
            rpm, s = lst[(i * len(lst) // per_class) % len(lst)]
            picked.append(s)
        # if fewer than per_class distinct patterns, fill from remaining clips
        if len(picked) < per_class:
            names = {s["name"] for s in picked}
            for pat, rpm, s in sorted(by_class[cls], key=lambda t: (t[0], t[1])):
                if s["name"] not in names:
                    picked.append(s)
                    names.add(s["name"])
                if len(picked) == per_class:
                    break
        selected.extend(picked[:per_class])
    return selected


def copy_clips_and_labels(selected):
    clips_dir = f"{PKG}/data/clips"
    os.makedirs(clips_dir, exist_ok=True)
    labels = []
    for s in selected:
        name = s["name"]
        shutil.copy(s["video_path"], f"{clips_dir}/{name}.mp4")
        p = json.load(open(s["parameters_path"]))

        def num(key):
            v = p.get(key)
            return float(v) if v is not None else None

        labels.append({
            "name": name,
            "cP": float(p["dynamic_viscosity_cP"]),
            "viscosity_class": int(p["class"]),
            "RPM": num("RPM"),
            "pattern": int(p["pattern"]),
            "light": int(p["light"]),
            "density_kg_m3": num("density"),
            "surface_tension_N_m": num("surface_tension"),
            "kinematic_viscosity_m2_s": num("kinematic_viscosity"),
            "Ga": num("Ga"),
        })
    labels.sort(key=lambda r: (r["viscosity_class"], r["pattern"], r["RPM"]))
    json.dump(labels, open(f"{PKG}/data/labels.json", "w"), indent=1)
    print(f"copied {len(labels)} clips + labels.json")


if __name__ == "__main__":
    copy_model_files()
    copy_weights()
    sel = select_clips(per_class=5)
    copy_clips_and_labels(sel)
    print("staging complete ->", PKG)
