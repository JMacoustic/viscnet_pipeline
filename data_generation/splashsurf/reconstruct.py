from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import meshio
import numpy as np
import pysplashsurf


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    return parser.parse_args()


def frame_number(path: Path) -> int | None:
    match = re.search(r"_(\d+)\.vtk$", path.name)
    return int(match.group(1)) if match else None


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    parameters = {
        key: value
        for key, value in config.items()
        if key not in {"frame_start", "frame_end"}
    }

    for vtk_path in sorted(args.input.glob("ParticleData_Fluid_*.vtk")):
        frame = frame_number(vtk_path)
        if frame is None or not config["frame_start"] <= frame <= config["frame_end"]:
            continue

        particles = np.asarray(meshio.read(vtk_path).points, dtype=np.float32)
        if not np.isfinite(particles).all():
            raise ValueError(f"Non-finite particles in {vtk_path}")

        mesh_data, _ = pysplashsurf.reconstruction_pipeline(particles, **parameters)
        surface = mesh_data.mesh
        output_path = args.output / f"ParticleData_Fluid_{frame:03d}.obj"
        meshio.write(
            output_path,
            meshio.Mesh(
                points=np.asarray(surface.vertices),
                cells=[("triangle", np.asarray(surface.triangles))],
            ),
        )
        print(f"{vtk_path} -> {output_path}")


if __name__ == "__main__":
    main()
