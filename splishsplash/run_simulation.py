from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pysplishsplash as sph


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--property-index", type=int, default=1)
    parser.add_argument("--rpm", type=float, default=270.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_property(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["SampleIndex"])) == index:
                return row
    raise ValueError(f"Property index not found: {index}")


def main() -> None:
    args = parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    scene = json.loads((ROOT / config["scene"]).read_text(encoding="utf-8"))
    row = load_property(ROOT / config["properties"], args.property_index)

    density = float(row["Density_kg_per_m3"])
    viscosity = float(row["KinematicViscosity_m2_per_s"])
    surface_tension = float(row["SurfaceTension_N_per_m"])
    omega = 2.0 * math.pi * args.rpm / 60.0

    scene["Configuration"]["density0"] = density
    material = scene["Materials"][0]
    material["surfaceTension"] = surface_tension
    material["viscosity"] = viscosity
    material["viscosityBoundary"] = viscosity
    material["Weiler et al. 2018"]["viscosity"] = viscosity
    material["Weiler et al. 2018"]["viscosityBoundary"] = viscosity
    scene["TargetVelocityMotorHingeJoints"][0]["targetSequence"] = [
        0,
        omega,
        config["power_cut_s"],
        omega,
        config["motor_stop_s"],
        0,
        config["duration_s"],
        0,
    ]

    for body in scene["RigidBodies"]:
        geometry = Path(body["geometryFile"])
        if not geometry.is_absolute():
            body["geometryFile"] = str((ROOT / geometry).resolve())

    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output path is not an empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    scene_path = output / "scene_resolved.json"
    scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    metadata = {
        "property_index": args.property_index,
        "dynamic_viscosity_cP": float(row["DynamicViscosity_cP"]),
        "density_kg_m3": density,
        "kinematic_viscosity_m2_s": viscosity,
        "surface_tension_N_m": surface_tension,
        "rpm": args.rpm,
        "duration_s": config["duration_s"],
        "export_fps": config["export_fps"],
        "geometry": config["geometry"],
        "target_sequence": scene["TargetVelocityMotorHingeJoints"][0]["targetSequence"],
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    simulator = sph.Exec.SimulatorBase()
    simulator.init(
        useGui=False,
        initialPause=False,
        outputDir=str(output),
        sceneFile=str(scene_path),
        stopAt=config["duration_s"],
    )
    simulator.setValueFloat(simulator.STOP_AT, config["duration_s"])
    simulator.activateExporter("VTK Exporter", True)
    simulator.activateExporter("Rigid Body OBJ Exporter", True)
    simulator.setValueFloat(simulator.DATA_EXPORT_FPS, config["export_fps"])
    simulator.run()


if __name__ == "__main__":
    main()
