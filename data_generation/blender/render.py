from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--fluid-dir", type=Path, required=True)
    parser.add_argument("--impeller-dir", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("CPU", "CUDA", "OPTIX"), default="OPTIX")
    return parser.parse_args(values)


def set_input(node, names: tuple[str, ...], value) -> None:
    for name in names:
        if name in node.inputs:
            node.inputs[name].default_value = value
            return


def principled_material(name: str, base_color, roughness, transmission, ior, metallic=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    set_input(shader, ("Base Color",), base_color)
    set_input(shader, ("Roughness",), roughness)
    set_input(shader, ("Transmission Weight", "Transmission"), transmission)
    set_input(shader, ("IOR",), ior)
    set_input(shader, ("Metallic",), metallic)
    return material


def impeller_material(metallic: float):
    material = principled_material("impeller", (0.33, 0.33, 0.33, 1), 0.35, 0.0, 1.45, metallic)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes["Principled BSDF"]
    texture = nodes.new("ShaderNodeTexNoise")
    texture.inputs["Scale"].default_value = 4
    texture.inputs["Detail"].default_value = 10
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.472
    ramp.color_ramp.elements[1].position = 0.91
    links.new(texture.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Roughness"])
    return material


def image_material(path: Path):
    material = principled_material("background", (1, 1, 1, 1), 1.0, 0.0, 1.45)
    nodes = material.node_tree.nodes
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(str(path.resolve()))
    texture.interpolation = "Closest"
    material.node_tree.links.new(texture.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    return material


def add_area_light(position, rotation, size, size_y, power):
    bpy.ops.object.light_add(type="AREA", location=position)
    light = bpy.context.active_object
    light.rotation_euler = rotation
    light.data.shape = "RECTANGLE"
    light.data.size = size
    light.data.size_y = size_y
    light.data.energy = power


def import_obj(path: Path, material, rotation, scale):
    existing = {obj.name for obj in bpy.data.objects}
    bpy.ops.wm.obj_import(filepath=str(path.resolve()))
    objects = [obj for obj in bpy.data.objects if obj.name not in existing]
    for obj in objects:
        obj.rotation_euler = rotation
        obj.scale = scale
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(material)
    return objects


def locate_frame(directory: Path, stem: str, frame: int) -> Path:
    for name in (f"{stem}_{frame}.obj", f"{stem}_{frame:03d}.obj", f"{stem}_{frame:04d}.obj"):
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"{stem}, frame {frame}, in {directory}")


def configure_cycles(scene, backend: str) -> None:
    if backend == "CPU":
        scene.cycles.device = "CPU"
        return
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = backend
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type == backend
    scene.cycles.device = "GPU"


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    fluid_cfg = config["fluid"]
    fluid = principled_material(
        "fluid",
        (1, 1, 1, 1),
        rng.uniform(*fluid_cfg["roughness_range"]),
        rng.uniform(*fluid_cfg["transmission_range"]),
        fluid_cfg["ior"] + rng.uniform(-fluid_cfg["ior_jitter"], fluid_cfg["ior_jitter"]),
    )
    glass_cfg = config["glass"]
    glass = principled_material(
        "glass",
        (1, 1, 1, 1),
        glass_cfg["roughness"],
        glass_cfg["transmission"],
        glass_cfg["ior"],
    )
    impeller_cfg = config["impeller"]
    steel = impeller_material(
        impeller_cfg["metallic"]
        + rng.uniform(-impeller_cfg["metallic_jitter"], impeller_cfg["metallic_jitter"])
    )

    import_obj(ROOT / "assets" / "vessel.obj", glass, (0, 0, math.radians(90)), (0.1, 0.1, 0.1))
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, -0.05))
    plane = bpy.context.active_object
    plane.scale = (2, 2, 2)
    plane.data.materials.append(image_material(args.background))

    camera_cfg = config["camera"]
    bpy.ops.object.camera_add(
        location=camera_cfg["position_m"],
        rotation=tuple(math.radians(value) for value in camera_cfg["rotation_deg"]),
    )
    camera = bpy.context.active_object
    camera.data.lens = camera_cfg["lens_mm"]
    bpy.context.scene.camera = camera

    spot_cfg = config["spot_light"]
    bpy.ops.object.light_add(type="SPOT", location=spot_cfg["position_m"])
    spot = bpy.context.active_object
    spot.rotation_euler = tuple(math.radians(value) for value in spot_cfg["rotation_deg"])
    spot.data.energy = spot_cfg["power_W"]
    spot.data.spot_size = math.radians(spot_cfg["cone_deg"])

    area_cfg = config["area_lights"]
    base_rotation = area_cfg["rotation_deg"]
    for position in area_cfg["positions_m"]:
        jittered_position = tuple(
            value + rng.uniform(-area_cfg["position_jitter_m"], area_cfg["position_jitter_m"])
            for value in position
        )
        jittered_rotation = tuple(
            math.radians(value + rng.uniform(-area_cfg["rotation_jitter_deg"], area_cfg["rotation_jitter_deg"]))
            for value in base_rotation
        )
        add_area_light(
            jittered_position,
            jittered_rotation,
            area_cfg["size_m"][0],
            area_cfg["size_m"][1],
            area_cfg["power_W"] + rng.uniform(-area_cfg["power_jitter_W"], area_cfg["power_jitter_W"]),
        )

    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    world_shader = scene.world.node_tree.nodes["Background"]
    world_shader.inputs["Color"].default_value = config["world"]["color"]
    world_shader.inputs["Strength"].default_value = config["world"]["strength"]
    scene.render.engine = "CYCLES"
    configure_cycles(scene, args.device)
    scene.cycles.samples = config["cycles_samples"]
    scene.render.resolution_x, scene.render.resolution_y = config["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = config["fps"]
    scene.frame_start = 1
    scene.frame_end = config["frame_end"] - config["frame_start"] + 1
    if hasattr(scene.render, "use_motion_blur"):
        scene.render.use_motion_blur = config["motion_blur"]

    dynamic_objects = []
    rotation = (math.radians(90), 0, 0)
    scale = (10, 10, 10)
    for output_frame, source_frame in enumerate(
        range(config["frame_start"], config["frame_end"] + 1),
        start=1,
    ):
        for obj in dynamic_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        dynamic_objects = import_obj(
            locate_frame(args.fluid_dir, "ParticleData_Fluid", source_frame),
            fluid,
            rotation,
            scale,
        )
        dynamic_objects += import_obj(
            locate_frame(args.impeller_dir, "rb_data_1", source_frame),
            steel,
            rotation,
            scale,
        )
        scene.frame_set(output_frame)
        scene.render.filepath = str(args.output / f"{output_frame:04d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
