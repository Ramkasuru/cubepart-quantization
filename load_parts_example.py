"""
Per part mesh access for the CubePart INT4 mesh package (visual_assets_36prompts.zip).

Layout after unzipping:
  visual_assets/manifest.json
  visual_assets/cube3d/<idx>_cube3d_bf16.glb         Cube3D BF16, whole mesh
  visual_assets/cube3d/<idx>_cube3d_int4.glb         Cube3D INT4, whole mesh
  visual_assets/cubepart/<idx>_bf16_part<NN>_<name>.glb   CubePart BF16, one file per part
  visual_assets/cubepart/<idx>_int4_part<NN>_<name>.glb   CubePart INT4, one file per part

manifest.json is keyed by prompt idx. Each entry has:
  prompt, category, parts_list       the prompt text and the ordered part names
  cube3d_bf16, cube3d_int4           whole mesh files
  cubepart_bf16_parts                list of {part_index, part_name, file}
  cubepart_int4_parts                same, for the INT4 model

part_index is the position of the part in parts_list. A part the model returned
empty is skipped, so a part list can be shorter than parts_list.
File paths in the manifest are absolute, so only the file name is used and it is
looked up under the unzipped folder.

Usage:
  pip install trimesh numpy
  python load_parts_example.py visual_assets 130
"""
import json
import sys
from pathlib import Path

import trimesh


def load_glb(path: Path) -> trimesh.Trimesh:
    return trimesh.load(str(path), force="mesh")


def load_prompt(root: Path, idx: str) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest[idx]

    out = {
        "prompt": entry["prompt"],
        "category": entry["category"],
        "parts_list": entry["parts_list"],
        "cube3d_bf16": load_glb(root / "cube3d" / Path(entry["cube3d_bf16"]).name),
        "cube3d_int4": load_glb(root / "cube3d" / Path(entry["cube3d_int4"]).name),
    }
    for tag in ("bf16", "int4"):
        parts = {}
        for p in entry[f"cubepart_{tag}_parts"]:
            parts[p["part_name"]] = load_glb(root / "cubepart" / Path(p["file"]).name)
        out[f"cubepart_{tag}_parts"] = parts
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("visual_assets")
    idx = sys.argv[2] if len(sys.argv) > 2 else "130"

    d = load_prompt(root, idx)
    print(f"idx {idx} | {d['category']} | {d['prompt']}")
    print("parts_list:", d["parts_list"])

    for key in ("cube3d_bf16", "cube3d_int4"):
        m = d[key]
        print(f"\n{key}: {len(m.vertices)} verts, {len(m.faces)} faces")

    for tag in ("bf16", "int4"):
        parts = d[f"cubepart_{tag}_parts"]
        print(f"\ncubepart_{tag}: {len(parts)} parts")
        for name, m in parts.items():
            print(f"  {name}: {len(m.vertices)} verts, {len(m.faces)} faces")
        if parts:
            whole = trimesh.util.concatenate(list(parts.values()))
            print(f"  union of parts bounds: {whole.bounds.round(3).tolist()}")

    print("\ncube3d_bf16 bounds:", d["cube3d_bf16"].bounds.round(3).tolist())

    # one scene per model with every part as its own node, for viewing or custom rendering
    scene = trimesh.Scene()
    for name, m in d["cubepart_bf16_parts"].items():
        scene.add_geometry(m, node_name=name, geom_name=name)
    print("\nscene nodes (CubePart BF16):", list(scene.graph.nodes_geometry))


if __name__ == "__main__":
    main()
