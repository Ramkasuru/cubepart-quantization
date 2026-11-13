"""
Maps a mesh's part files back to part names and writes them as one scene.

Usage:
    python tutorial_part_mapping.py --manifest results/visual_assets/manifest.json --idx 130 --config bf16
    python tutorial_part_mapping.py --manifest results/visual_assets/manifest.json --idx 130 --config bf16 --render out.png
    python tutorial_part_mapping.py --manifest results/visual_assets/manifest.json --idx 130 --all-4-models

--all-4-models prints the files for Cube3D BF16, Cube3D INT4, CubePart BF16 (config A) and CubePart INT4 (config D).
Needs trimesh and numpy. --render needs an offscreen backend (pyrender with OSMesa/EGL),
without it the combined .glb can be opened in any viewer.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

# one color per part index, for viewing only
PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--idx", required=True, help="e.g. 130")
    ap.add_argument("--config", default="bf16", help="bf16 | int4 | bf16input_int4model | int4input_bf16model")
    ap.add_argument("--render", default=None, help="optional output PNG path (needs a working offscreen GL backend)")
    ap.add_argument("--combined-glb", default=None, help="optional output .glb path with all parts, one color each")
    ap.add_argument("--all-4-models", action="store_true",
                     help="print file access for Cube3D BF16/INT4 (whole mesh) + CubePart BF16/INT4 (parts) on this idx")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    entry = manifest[args.idx]

    if args.all_4_models:
        print(f"idx {args.idx}  category={entry['category']}  prompt={entry['prompt']!r}")
        print(f"parts list (schema order): {entry['parts_list']}\n")

        print("Model 1/4 -- Cube3D BF16 (whole mesh, no parts):")
        print(f"  mesh = trimesh.load({entry['cube3d_bf16']!r}, force='mesh')\n")

        print("Model 2/4 -- Cube3D INT4 (whole mesh, no parts):")
        print(f"  mesh = trimesh.load({entry['cube3d_int4']!r}, force='mesh')\n")

        for label, key in [("Model 3/4 -- CubePart BF16 (parts, config A)", "cubepart_bf16_parts"),
                            ("Model 4/4 -- CubePart INT4 (parts, config D)", "cubepart_int4_parts")]:
            print(f"{label}:")
            for p in entry.get(key, []):
                print(f"  part{p['part_index']:02d}  {p['part_name']!r:24s}  "
                      f"mesh = trimesh.load({p['file']!r}, force='mesh')")
            print()
        return

    parts = entry.get(f"cubepart_{args.config}_parts")
    if not parts:
        raise SystemExit(f"no cubepart_{args.config}_parts for idx {args.idx} -- available: "
                          f"{[k for k in entry if k.startswith('cubepart_')]}")

    print(f"idx {args.idx}  category={entry['category']}  prompt={entry['prompt']!r}")
    print(f"parts list (schema order): {entry['parts_list']}")
    print(f"\npart_index -> part_name -> mesh file  (config={args.config})")

    meshes = []
    for p in parts:
        print(f"  part{p['part_index']:02d}  {p['part_name']!r:30s}  {p['file']}")
        m = trimesh.load(p["file"], force="mesh")
        m.visual.vertex_colors = np.tile(PALETTE[p["part_index"] % len(PALETTE)] + (255,), (len(m.vertices), 1))
        meshes.append(m)

    scene = trimesh.Scene(meshes)

    if args.combined_glb:
        Path(args.combined_glb).parent.mkdir(parents=True, exist_ok=True)
        scene.export(args.combined_glb)
        print(f"\nWrote combined multi-part glb (one color per part): {args.combined_glb}")

    if args.render:
        try:
            png = scene.save_image(resolution=(1024, 1024))
            Path(args.render).write_bytes(png)
            print(f"Wrote barebone render: {args.render}")
        except Exception as e:
            print(f"\n[render skipped] offscreen rendering unavailable ({e}). "
                  f"Use --combined-glb instead and open the .glb in any viewer, "
                  f"or run this on a machine with pyrender's OSMesa/EGL backend installed.")


if __name__ == "__main__":
    main()
