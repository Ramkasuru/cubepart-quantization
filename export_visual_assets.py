"""
Exports Cube3D whole meshes and CubePart part meshes as .glb for the 36 prompts from
select_visual_prompts.py. CubePart configs A and D by default, --all-configs adds B and C.
Uses the helpers in benchmark_4config_sweep.py, same environment:

  export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_DISABLE=1
  python export_visual_assets.py --selected visual_prompts_36.json --all-configs
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from torchao.quantization import quantize_, int4_weight_only

from benchmark_4config_sweep import (
    BASELINE_DIR, INT4_DIR, CONFIG_PATH, CHECKPOINT_PATH, VAE_CHECKPOINT_PATH,
    GROUP_SIZE, SKIP_SUBSTRINGS, build_filter_fn, load_mesh_from_path, generate_parts,
)
from cube_part.pipelines import PartShapeDenoiserPipeline

OUT_DIR = Path("results/visual_assets")


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")[:40]


def export_mesh(mesh, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def export_parts(idx: str, tag: str, part_names, part_meshes, manifest_entry: dict):
    paths = []
    for n, (verts, faces) in enumerate(part_meshes):
        if verts is None:
            continue
        p = OUT_DIR / "cubepart" / f"{idx}_{tag}_part{n:02d}_{safe(part_names[n])}.glb"
        export_mesh(trimesh.Trimesh(verts, faces), p)
        paths.append({"part_index": n, "part_name": part_names[n], "file": str(p)})
    manifest_entry[f"cubepart_{tag}_parts"] = paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", type=str, default="visual_prompts_36.json")
    ap.add_argument("--out-manifest", type=str, default=str(OUT_DIR / "manifest.json"))
    ap.add_argument("--all-configs", action="store_true", help="also export B and C, not just A and D")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "Run on a GPU node."
    selected = json.loads(Path(args.selected).read_text())

    flat = []
    for cat, rows in selected.items():
        for r in rows:
            flat.append({**r, "category": cat})
    print(f">> {len(flat)} prompts selected across {len(selected)} categories")

    manifest = {}

    # 1. Cube3D whole meshes -> .glb (no GPU needed)
    for r in flat:
        idx = r["idx"]
        bf16_path = BASELINE_DIR / f"bf16_{idx}.npz"
        int4_path = INT4_DIR / f"rtn_int4_w4a16_{idx}.npz"
        entry = manifest.setdefault(idx, {"prompt": r["prompt"], "category": r["category"], "parts_list": r["parts"]})
        bf16_out = OUT_DIR / "cube3d" / f"{idx}_cube3d_bf16.glb"
        int4_out = OUT_DIR / "cube3d" / f"{idx}_cube3d_int4.glb"
        export_mesh(load_mesh_from_path(bf16_path), bf16_out)
        export_mesh(load_mesh_from_path(int4_path), int4_out)
        entry["cube3d_bf16"] = str(bf16_out)
        entry["cube3d_int4"] = str(int4_out)
    print(">> Cube3D whole meshes exported")

    # 2. CubePart parts, config A (bf16/bf16)
    print("\n>> Loading BF16 pipeline")
    pipe = PartShapeDenoiserPipeline(
        config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH,
        vae_checkpoint_path=VAE_CHECKPOINT_PATH,
        extract_geometry_fn_name="extract_geometry_coarse_to_fine",
    )
    c_results = {}
    for r in flat:
        idx, parts = r["idx"], r["parts"]
        seed = int(idx)
        print(f"[A] mesh {idx}, parts={parts}")
        mesh = load_mesh_from_path(BASELINE_DIR / f"bf16_{idx}.npz")
        part_meshes, _ = generate_parts(pipe, mesh, parts, seed=seed)
        export_parts(idx, "bf16", parts, part_meshes, manifest[idx])
        if args.all_configs:
            print(f"[C] mesh {idx} (int4 input, bf16 model)")
            mesh_c = load_mesh_from_path(INT4_DIR / f"rtn_int4_w4a16_{idx}.npz")
            c_parts, _ = generate_parts(pipe, mesh_c, parts, seed=seed)
            c_results[idx] = c_parts

    # 3. quantize in place, then config D (int4/int4) [+ B if --all-configs]
    print("\n>> Quantizing diffusion_model in-place (HQQ W4A16 g=128)")
    quantize_(pipe.system.diffusion_model, int4_weight_only(group_size=GROUP_SIZE, use_hqq=True),
               filter_fn=build_filter_fn(SKIP_SUBSTRINGS))
    torch.cuda.synchronize()

    for r in flat:
        idx, parts = r["idx"], r["parts"]
        seed = int(idx)
        print(f"[D] mesh {idx} (int4 input, int4 model)")
        mesh_d = load_mesh_from_path(INT4_DIR / f"rtn_int4_w4a16_{idx}.npz")
        d_parts, _ = generate_parts(pipe, mesh_d, parts, seed=seed)
        export_parts(idx, "int4", parts, d_parts, manifest[idx])
        if args.all_configs:
            print(f"[B] mesh {idx} (bf16 input, int4 model)")
            mesh_b = load_mesh_from_path(BASELINE_DIR / f"bf16_{idx}.npz")
            b_parts, _ = generate_parts(pipe, mesh_b, parts, seed=seed)
            export_parts(idx, "bf16input_int4model", parts, b_parts, manifest[idx])
            export_parts(idx, "int4input_bf16model", parts, c_results[idx], manifest[idx])

    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_manifest).write_text(json.dumps(manifest, indent=2))
    print(f"\n>> Done. Assets in {OUT_DIR}/, manifest at {args.out_manifest}")


if __name__ == "__main__":
    main()
