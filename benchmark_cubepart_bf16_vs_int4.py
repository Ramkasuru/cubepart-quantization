"""
BF16 vs RTN INT4 CubePart on the bundled example meshes.
Both runs are in one process and the model is quantized in place after the BF16 pass,
since the torchao quantized tensors do not support cpu->cuda .to().
Chamfer distance per matched part. Part lists from examples/inputs/examples.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torchao.quantization import quantize_, int4_weight_only

from cube_part.pipelines import PartShapeDenoiserPipeline, ShapeInput
from cube_part.utils.mesh import load_mesh, sample_surface

CONFIG_PATH = "configs/shape_denoiser_multimesh.yaml"
CHECKPOINT_PATH = "weights/multi_part_dit.safetensors"
VAE_CHECKPOINT_PATH = "weights/vae.safetensors"
GROUP_SIZE = 128

# part lists from examples/inputs/examples.json
MESH_PARTS = {
    "jellyfish_car": ["body", "front right wheel", "front left wheel", "rear right wheel",
                       "rear left wheel", "exhaust pipe", "headlights", "gun"],
    "drone": ["body", "blades", "landing legs"],
    "wizard": ["head", "feather", "hat", "torso body", "cloak", "hands", "magic staff", "orb"],
    "pirate_chest": ["lid", "base"],
    "flower": ["flower stem", "flower petals", "leaves"],
}

SKIP_SUBSTRINGS = {"img_in", "txt_in", "proj_out", "pos_embed", "time_text_embed", "norm"}


def build_filter_fn(skip_substrings):
    def _filter(module, fqn: str) -> bool:
        return not any(s in fqn for s in skip_substrings)
    return _filter


def chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    tree_a, tree_b = cKDTree(pts_a), cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)
    d_ba, _ = tree_a.query(pts_b, k=1)
    return float(d_ab.mean() + d_ba.mean())


def generate_parts(pipe, mesh_path: str, part_names: list, guidance_scale=7.5, num_inference_steps=50):
    mesh, _, _ = load_mesh(mesh_path)
    surface = sample_surface(mesh, num_samples=128_000)
    surface = torch.from_numpy(surface).to(pipe.device).unsqueeze(0).float()
    latents, _ = pipe.encode_shape(surface)

    t0 = time.time()
    part_meshes = pipe.input_to_part_shape(
        ShapeInput(prompt=[part_names], latents=latents),
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    latency = time.time() - t0
    return part_meshes, latency


def main():
    assert torch.cuda.is_available(), "Run on a GPU node."

    print(">> Loading BF16 CubePart pipeline")
    pipe = PartShapeDenoiserPipeline(
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        vae_checkpoint_path=VAE_CHECKPOINT_PATH,
        extract_geometry_fn_name="extract_geometry_coarse_to_fine",
    )

    results = {}

    print(">> Running BF16 generation on all example meshes")
    bf16_outputs = {}
    torch.cuda.reset_peak_memory_stats()
    for name, parts in MESH_PARTS.items():
        mesh_path = f"examples/inputs/{name}.glb"
        if not Path(mesh_path).exists():
            print(f"   [!] skipping {name}: mesh not found at {mesh_path}")
            continue
        print(f"   generating {name} ({len(parts)} parts)...")
        part_meshes, latency = generate_parts(pipe, mesh_path, parts)
        bf16_outputs[name] = part_meshes
        results.setdefault(name, {})["bf16_latency_s"] = round(latency, 2)
        print(f"      done in {latency:.1f}s")
    bf16_peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"   BF16 peak VRAM: {bf16_peak_vram:.2f} GB")

    print(f"\n>> Quantizing diffusion_model in-place (RTN W4A16, group_size={GROUP_SIZE})")
    filter_fn = build_filter_fn(SKIP_SUBSTRINGS)
    quantize_(pipe.system.diffusion_model, int4_weight_only(group_size=GROUP_SIZE), filter_fn=filter_fn)

    print(">> Running INT4 generation on all example meshes")
    int4_outputs = {}
    torch.cuda.reset_peak_memory_stats()
    for name, parts in MESH_PARTS.items():
        mesh_path = f"examples/inputs/{name}.glb"
        if not Path(mesh_path).exists():
            continue
        print(f"   generating {name} ({len(parts)} parts)...")
        part_meshes, latency = generate_parts(pipe, mesh_path, parts)
        int4_outputs[name] = part_meshes
        results.setdefault(name, {})["int4_latency_s"] = round(latency, 2)
        print(f"      done in {latency:.1f}s")
    int4_peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"   INT4 peak VRAM: {int4_peak_vram:.2f} GB")

    print("\n>> Computing Chamfer distance per part")
    all_cds = []
    for name in bf16_outputs:
        if name not in int4_outputs:
            continue
        part_cds = []
        for i, (bf16_part, int4_part) in enumerate(zip(bf16_outputs[name], int4_outputs[name])):
            bf16_v, bf16_f = bf16_part
            int4_v, int4_f = int4_part
            if bf16_v is None or int4_v is None:
                continue
            import trimesh
            pts_a = trimesh.Trimesh(bf16_v, bf16_f).sample(5000)
            pts_b = trimesh.Trimesh(int4_v, int4_f).sample(5000)
            cd = chamfer_distance(np.asarray(pts_a), np.asarray(pts_b))
            part_cds.append(cd)
            all_cds.append(cd)
        results[name]["chamfer_distance_per_part"] = [round(c, 5) for c in part_cds]
        results[name]["chamfer_distance_mean"] = round(float(np.mean(part_cds)), 5) if part_cds else None

    summary = {
        "vram_gb": {"bf16_peak": round(bf16_peak_vram, 2), "int4_peak": round(int4_peak_vram, 2)},
        "chamfer_distance_median_across_all_parts": round(float(np.median(all_cds)), 5) if all_cds else None,
        "chamfer_distance_mean_across_all_parts": round(float(np.mean(all_cds)), 5) if all_cds else None,
        "per_mesh": results,
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/cubepart_bf16_vs_int4.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + json.dumps(summary, indent=2))
    print("\n>> Wrote results/cubepart_bf16_vs_int4.json")


if __name__ == "__main__":
    main()
