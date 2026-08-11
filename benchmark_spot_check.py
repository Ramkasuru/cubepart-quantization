"""
Spot-check benchmark: BF16 vs INT4-quantized CubePart on Nisarg's 2 baseline meshes.

Routes:
  bf16_003.obj  (witch/wizard hat, watertight)      -> BF16 CubePart -> compare vs INT4 output
  bf16_130.obj  (yarn ball, non-watertight)         -> same
  rtn_int4_w4a16_003.npz (Cube3D INT4 for hat)      -> INT4 CubePart -> compare vs BF16 output
  rtn_int4_w4a16_130.npz (Cube3D INT4 for yarn)     -> same

Design choice: rather than trying to load the pre-saved multi_part_dit_rtn_int4_g128.pt
back into a fresh pipeline (untested path, likely to fail), this quantizes the
BF16 pipeline in-place after the BF16 pass. Same weights, same result, no risk
of state_dict-vs-quantized-layer mismatch on reload.

Per-mesh routing per Nisarg's instruction:
    bf16 Cube3D mesh -> bf16 CubePart
    int4 Cube3D mesh -> int4 CubePart

Then compute Chamfer distance PER PART between BF16 and INT4 CubePart outputs.

Caveats to be honest about in results:
- bf16_130 is non-watertight (yarn strands with gaps between them). CubePart's own
  docs flag this as degrading output quality regardless of quantization.
- Parts lists are best-guess based on visual inspection of the meshes; without
  Roblox-official ground-truth for these specific meshes, part correspondence is
  approximate.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from torchao.quantization import quantize_, int4_weight_only

from cube_part.pipelines import PartShapeDenoiserPipeline, ShapeInput
from cube_part.utils.mesh import sample_surface

CONFIG_PATH = "configs/shape_denoiser_multimesh.yaml"
CHECKPOINT_PATH = "weights/multi_part_dit.safetensors"
VAE_CHECKPOINT_PATH = "weights/vae.safetensors"
GROUP_SIZE = 128

SKIP_SUBSTRINGS = {"img_in", "txt_in", "proj_out", "pos_embed", "time_text_embed", "norm"}

# Mesh -> (bf16 baseline path, int4 baseline path, parts list)
# Parts lists based on visual inspection: hat has clear pointed top + wide brim,
# yarn ball is genuinely ambiguous but "strands" over a "sphere core" is a
# reasonable first cut.
BENCHMARK_SET = {
    "003_hat": {
        "bf16_mesh": "nisarg_shared/baseline_meshes/bf16_003.obj",
        "int4_mesh": "nisarg_shared/benchmark_results_rtn_int4/rtn_int4_w4a16_003.npz",
        "parts": ["pointed top", "wide brim"],
    },
    "130_yarn": {
        "bf16_mesh": "nisarg_shared/baseline_meshes/bf16_130.obj",
        "int4_mesh": "nisarg_shared/benchmark_results_rtn_int4/rtn_int4_w4a16_130.npz",
        "parts": ["strands", "sphere core"],
    },
}


def build_filter_fn(skip_substrings):
    def _filter(module, fqn: str) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        return not any(s in fqn for s in skip_substrings)
    return _filter


def chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    tree_a, tree_b = cKDTree(pts_a), cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)
    d_ba, _ = tree_a.query(pts_b, k=1)
    return float(d_ab.mean() + d_ba.mean())


def load_mesh_from_path(path: str) -> trimesh.Trimesh:
    """Loads a .obj or .npz mesh into a trimesh.Trimesh."""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        return trimesh.Trimesh(vertices=d["vertices"], faces=d["faces"])
    return trimesh.load(path, force="mesh")


def generate_parts(pipe, mesh: trimesh.Trimesh, part_names: list,
                    guidance_scale=7.5, num_inference_steps=50):
    """Encodes an input mesh, generates part meshes, returns list of (verts, faces) tuples."""
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


def save_output_meshes(part_meshes, part_names, out_dir: Path, tag: str):
    """Save each generated part to disk for later visual inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, (verts, faces) in enumerate(part_meshes):
        if verts is None or faces is None:
            saved.append(None)
            continue
        pname = part_names[i].replace(" ", "_")
        path = out_dir / f"{tag}_part_{i:02d}_{pname}.glb"
        trimesh.Trimesh(verts, faces).export(path)
        saved.append(str(path))
    return saved


def main():
    assert torch.cuda.is_available(), "Run on a GPU node."

    out_root = Path("results/spot_check")
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------- BF16 pass ----------
    print("=" * 60)
    print(">> Loading CubePart pipeline (BF16 baseline)")
    print("=" * 60)
    pipe = PartShapeDenoiserPipeline(
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        vae_checkpoint_path=VAE_CHECKPOINT_PATH,
        extract_geometry_fn_name="extract_geometry_coarse_to_fine",
    )

    bf16_outputs = {}
    torch.cuda.reset_peak_memory_stats()
    for name, cfg in BENCHMARK_SET.items():
        bf16_path = cfg["bf16_mesh"]
        if not Path(bf16_path).exists():
            print(f"   [!] {name}: missing {bf16_path}, skipping")
            continue
        print(f"\n>> BF16 CubePart on {name}")
        print(f"   input: {bf16_path}")
        print(f"   parts: {cfg['parts']}")
        input_mesh = load_mesh_from_path(bf16_path)
        print(f"   input mesh: {len(input_mesh.vertices)} verts, {len(input_mesh.faces)} faces, "
              f"watertight={input_mesh.is_watertight}")
        part_meshes, latency = generate_parts(pipe, input_mesh, cfg["parts"])
        saved = save_output_meshes(part_meshes, cfg["parts"], out_root / name, "bf16")
        bf16_outputs[name] = {"parts": part_meshes, "saved": saved, "latency": latency}
        print(f"   generated in {latency:.1f}s, saved to {out_root / name}")
    bf16_peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n   BF16 peak VRAM: {bf16_peak_vram:.2f} GB")

    # ---------- Quantize in-place ----------
    print("\n" + "=" * 60)
    print(">> Quantizing diffusion_model in-place (HQQ, W4A16, group_size=128)")
    print("=" * 60)
    diffusion_model = pipe.system.diffusion_model
    diffusion_model = diffusion_model.to(torch.float32)  # required for HQQ packing math
    filter_fn = build_filter_fn(SKIP_SUBSTRINGS)
    t0 = time.time()
    quantize_(
        diffusion_model,
        int4_weight_only(group_size=GROUP_SIZE, use_hqq=True),
        filter_fn=filter_fn,
    )
    torch.cuda.synchronize()
    print(f"   quantized in {time.time() - t0:.1f}s")
    print(f"   VRAM after quantization: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ---------- INT4 pass ----------
    print("\n" + "=" * 60)
    print(">> Running INT4 CubePart on int4 Cube3D input meshes")
    print("=" * 60)
    int4_outputs = {}
    torch.cuda.reset_peak_memory_stats()
    for name, cfg in BENCHMARK_SET.items():
        int4_path = cfg["int4_mesh"]
        if not Path(int4_path).exists():
            print(f"   [!] {name}: missing {int4_path}, skipping")
            continue
        print(f"\n>> INT4 CubePart on {name}")
        print(f"   input: {int4_path}")
        input_mesh = load_mesh_from_path(int4_path)
        print(f"   input mesh: {len(input_mesh.vertices)} verts, {len(input_mesh.faces)} faces")
        part_meshes, latency = generate_parts(pipe, input_mesh, cfg["parts"])
        saved = save_output_meshes(part_meshes, cfg["parts"], out_root / name, "int4")
        int4_outputs[name] = {"parts": part_meshes, "saved": saved, "latency": latency}
        print(f"   generated in {latency:.1f}s, saved to {out_root / name}")
    int4_peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n   INT4 peak VRAM: {int4_peak_vram:.2f} GB")

    # ---------- Chamfer distance per part ----------
    print("\n" + "=" * 60)
    print(">> Computing Chamfer distance per part (BF16 output vs INT4 output)")
    print("=" * 60)
    per_mesh_results = {}
    for name in BENCHMARK_SET:
        if name not in bf16_outputs or name not in int4_outputs:
            continue
        part_cds = []
        for i, (bf16_pm, int4_pm) in enumerate(zip(
                bf16_outputs[name]["parts"], int4_outputs[name]["parts"])):
            bf16_v, bf16_f = bf16_pm
            int4_v, int4_f = int4_pm
            if bf16_v is None or int4_v is None:
                part_cds.append(None)
                continue
            pts_a = trimesh.Trimesh(bf16_v, bf16_f).sample(5000)
            pts_b = trimesh.Trimesh(int4_v, int4_f).sample(5000)
            cd = chamfer_distance(np.asarray(pts_a), np.asarray(pts_b))
            part_cds.append(round(cd, 5))
            print(f"   {name} part {i} ({BENCHMARK_SET[name]['parts'][i]}): CD = {cd:.5f}")
        valid = [c for c in part_cds if c is not None]
        per_mesh_results[name] = {
            "parts": BENCHMARK_SET[name]["parts"],
            "chamfer_distance_per_part": part_cds,
            "chamfer_distance_mean": round(float(np.mean(valid)), 5) if valid else None,
            "bf16_latency_s": round(bf16_outputs[name]["latency"], 2),
            "int4_latency_s": round(int4_outputs[name]["latency"], 2),
        }

    # ---------- Summary ----------
    all_cds = [c for m in per_mesh_results.values()
                 for c in m["chamfer_distance_per_part"] if c is not None]
    summary = {
        "vram_gb": {
            "bf16_peak": round(bf16_peak_vram, 2),
            "int4_peak": round(int4_peak_vram, 2),
        },
        "chamfer_distance_median_across_all_parts": (
            round(float(np.median(all_cds)), 5) if all_cds else None),
        "chamfer_distance_mean_across_all_parts": (
            round(float(np.mean(all_cds)), 5) if all_cds else None),
        "per_mesh": per_mesh_results,
        "caveats": [
            "bf16_130 (yarn ball) is non-watertight; CubePart docs note that "
            "duplicated-shell meshes degrade quality regardless of quantization.",
            "Parts lists are best-guess from visual inspection; not Roblox-official.",
            "INT4 method is HQQ (not pure RTN) due to torchao bug in default RTN "
            "packing path across versions 0.10-0.13. Same W4A16 scheme.",
        ],
    }
    out_path = Path("results/spot_check_summary.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(">> Summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\n>> Wrote {out_path}")
    print(f">> Output meshes in {out_root}/ (bf16_*.glb and int4_*.glb per mesh)")


if __name__ == "__main__":
    main()
