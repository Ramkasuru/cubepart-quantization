"""
BF16 vs INT4 CubePart on all Cube3D baseline meshes (bf16_*.npz paired with rtn_int4_w4a16_*.npz).
Results are written per mesh, rerun to resume.
"""

import argparse
import json
import re
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
DEFAULT_PARTS = ["main body", "top", "bottom"]

SKIP_SUBSTRINGS = {"img_in", "txt_in", "proj_out", "pos_embed", "time_text_embed", "norm"}

BASELINE_DIR = Path("nisarg_shared/baseline_meshes")
INT4_DIR = Path("nisarg_shared/benchmark_results_rtn_int4")
PARTS_OVERRIDE = Path("nisarg_shared/parts_override.json")

RESULTS_DIR = Path("results/full_sweep")
PER_MESH_DIR = RESULTS_DIR / "per_mesh"
SUMMARY_PATH = RESULTS_DIR / "summary.json"


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


def load_mesh_from_path(path: Path) -> trimesh.Trimesh:
    if str(path).endswith(".npz"):
        d = np.load(str(path), allow_pickle=True)
        return trimesh.Trimesh(vertices=d["vertices"], faces=d["faces"])
    return trimesh.load(str(path), force="mesh")


def generate_parts(pipe, mesh: trimesh.Trimesh, part_names,
                    guidance_scale=7.5, num_inference_steps=50):
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


def discover_mesh_pairs():
    pattern = re.compile(r"bf16_(\d+)\.npz$")
    pairs = []
    for bf16_path in sorted(BASELINE_DIR.glob("bf16_*.npz")):
        m = pattern.search(bf16_path.name)
        if not m:
            continue
        idx = m.group(1)
        int4_path = INT4_DIR / f"rtn_int4_w4a16_{idx}.npz"
        if int4_path.exists():
            pairs.append((idx, bf16_path, int4_path))
    return pairs


def load_parts_override():
    if PARTS_OVERRIDE.exists():
        return json.loads(PARTS_OVERRIDE.read_text())
    return {}


def already_done(idx: str) -> bool:
    return (PER_MESH_DIR / f"{idx}.json").exists()


def save_per_mesh_result(idx: str, result: dict):
    PER_MESH_DIR.mkdir(parents=True, exist_ok=True)
    with open(PER_MESH_DIR / f"{idx}.json", "w") as f:
        json.dump(result, f, indent=2)


def load_all_per_mesh_results():
    if not PER_MESH_DIR.exists():
        return {}
    results = {}
    for p in sorted(PER_MESH_DIR.glob("*.json")):
        results[p.stem] = json.loads(p.read_text())
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "Run on a GPU node."

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PER_MESH_DIR.mkdir(parents=True, exist_ok=True)

    pairs = discover_mesh_pairs()
    if args.limit:
        pairs = pairs[:args.limit]
    parts_override = load_parts_override()

    print(f">> Discovered {len(pairs)} mesh pairs")
    if not args.redo:
        done_count = sum(1 for idx, _, _ in pairs if already_done(idx))
        print(f"   {done_count} already completed (skipping unless --redo passed)")

    print("\n>> Loading BF16 pipeline")
    pipe = PartShapeDenoiserPipeline(
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        vae_checkpoint_path=VAE_CHECKPOINT_PATH,
        extract_geometry_fn_name="extract_geometry_coarse_to_fine",
    )
    quantized = False
    filter_fn = build_filter_fn(SKIP_SUBSTRINGS)

    print("\n" + "=" * 60)
    print(f">> BF16 pass on {len(pairs)} meshes")
    print("=" * 60)
    bf16_results = {}
    for i, (idx, bf16_path, int4_path) in enumerate(pairs):
        if not args.redo and already_done(idx):
            continue
        parts = parts_override.get(idx, DEFAULT_PARTS)
        print(f"\n[{i+1}/{len(pairs)}] BF16 mesh {idx}: {bf16_path.name}, parts={parts}")
        try:
            mesh = load_mesh_from_path(bf16_path)
            part_meshes, latency = generate_parts(pipe, mesh, parts)
            bf16_results[idx] = {"parts": part_meshes, "latency": latency, "parts_list": parts}
            print(f"   done in {latency:.1f}s")
        except Exception as e:
            print(f"   [ERROR] {e}")
            save_per_mesh_result(idx, {"error": f"bf16 failed: {e}", "parts_list": parts})

    if not quantized and bf16_results:
        print("\n" + "=" * 60)
        print(">> Quantizing diffusion_model in-place (HQQ W4A16 g=128)")
        print("=" * 60)
        diffusion_model = pipe.system.diffusion_model
        diffusion_model = diffusion_model.to(torch.float32)
        t0 = time.time()
        quantize_(
            diffusion_model,
            int4_weight_only(group_size=GROUP_SIZE, use_hqq=True),
            filter_fn=filter_fn,
        )
        torch.cuda.synchronize()
        print(f"   done in {time.time() - t0:.1f}s")
        quantized = True

    print("\n" + "=" * 60)
    print(">> INT4 pass + Chamfer distance")
    print("=" * 60)
    for i, (idx, bf16_path, int4_path) in enumerate(pairs):
        if not args.redo and already_done(idx):
            continue
        if idx not in bf16_results:
            continue
        parts = bf16_results[idx]["parts_list"]
        print(f"\n[{i+1}/{len(pairs)}] INT4 mesh {idx}: {int4_path.name}")
        try:
            mesh = load_mesh_from_path(int4_path)
            int4_part_meshes, int4_latency = generate_parts(pipe, mesh, parts)
            print(f"   generated in {int4_latency:.1f}s, computing CD per part...")

            part_cds = []
            for j, (bf16_pm, int4_pm) in enumerate(zip(
                    bf16_results[idx]["parts"], int4_part_meshes)):
                bv, bf = bf16_pm
                iv, if_ = int4_pm
                if bv is None or iv is None:
                    part_cds.append(None)
                    continue
                pts_a = trimesh.Trimesh(bv, bf).sample(5000)
                pts_b = trimesh.Trimesh(iv, if_).sample(5000)
                cd = chamfer_distance(np.asarray(pts_a), np.asarray(pts_b))
                part_cds.append(round(cd, 5))

            valid_cds = [c for c in part_cds if c is not None]
            result = {
                "idx": idx,
                "parts_list": parts,
                "chamfer_distance_per_part": part_cds,
                "chamfer_distance_mean": (round(float(np.mean(valid_cds)), 5)
                                            if valid_cds else None),
                "bf16_latency_s": round(bf16_results[idx]["latency"], 2),
                "int4_latency_s": round(int4_latency, 2),
            }
            save_per_mesh_result(idx, result)
            print(f"   CD per part: {part_cds}")
        except Exception as e:
            print(f"   [ERROR] {e}")
            save_per_mesh_result(idx, {
                "error": f"int4 failed: {e}",
                "parts_list": parts,
                "bf16_latency_s": round(bf16_results[idx]["latency"], 2),
            })
        finally:
            bf16_results.pop(idx, None)

    print("\n" + "=" * 60)
    print(">> Aggregating results")
    print("=" * 60)
    all_results = load_all_per_mesh_results()
    all_cds = []
    for r in all_results.values():
        if "chamfer_distance_per_part" in r:
            all_cds.extend(c for c in r["chamfer_distance_per_part"] if c is not None)

    summary = {
        "n_meshes_attempted": len(all_results),
        "n_meshes_with_errors": sum(1 for r in all_results.values() if "error" in r),
        "chamfer_distance_median_across_all_parts": (
            round(float(np.median(all_cds)), 5) if all_cds else None),
        "chamfer_distance_mean_across_all_parts": (
            round(float(np.mean(all_cds)), 5) if all_cds else None),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n>> Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
