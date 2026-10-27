"""
4-config CubePart quantization benchmark (A/B/C/D), extending
benchmark_full_sweep.py's proven 2-config (bf16 vs int4) sweep.

  Config | Cube3D input (npz)     | CubePart model | Label
  -------|------------------------|----------------|----------------------------
    A    | bf16_*.npz             | BF16           | baseline
    B    | bf16_*.npz             | INT4           | CubePart quant, clean input
    C    | rtn_int4_w4a16_*.npz   | BF16           | baseline on quantized input
    D    | rtn_int4_w4a16_*.npz   | INT4           | CubePart quant, quant input

CD(A,B) isolates CubePart's own quantization effect on a clean Cube3D input.
CD(C,D) isolates the same effect on an already-quantized Cube3D input.

Run order quantizes diffusion_model exactly once, same in-place pattern as
benchmark_full_sweep.py (AffineQuantizedTensor doesn't survive a cpu<->cuda
round trip, so every INT4 config must run in this same process, after
quantization, before exit):
  1. Load BF16 pipeline.
  2. For every mesh: generate config A (bf16 input), holding part meshes in
     memory; generate config C (int4 input), holding part meshes in memory.
  3. Quantize diffusion_model in place (HQQ W4A16, g=128 -- identical filter
     and call to benchmark_full_sweep.py / quantize_cubepart_rtn.py).
  4. For every mesh with a stored A result: generate config B, compute
     CD(A,B) immediately, write it out, drop the stored A result.
  5. For every mesh with a stored C result: generate config D, compute
     CD(C,D) immediately, write it out, drop the stored C result.

Resumability is per (mesh, pair) rather than per mesh, since AB and CD are
independent passes that can fail or be interrupted separately:
  results/full_sweep_4config/per_mesh/<idx>_ab.json
  results/full_sweep_4config/per_mesh/<idx>_cd.json
A mesh's A/C generation is skipped only once BOTH files exist (nothing left
to compute); otherwise whichever of A or C is still needed gets regenerated.

Finer-grained checkpointing: every completed A/C generation is ALSO cached
to disk immediately (results/full_sweep_4config/cache/<idx>_a.pkl /
<idx>_c.pkl), not just held in memory. Without this, a kill during the B/D
pass (e.g. hitting the SLURM time limit) would silently discard every
already-completed A/C generation that the B/D pass hadn't reached yet --
on a large sweep that's potentially hours of GPU work redone for nothing.
With the cache, a kill at any point loses at most the one generation
in flight, since a resumed run loads any pending A/C work straight from
disk instead of regenerating it. Cache files are deleted once their final
<idx>_ab.json / <idx>_cd.json is written -- they're pure intermediate state.

--idx-min/--idx-max select an index band (e.g. 120-139 for furniture) since
discover_mesh_pairs() has no other way to target one category; --limit still
works as a plain head-of-list cap for quick smoke runs.
"""

import argparse
import json
import pickle
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

RESULTS_DIR = Path("results/full_sweep_4config")
PER_MESH_DIR = RESULTS_DIR / "per_mesh"
CACHE_DIR = RESULTS_DIR / "cache"
SUMMARY_PATH = RESULTS_DIR / "summary.json"


def build_filter_fn(skip_substrings):
    def _filter(module, fqn: str) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        return not any(s in fqn for s in skip_substrings)
    return _filter


def chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    # True average of forward and backward directional means, per Nisarg's
    # explicit spec ("each chamfer distance should be an average of forward
    # and backward computation"). benchmark_full_sweep.py computed a SUM
    # (d_ab.mean() + d_ba.mean(), no /2) -- kept here originally to match
    # already-approved smoke-test numbers, but Nisarg's own verbatim
    # instruction takes priority now that furniture-run numbers are being
    # reviewed for real. See Decisions.md -- any pre-existing per-mesh JSON
    # computed under the old sum convention needs its CD values halved
    # (exact, lossless) rather than being regenerated.
    tree_a, tree_b = cKDTree(pts_a), cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)
    d_ba, _ = tree_a.query(pts_b, k=1)
    return float((d_ab.mean() + d_ba.mean()) / 2.0)


def load_mesh_from_path(path: Path) -> trimesh.Trimesh:
    if str(path).endswith(".npz"):
        d = np.load(str(path), allow_pickle=True)
        return trimesh.Trimesh(vertices=d["vertices"], faces=d["faces"])
    return trimesh.load(str(path), force="mesh")


def generate_parts(pipe, mesh: trimesh.Trimesh, part_names, seed: int = 0,
                    guidance_scale=7.5, num_inference_steps=50):
    # sample_surface() draws its random subsample via trimesh's Trimesh.sample(),
    # which uses numpy's global RNG and takes no seed of its own. Without seeding
    # it here, two calls on the identical mesh get different input points and
    # therefore different output geometry even though input_to_part_shape's own
    # seed=0 default makes the diffusion step itself deterministic. Confirmed
    # empirically: CD(A,A) (same model, same input, run twice) was ~0.017-0.021,
    # the same order of magnitude as the CD(A,B) "quantization effect" we were
    # trying to measure -- i.e. pure noise, not signal. See Decisions.md.
    np.random.seed(seed)
    surface = sample_surface(mesh, num_samples=128_000)
    surface = torch.from_numpy(surface).to(pipe.device).unsqueeze(0).float()
    latents, _ = pipe.encode_shape(surface)
    t0 = time.time()
    part_meshes = pipe.input_to_part_shape(
        ShapeInput(prompt=[part_names], latents=latents),
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    latency = time.time() - t0
    return part_meshes, latency


def discover_mesh_pairs(idx_min=None, idx_max=None):
    pattern = re.compile(r"bf16_(\d+)\.npz$")
    pairs = []
    for bf16_path in sorted(BASELINE_DIR.glob("bf16_*.npz")):
        m = pattern.search(bf16_path.name)
        if not m:
            continue
        idx = m.group(1)
        if idx_min is not None and int(idx) < idx_min:
            continue
        if idx_max is not None and int(idx) > idx_max:
            continue
        int4_path = INT4_DIR / f"rtn_int4_w4a16_{idx}.npz"
        if int4_path.exists():
            pairs.append((idx, bf16_path, int4_path))
    return pairs


def load_parts_override():
    if PARTS_OVERRIDE.exists():
        return json.loads(PARTS_OVERRIDE.read_text())
    return {}


def ab_path(idx: str) -> Path:
    return PER_MESH_DIR / f"{idx}_ab.json"


def cd_path(idx: str) -> Path:
    return PER_MESH_DIR / f"{idx}_cd.json"


def save_json(path: Path, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def cache_path(idx: str, letter: str) -> Path:
    return CACHE_DIR / f"{idx}_{letter}.pkl"


def save_gen_cache(idx: str, letter: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path(idx, letter), "wb") as f:
        pickle.dump(result, f)


def load_gen_cache(idx: str, letter: str):
    path = cache_path(idx, letter)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def delete_gen_cache(idx: str, letter: str):
    cache_path(idx, letter).unlink(missing_ok=True)


def part_pair_cds(meshes_x, meshes_y, seed: int):
    # Reset the RNG before sampling each mesh so identical input geometry
    # (e.g. a CD(A,A) sanity check) produces identical point clouds and a
    # true CD of 0 -- otherwise trimesh's unseeded .sample() adds its own
    # Monte Carlo noise on top of any real quantization-induced difference.
    # Confirmed empirically 2026-08-31: generation itself was already fully
    # deterministic (bit-identical vertices/faces across repeat runs with a
    # fixed seed); this was the last unseeded sampling call in the pipeline.
    cds = []
    for (vx, fx), (vy, fy) in zip(meshes_x, meshes_y):
        if vx is None or vy is None:
            cds.append(None)
            continue
        np.random.seed(seed)
        pts_a = trimesh.Trimesh(vx, fx).sample(5000)
        np.random.seed(seed)
        pts_b = trimesh.Trimesh(vy, fy).sample(5000)
        cds.append(round(chamfer_distance(np.asarray(pts_a), np.asarray(pts_b)), 5))
    return cds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap on number of meshes after idx filtering")
    ap.add_argument("--idx-min", type=int, default=None, help="e.g. 120 for furniture")
    ap.add_argument("--idx-max", type=int, default=None, help="e.g. 139 for furniture")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "Run on a GPU node."

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PER_MESH_DIR.mkdir(parents=True, exist_ok=True)

    parts_override = load_parts_override()
    all_pairs = discover_mesh_pairs(args.idx_min, args.idx_max)
    pairs = [p for p in all_pairs if p[0] in parts_override]
    skipped = len(all_pairs) - len(pairs)
    if args.limit:
        pairs = pairs[:args.limit]

    print(f">> Discovered {len(pairs)} mesh pairs with parts_override entries "
          f"(idx range {args.idx_min}-{args.idx_max}); skipped {skipped} with no "
          f"override entry (excluded categories)")

    print("\n>> Loading BF16 pipeline")
    pipe = PartShapeDenoiserPipeline(
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        vae_checkpoint_path=VAE_CHECKPOINT_PATH,
        extract_geometry_fn_name="extract_geometry_coarse_to_fine",
    )
    filter_fn = build_filter_fn(SKIP_SUBSTRINGS)

    print("\n" + "=" * 60)
    print(f">> BF16 pass: config A (bf16 input) + config C (int4 input) on {len(pairs)} meshes")
    print("=" * 60)
    a_results, c_results = {}, {}
    for i, (idx, bf16_path, int4_path) in enumerate(pairs):
        need_a = args.redo or not ab_path(idx).exists()
        need_c = args.redo or not cd_path(idx).exists()
        if not need_a and not need_c:
            continue
        parts = parts_override.get(idx, DEFAULT_PARTS)

        seed = int(idx)  # same seed reused for A/B (bf16 mesh) and C/D (int4 mesh)
                         # so the only difference between paired runs is model precision

        if need_a:
            cached = None if args.redo else load_gen_cache(idx, "a")
            if cached is not None:
                a_results[idx] = cached
                print(f"[{i+1}/{len(pairs)}] A: mesh {idx} loaded from cache (resumed run)")
            else:
                print(f"\n[{i+1}/{len(pairs)}] A: bf16 input, mesh {idx}, parts={parts}")
                try:
                    mesh = load_mesh_from_path(bf16_path)
                    part_meshes, latency = generate_parts(pipe, mesh, parts, seed=seed)
                    result = {"parts": part_meshes, "latency": latency, "parts_list": parts, "seed": seed}
                    a_results[idx] = result
                    save_gen_cache(idx, "a", result)
                    print(f"   done in {latency:.1f}s")
                except Exception as e:
                    print(f"   [ERROR] {e}")
                    save_json(ab_path(idx), {"error": f"config A failed: {e}", "parts_list": parts})

        if need_c:
            cached = None if args.redo else load_gen_cache(idx, "c")
            if cached is not None:
                c_results[idx] = cached
                print(f"[{i+1}/{len(pairs)}] C: mesh {idx} loaded from cache (resumed run)")
            else:
                print(f"[{i+1}/{len(pairs)}] C: int4 input, mesh {idx}")
                try:
                    mesh = load_mesh_from_path(int4_path)
                    part_meshes, latency = generate_parts(pipe, mesh, parts, seed=seed)
                    result = {"parts": part_meshes, "latency": latency, "parts_list": parts, "seed": seed}
                    c_results[idx] = result
                    save_gen_cache(idx, "c", result)
                    print(f"   done in {latency:.1f}s")
                except Exception as e:
                    print(f"   [ERROR] {e}")
                    save_json(cd_path(idx), {"error": f"config C failed: {e}", "parts_list": parts})

    if a_results or c_results:
        print("\n" + "=" * 60)
        print(">> Quantizing diffusion_model in-place (HQQ W4A16 g=128)")
        print("=" * 60)
        diffusion_model = pipe.system.diffusion_model
        t0 = time.time()
        quantize_(
            diffusion_model,
            int4_weight_only(group_size=GROUP_SIZE, use_hqq=True),
            filter_fn=filter_fn,
        )
        torch.cuda.synchronize()
        print(f"   done in {time.time() - t0:.1f}s")

    print("\n" + "=" * 60)
    print(">> INT4 pass: config B (bf16 input) + CD(A,B)")
    print("=" * 60)
    for i, (idx, bf16_path, int4_path) in enumerate(pairs):
        if idx not in a_results:
            continue
        parts = a_results[idx]["parts_list"]
        print(f"\n[{i+1}/{len(pairs)}] B: bf16 input, mesh {idx}")
        try:
            mesh = load_mesh_from_path(bf16_path)
            b_parts, b_latency = generate_parts(pipe, mesh, parts, seed=int(idx))
            print(f"   generated in {b_latency:.1f}s, computing CD(A,B) per part...")
            cds = part_pair_cds(a_results[idx]["parts"], b_parts, seed=int(idx))
            valid = [c for c in cds if c is not None]
            save_json(ab_path(idx), {
                "idx": idx,
                "pair": "CD(A,B)",
                "parts_list": parts,
                "chamfer_distance_per_part": cds,
                "chamfer_distance_mean": round(float(np.mean(valid)), 5) if valid else None,
                "a_latency_s": round(a_results[idx]["latency"], 2),
                "b_latency_s": round(b_latency, 2),
            })
            print(f"   CD(A,B) per part: {cds}")
        except Exception as e:
            print(f"   [ERROR] {e}")
            save_json(ab_path(idx), {"error": f"config B failed: {e}", "parts_list": parts})
        finally:
            a_results.pop(idx, None)
            delete_gen_cache(idx, "a")

    print("\n" + "=" * 60)
    print(">> INT4 pass: config D (int4 input) + CD(C,D)")
    print("=" * 60)
    for i, (idx, bf16_path, int4_path) in enumerate(pairs):
        if idx not in c_results:
            continue
        parts = c_results[idx]["parts_list"]
        print(f"\n[{i+1}/{len(pairs)}] D: int4 input, mesh {idx}")
        try:
            mesh = load_mesh_from_path(int4_path)
            d_parts, d_latency = generate_parts(pipe, mesh, parts, seed=int(idx))
            print(f"   generated in {d_latency:.1f}s, computing CD(C,D) per part...")
            cds = part_pair_cds(c_results[idx]["parts"], d_parts, seed=int(idx))
            valid = [c for c in cds if c is not None]
            save_json(cd_path(idx), {
                "idx": idx,
                "pair": "CD(C,D)",
                "parts_list": parts,
                "chamfer_distance_per_part": cds,
                "chamfer_distance_mean": round(float(np.mean(valid)), 5) if valid else None,
                "c_latency_s": round(c_results[idx]["latency"], 2),
                "d_latency_s": round(d_latency, 2),
            })
            print(f"   CD(C,D) per part: {cds}")
        except Exception as e:
            print(f"   [ERROR] {e}")
            save_json(cd_path(idx), {"error": f"config D failed: {e}", "parts_list": parts})
        finally:
            c_results.pop(idx, None)
            delete_gen_cache(idx, "c")

    print(f"\n>> Done. Per-mesh results in {PER_MESH_DIR}/. Run compute_4config_cd_stats.py to aggregate.")


if __name__ == "__main__":
    main()
