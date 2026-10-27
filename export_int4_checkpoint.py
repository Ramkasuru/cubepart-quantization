"""
Full-scale version of test_int4_checkpoint_save.py's confirmed recipe (Pattern
A: quantize a fresh module first, then plain load_state_dict()) applied to
the REAL diffusion_model, not the toy stand-in. Produces the actual shippable
checkpoint file + its disk size (fills in the "[ pending ]" cell in
HF_Page_Draft_CubePart_INT4.md), and proves round-trip correctness by
comparing a real generate_parts() output, not just tensor values.

    python export_int4_checkpoint.py --write     # quantize, save checkpoint, save reference output
    python export_int4_checkpoint.py --verify    # NEW PROCESS: fresh-load checkpoint, compare
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchao.quantization import quantize_, int4_weight_only

from benchmark_4config_sweep import (
    BASELINE_DIR, CONFIG_PATH, CHECKPOINT_PATH, VAE_CHECKPOINT_PATH,
    GROUP_SIZE, SKIP_SUBSTRINGS, build_filter_fn, load_mesh_from_path, generate_parts,
)
from cube_part.pipelines import PartShapeDenoiserPipeline

OUT_DIR = Path("results/checkpoint_export")
CKPT_PATH = OUT_DIR / "multi_part_dit_int4_hqq_g128.pt"
REF_PATH = OUT_DIR / "reference_parts.npz"

TEST_IDX = "130"  # bar stool -- well-characterized, small part count, fast


def load_test_parts():
    po = json.loads(Path("nisarg_shared/parts_override.json").read_text())
    return po[TEST_IDX]


def build_and_quantize_pipe():
    pipe = PartShapeDenoiserPipeline(
        config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH,
        vae_checkpoint_path=VAE_CHECKPOINT_PATH,
        extract_geometry_fn_name="extract_geometry_coarse_to_fine",
    )
    quantize_(pipe.system.diffusion_model, int4_weight_only(group_size=GROUP_SIZE, use_hqq=True),
               filter_fn=build_filter_fn(SKIP_SUBSTRINGS))
    torch.cuda.synchronize()
    return pipe


def write():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = build_and_quantize_pipe()

    # save the checkpoint -- state_dict off a CUDA-resident model, no .to() first
    torch.save(pipe.system.diffusion_model.state_dict(), CKPT_PATH)
    size_gb = CKPT_PATH.stat().st_size / (1024 ** 3)
    print(f">> Saved {CKPT_PATH} ({size_gb:.3f} GB) <-- this is the HF page's real 'model size on disk' number")

    parts = load_test_parts()
    mesh = load_mesh_from_path(BASELINE_DIR / f"bf16_{TEST_IDX}.npz")
    part_meshes, latency = generate_parts(pipe, mesh, parts, seed=int(TEST_IDX))
    print(f">> Reference generation (idx {TEST_IDX}, parts={parts}) done in {latency:.1f}s")

    np.savez(REF_PATH,
             n_parts=len(part_meshes),
             **{f"v{i}": v for i, (v, f) in enumerate(part_meshes)},
             **{f"f{i}": f for i, (v, f) in enumerate(part_meshes)})
    print(f">> Wrote reference output to {REF_PATH}")
    print("\nNow run in a FRESH process: python export_int4_checkpoint.py --verify")


def verify():
    assert CKPT_PATH.exists(), f"{CKPT_PATH} missing -- run --write first"
    size_gb = CKPT_PATH.stat().st_size / (1024 ** 3)

    pipe = PartShapeDenoiserPipeline(
        config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH,
        vae_checkpoint_path=VAE_CHECKPOINT_PATH,
        extract_geometry_fn_name="extract_geometry_coarse_to_fine",
    )
    # Pattern A: quantize FIRST (establishes AffineQuantizedTensor structure),
    # then plain load_state_dict (no assign=True) to overwrite with the saved values
    quantize_(pipe.system.diffusion_model, int4_weight_only(group_size=GROUP_SIZE, use_hqq=True),
               filter_fn=build_filter_fn(SKIP_SUBSTRINGS))
    sd = torch.load(CKPT_PATH, weights_only=False, map_location="cuda")
    pipe.system.diffusion_model.load_state_dict(sd)
    torch.cuda.synchronize()
    print(f">> Loaded checkpoint fresh ({size_gb:.3f} GB) via quantize-then-load_state_dict")

    parts = load_test_parts()
    mesh = load_mesh_from_path(BASELINE_DIR / f"bf16_{TEST_IDX}.npz")
    part_meshes, latency = generate_parts(pipe, mesh, parts, seed=int(TEST_IDX))
    print(f">> Reloaded-model generation done in {latency:.1f}s")

    ref = np.load(REF_PATH)
    n = int(ref["n_parts"])
    assert n == len(part_meshes), f"part count mismatch: ref={n} reloaded={len(part_meshes)}"
    all_match = True
    for i, (v, f) in enumerate(part_meshes):
        v_ok = np.array_equal(v, ref[f"v{i}"])
        f_ok = np.array_equal(f, ref[f"f{i}"])
        print(f"  part {i} ({parts[i]!r}): vertices {'MATCH' if v_ok else 'MISMATCH'}, "
              f"faces {'MATCH' if f_ok else 'MISMATCH'}")
        all_match = all_match and v_ok and f_ok

    print(f"\n{'PASS' if all_match else 'FAIL'} -- reloaded checkpoint {'reproduces' if all_match else 'does NOT reproduce'} "
          f"the live-quantized model's output bit-for-bit.")
    if all_match:
        print(f">> Checkpoint is shippable: {CKPT_PATH} ({size_gb:.3f} GB)")
    else:
        print(">> Do not ship yet -- investigate before uploading to HF.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "Run on a GPU node."
    if args.write:
        write()
    elif args.verify:
        verify()
    else:
        raise SystemExit("pass --write first, then --verify in a fresh process")
