"""
Answers "what will it take to ship CubePart INT4 weights" (Nisarg's ask,
2026-09-19 10:30 PM). The project already hit this once and worked around it
rather than solving it: Errors_and_Fixes.md "AffineQuantizedTensor can't
survive cpu<->cuda .to()" -- reloading a saved INT4 checkpoint
(multi_part_dit_rtn_int4_g128.pt) and moving device broke the quantized
tensors, so benchmark_4config_sweep.py has never saved/reloaded across
process boundaries -- it only ever quantizes in-place and keeps using the
live model in the same process.

Nisarg's own Cube3D INT4 checkpoint (shape_gpt_rtn_int4_g128.pt) IS
downloadable and loadable fresh from HF, so SOME save/load pattern works for
an AffineQuantizedTensor model -- we just haven't found the right one for
CubePart. This script tries 3 candidate patterns against a tiny slice of the
real model (not the full 8.58 GB DiT, to keep iteration fast) and checks
whether a FRESH PROCESS reload reproduces the live quantized output exactly.

Run from ~/cube/cubepart, same env as the sweep:
    python test_int4_checkpoint_save.py --write   # do the quantize + 3 save attempts
    python test_int4_checkpoint_save.py --read     # NEW PROCESS: try loading each, compare
"""
import argparse
from pathlib import Path

import torch
from torchao.quantization import quantize_, int4_weight_only

OUT_DIR = Path("results/checkpoint_test")
GROUP_SIZE = 128


def toy_linear():
    # stand-in for one transformer_blocks Linear layer -- same dtype/shape
    # class as the real model, small enough to save/load in seconds while we
    # figure out which serialization pattern survives
    torch.manual_seed(0)
    m = torch.nn.Sequential(
        torch.nn.Linear(2048, 2048, bias=False),
        torch.nn.Linear(2048, 2048, bias=False),
    ).to("cuda").to(torch.bfloat16)
    return m


def write():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = toy_linear()
    quantize_(m, int4_weight_only(group_size=GROUP_SIZE, use_hqq=True))
    torch.cuda.synchronize()

    x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ref_out = m(x).float().cpu()
    torch.save(x.cpu(), OUT_DIR / "probe_input.pt")
    torch.save(ref_out, OUT_DIR / "reference_output.pt")

    # Pattern A: save state_dict while model is still on cuda (no .to() before save)
    torch.save(m.state_dict(), OUT_DIR / "pattern_a_state_dict.pt")

    # Pattern B: save the whole module object (pickle), not just state_dict --
    # preserves the AffineQuantizedTensor subclass instances directly rather
    # than round-tripping through load_state_dict's tensor-copy path
    torch.save(m, OUT_DIR / "pattern_b_full_module.pt")

    # Pattern C: save state_dict, but also save the architecture separately so
    # on load we can quantize_() an identical fresh module FIRST (so its
    # Linear layers are already AffineQuantizedTensor-shaped) and only then
    # load_state_dict(..., assign=True) -- assign=True replaces parameters
    # instead of copying into existing storage, which plain tensor subclasses
    # often need.
    torch.save(m.state_dict(), OUT_DIR / "pattern_c_state_dict.pt")

    print("Wrote probe_input.pt, reference_output.pt, and 3 candidate checkpoints to", OUT_DIR)
    print("Now run in a FRESH process: python test_int4_checkpoint_save.py --read")


def compare(name, out, ref):
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    max_diff = (out - ref).abs().max().item()
    print(f"  {name}: {'MATCH' if ok else 'MISMATCH'}  (max abs diff {max_diff:.6f})")
    return ok


def read():
    x = torch.load(OUT_DIR / "probe_input.pt").to("cuda")
    ref = torch.load(OUT_DIR / "reference_output.pt")
    results = {}

    # --- Pattern A: plain state_dict load into a freshly quantized module ---
    try:
        m = toy_linear()
        quantize_(m, int4_weight_only(group_size=GROUP_SIZE, use_hqq=True))
        sd = torch.load(OUT_DIR / "pattern_a_state_dict.pt", weights_only=False, map_location="cuda")
        m.load_state_dict(sd)
        with torch.no_grad():
            out = m(x).float().cpu()
        results["A: state_dict -> load_state_dict (default)"] = compare("A", out, ref)
    except Exception as e:
        print(f"  A: FAILED to even load -- {e}")
        results["A"] = False

    # --- Pattern B: whole-module pickle, loaded directly onto cuda ---
    try:
        m2 = torch.load(OUT_DIR / "pattern_b_full_module.pt", weights_only=False, map_location="cuda")
        with torch.no_grad():
            out2 = m2(x).float().cpu()
        results["B: torch.save(module) -> torch.load(map_location=cuda)"] = compare("B", out2, ref)
    except Exception as e:
        print(f"  B: FAILED -- {e}")
        results["B"] = False

    # --- Pattern C: quantize a fresh module first, then load_state_dict(assign=True) ---
    try:
        m3 = toy_linear()
        quantize_(m3, int4_weight_only(group_size=GROUP_SIZE, use_hqq=True))
        sd3 = torch.load(OUT_DIR / "pattern_c_state_dict.pt", weights_only=False, map_location="cuda")
        m3.load_state_dict(sd3, assign=True)
        with torch.no_grad():
            out3 = m3(x).float().cpu()
        results["C: quantize-then-load_state_dict(assign=True)"] = compare("C", out3, ref)
    except Exception as e:
        print(f"  C: FAILED -- {e}")
        results["C"] = False

    print("\nSummary:")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    working = [k for k, v in results.items() if v]
    if working:
        print(f"\n>> Use pattern: {working[0]}")
        print(">> Next: repeat with the REAL diffusion_model (not this toy Linear stack)")
        print(">> to confirm the pattern holds at full scale before shipping the real checkpoint.")
    else:
        print("\n>> None of the 3 patterns round-tripped cleanly. Check torchao version "
              "(`pip show torchao`) for a documented serialization API -- newer torchao "
              "releases added explicit quantized-checkpoint save/load helpers that may "
              "supersede all three patterns tried here.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--read", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "Run on a GPU node."
    if args.write:
        write()
    elif args.read:
        read()
    else:
        raise SystemExit("pass --write first, then --read in a fresh process")
