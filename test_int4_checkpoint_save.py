"""
Tries 3 save/reload patterns for torchao INT4 weights on a small Linear layer and checks
that a reload in a new process gives the same output.

    python test_int4_checkpoint_save.py --write   # quantize and save with the 3 patterns
    python test_int4_checkpoint_save.py --read    # new process: load each and compare
"""
import argparse
from pathlib import Path

import torch
from torchao.quantization import quantize_, int4_weight_only

OUT_DIR = Path("results/checkpoint_test")
GROUP_SIZE = 128


def toy_linear():
    # stand-in for one transformer_blocks Linear, same dtype
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

    # Pattern B: pickle the whole module
    torch.save(m, OUT_DIR / "pattern_b_full_module.pt")

    # Pattern C: quantize a fresh module first, then load_state_dict(assign=True)
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
