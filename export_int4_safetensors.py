"""
Converts the INT4 checkpoint to .safetensors and checks the round trip.
safetensors only stores plain tensors, so each torchao INT4 tensor is flattened with
__tensor_flatten__ and saved under keys like "<param>::tensor_impl::packed_weight".
To load, run quantize_() and copy the saved tensors in.
The verify step zeroes every tensor before the copy and compares a real generation bit for bit.

Needs multi_part_dit_int4_hqq_g128.pt and reference_parts.npz from export_int4_checkpoint.py --write.

    python export_int4_safetensors.py --write
    python export_int4_safetensors.py --verify     # new process
"""
import argparse

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from benchmark_4config_sweep import BASELINE_DIR, generate_parts, load_mesh_from_path
from export_int4_checkpoint import (
    CKPT_PATH, OUT_DIR, REF_PATH, TEST_IDX, build_and_quantize_pipe, load_test_parts,
)

ST_PATH = OUT_DIR / "multi_part_dit_int4_hqq_g128.safetensors"
OK_DTYPES = {torch.bfloat16, torch.float16, torch.float32, torch.int32, torch.int8,
             torch.uint8, torch.int64, torch.bool}


def leaves(prefix, t, out):
    if hasattr(t, "__tensor_flatten__"):
        names, _ = t.__tensor_flatten__()
        for n in names:
            leaves(f"{prefix}::{n}", getattr(t, n), out)
    else:
        out[prefix] = t


def collect(model):
    flat = {}
    for k, v in model.state_dict(keep_vars=True).items():
        leaves(k, v, flat)
    return flat


def write():
    pipe = build_and_quantize_pipe()
    model = pipe.system.diffusion_model
    sd = torch.load(CKPT_PATH, weights_only=False, map_location="cuda")
    model.load_state_dict(sd)

    flat = collect(model)
    bad = {k: str(t.dtype) for k, t in flat.items() if t.dtype not in OK_DTYPES}
    if bad:
        raise SystemExit(f"unsupported dtypes for safetensors: {dict(list(bad.items())[:10])}")

    tensors = {k: t.detach().cpu().contiguous() for k, t in flat.items()}
    save_file(tensors, str(ST_PATH), metadata={"format": "pt", "torchao": "0.11.0",
                                               "recipe": "quantize_ then copy leaves"})
    n_bytes = ST_PATH.stat().st_size
    print(f">> Wrote {ST_PATH}: {len(tensors)} tensors, {n_bytes:,} bytes "
          f"({n_bytes / 1e9:.3f} GB, {n_bytes / 2**30:.3f} GiB)")
    print("Now run in a FRESH process: python export_int4_safetensors.py --verify")


def verify():
    assert ST_PATH.exists(), f"{ST_PATH} missing, run --write first"
    pipe = build_and_quantize_pipe()
    model = pipe.system.diffusion_model
    saved = load_file(str(ST_PATH), device="cuda")

    flat = collect(model)
    missing = sorted(set(flat) - set(saved))
    extra = sorted(set(saved) - set(flat))
    assert not missing and not extra, f"key mismatch: missing={missing[:5]} extra={extra[:5]}"

    with torch.no_grad():
        for k, t in flat.items():
            s = saved[k]
            assert s.shape == t.shape and s.dtype == t.dtype, f"{k}: {s.shape}/{s.dtype} vs {t.shape}/{t.dtype}"
            t.zero_()      # zero the fresh weights so a pass comes from the file
            t.copy_(s)
    torch.cuda.synchronize()
    print(f">> Loaded {len(flat)} tensors from {ST_PATH.name}")

    parts = load_test_parts()
    mesh = load_mesh_from_path(BASELINE_DIR / f"bf16_{TEST_IDX}.npz")
    part_meshes, latency = generate_parts(pipe, mesh, parts, seed=int(TEST_IDX))
    print(f">> Generation done in {latency:.1f}s")

    ref = np.load(REF_PATH)
    assert int(ref["n_parts"]) == len(part_meshes)
    ok = True
    for i, (v, f) in enumerate(part_meshes):
        v_ok, f_ok = np.array_equal(v, ref[f"v{i}"]), np.array_equal(f, ref[f"f{i}"])
        print(f"  part {i} ({parts[i]!r}): vertices {'MATCH' if v_ok else 'MISMATCH'}, "
              f"faces {'MATCH' if f_ok else 'MISMATCH'}")
        ok = ok and v_ok and f_ok
    print(f"\n{'PASS' if ok else 'FAIL'}: safetensors {'reproduces' if ok else 'does NOT reproduce'} "
          f"the live quantized output bit for bit.")


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
