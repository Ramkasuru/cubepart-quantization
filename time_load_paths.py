"""
Times repeated idx 130 generations for one load path (the safetensors verify took 188.8 s
in job 1215174 against the usual ~66 s for INT4). One process per source:

    python time_load_paths.py --source bf16 --repeat 3   # no quantization
    python time_load_paths.py --source pt   --repeat 3   # quantize_ + load_state_dict(.pt)
    python time_load_paths.py --source st   --repeat 3   # quantize_ + copy from .safetensors

INT4 outputs are checked against reference_parts.npz each time.
"""
import argparse

import numpy as np
import torch
from safetensors.torch import load_file

from benchmark_4config_sweep import (
    BASELINE_DIR, CONFIG_PATH, CHECKPOINT_PATH, VAE_CHECKPOINT_PATH, generate_parts, load_mesh_from_path,
)
from cube_part.pipelines import PartShapeDenoiserPipeline
from export_int4_checkpoint import CKPT_PATH, REF_PATH, TEST_IDX, build_and_quantize_pipe, load_test_parts
from export_int4_safetensors import ST_PATH, collect


def load(source):
    if source == "bf16":
        return PartShapeDenoiserPipeline(
            config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH,
            vae_checkpoint_path=VAE_CHECKPOINT_PATH,
            extract_geometry_fn_name="extract_geometry_coarse_to_fine",
        )
    pipe = build_and_quantize_pipe()
    model = pipe.system.diffusion_model
    if source == "pt":
        model.load_state_dict(torch.load(CKPT_PATH, weights_only=False, map_location="cuda"))
    else:
        saved = load_file(str(ST_PATH), device="cuda")
        flat = collect(model)
        assert set(flat) == set(saved), "key mismatch between model and safetensors file"
        with torch.no_grad():
            for k, t in flat.items():
                t.copy_(saved[k])
    torch.cuda.synchronize()
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bf16", "pt", "st"], required=True)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "Run on a GPU node."

    pipe = load(args.source)
    parts = load_test_parts()
    mesh = load_mesh_from_path(BASELINE_DIR / f"bf16_{TEST_IDX}.npz")
    ref = np.load(REF_PATH)

    for r in range(1, args.repeat + 1):
        part_meshes, latency = generate_parts(pipe, mesh, parts, seed=int(TEST_IDX))
        if args.source == "bf16":
            status = "n/a (BF16, no INT4 reference)"
        else:
            ok = all(np.array_equal(v, ref[f"v{i}"]) and np.array_equal(f, ref[f"f{i}"])
                     for i, (v, f) in enumerate(part_meshes))
            status = "MATCH" if ok else "MISMATCH"
        print(f"TIMING source={args.source} gen={r} latency={latency:.1f}s output={status}", flush=True)


if __name__ == "__main__":
    main()
