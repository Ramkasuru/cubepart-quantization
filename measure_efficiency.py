"""
Latency (mean over 10 meshes), setup time and peak VRAM for BF16 and INT4 in one run.

    python measure_efficiency.py

Takes about 15-25 min on one L40S.
"""
import json
import time
from pathlib import Path

import torch
from torchao.quantization import quantize_, int4_weight_only

from benchmark_4config_sweep import (
    BASELINE_DIR, CONFIG_PATH, CHECKPOINT_PATH, VAE_CHECKPOINT_PATH,
    GROUP_SIZE, SKIP_SUBSTRINGS, build_filter_fn, load_mesh_from_path, generate_parts,
)
from cube_part.pipelines import PartShapeDenoiserPipeline

# one mesh from each of 10 categories
LATENCY_TEST_IDX = ["130", "060", "020", "160", "240", "220", "080", "180", "260", "290"]


def gb(nbytes):
    return nbytes / (1024 ** 3)


def main():
    assert torch.cuda.is_available(), "Run on a GPU node."
    po = json.loads(Path("nisarg_shared/parts_override.json").read_text())
    test_idx = [i for i in LATENCY_TEST_IDX if i in po]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    pipe = PartShapeDenoiserPipeline(
        config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH,
        vae_checkpoint_path=VAE_CHECKPOINT_PATH,
        extract_geometry_fn_name="extract_geometry_coarse_to_fine",
    )
    bf16_load_s = time.time() - t0
    bf16_load_peak_gb = gb(torch.cuda.max_memory_allocated())
    print(f">> BF16 load/setup: {bf16_load_s:.1f}s, peak VRAM during load {bf16_load_peak_gb:.2f} GB")

    print(f"\n>> BF16 latency over {len(test_idx)} meshes:")
    torch.cuda.reset_peak_memory_stats()
    bf16_lat = []
    for idx in test_idx:
        mesh = load_mesh_from_path(BASELINE_DIR / f"bf16_{idx}.npz")
        _, lat = generate_parts(pipe, mesh, po[idx], seed=int(idx))
        bf16_lat.append(lat)
        print(f"   idx {idx}: {lat:.1f}s")
    bf16_peak_gb = gb(torch.cuda.max_memory_allocated())
    bf16_mean = sum(bf16_lat) / len(bf16_lat)
    print(f">> BF16 mean latency: {bf16_mean:.1f}s, peak VRAM during generation {bf16_peak_gb:.2f} GB")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    quantize_(pipe.system.diffusion_model, int4_weight_only(group_size=GROUP_SIZE, use_hqq=True),
               filter_fn=build_filter_fn(SKIP_SUBSTRINGS))
    torch.cuda.synchronize()
    quant_s = time.time() - t0
    print(f"\n>> Quantize (in-place, on top of already-loaded BF16): {quant_s:.1f}s")

    print(f"\n>> INT4 latency over {len(test_idx)} meshes:")
    torch.cuda.reset_peak_memory_stats()
    int4_lat = []
    for idx in test_idx:
        mesh = load_mesh_from_path(BASELINE_DIR / f"bf16_{idx}.npz")
        _, lat = generate_parts(pipe, mesh, po[idx], seed=int(idx))
        int4_lat.append(lat)
        print(f"   idx {idx}: {lat:.1f}s")
    int4_peak_gb = gb(torch.cuda.max_memory_allocated())
    int4_mean = sum(int4_lat) / len(int4_lat)
    print(f">> INT4 mean latency: {int4_mean:.1f}s, peak VRAM during generation {int4_peak_gb:.2f} GB")

    print("\n=== SUMMARY (copy into the report) ===")
    print(f"BF16 load/setup time: {bf16_load_s:.1f}s")
    print(f"BF16 mean per-inference latency (n={len(bf16_lat)}): {bf16_mean:.1f}s")
    print(f"BF16 peak VRAM (load or generate, whichever higher): {max(bf16_load_peak_gb, bf16_peak_gb):.2f} GB")
    print(f"Quantize time (on top of a loaded BF16 model): {quant_s:.1f}s")
    print(f"INT4 mean per-inference latency (n={len(int4_lat)}): {int4_mean:.1f}s")
    print(f"INT4 peak VRAM during generation: {int4_peak_gb:.2f} GB")
    print(f"\nINT4 setup time = BF16 load + quantize = {bf16_load_s + quant_s:.1f}s "
          f"(there is no direct INT4 load path yet).")
    print(f"INT4 peak VRAM {int4_peak_gb:.2f} GB was measured on L40S only, smaller GPUs were not tested.")


if __name__ == "__main__":
    main()
