"""
RTN W4A16 INT4 quantization of CubePart's diffusion_model (QwenImageTransformer2DModel),
matching the exact approach from TrNi/efficient-cube3d: torchao int4_weight_only,
group_size=128, no calibration data needed.
"""

import json
import time
from pathlib import Path

import torch
from torchao.quantization import quantize_, int4_weight_only
from torchao.quantization.quant_primitives import ZeroPointDomain
from cube_part.pipelines import PartShapeDenoiserPipeline

CONFIG_PATH = "configs/shape_denoiser_multimesh.yaml"
CHECKPOINT_PATH = "weights/multi_part_dit.safetensors"
VAE_CHECKPOINT_PATH = "weights/vae.safetensors"
GROUP_SIZE = 128
OUTPUT_PATH = "weights/multi_part_dit_rtn_int4_g128.pt"

SKIP_SUBSTRINGS = {
    "img_in", "txt_in", "proj_out", "pos_embed", "time_text_embed", "norm",
}


def build_filter_fn(skip_substrings):
    def _filter(module, fqn: str) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        return not any(s in fqn for s in skip_substrings)
    return _filter


def count_linear_params(module, filter_fn):
    total, quantized = 0, 0
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Linear):
            n = sum(p.numel() for p in m.parameters())
            total += n
            if filter_fn(m, name):
                quantized += n
    return total, quantized


def main():
    assert torch.cuda.is_available(), "Run on a GPU node."

    print(">> Loading BF16 CubePart pipeline")
    pipe = PartShapeDenoiserPipeline(
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        vae_checkpoint_path=VAE_CHECKPOINT_PATH,
        extract_geometry_fn_name="extract_geometry_coarse_to_fine",
    )

    diffusion_model = pipe.system.diffusion_model
    filter_fn = build_filter_fn(SKIP_SUBSTRINGS)

    total_params, quantized_params = count_linear_params(diffusion_model, filter_fn)
    print(f"   diffusion_model Linear params: {total_params/1e6:.1f}M total, "
          f"{quantized_params/1e6:.1f}M ({100*quantized_params/total_params:.1f}%) will be quantized")

    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated() / 1e9
    print(f"   VRAM before quantization: {vram_before:.2f} GB")

    print(f">> Applying RTN W4A16 quantization (group_size={GROUP_SIZE})")
    t0 = time.time()
    quantize_(
        diffusion_model,
        int4_weight_only(group_size=GROUP_SIZE, zero_point_domain=ZeroPointDomain.INT),
        filter_fn=filter_fn,
    )
    torch.cuda.synchronize()
    quant_time = time.time() - t0
    print(f"   quantization took {quant_time:.1f}s")

    vram_after = torch.cuda.memory_allocated() / 1e9
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"   VRAM after quantization: {vram_after:.2f} GB (peak during: {peak_vram:.2f} GB)")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pipe.system.state_dict(), OUTPUT_PATH)
    print(f">> Saved quantized state dict to {OUTPUT_PATH}")

    meta = {
        "method": "rtn",
        "scheme": "W4A16",
        "group_size": GROUP_SIZE,
        "zero_point_domain": "INT",
        "scope": "diffusion_model only (transformer_blocks + multi_transformer_blocks)",
        "skipped_substrings": sorted(SKIP_SUBSTRINGS),
        "linear_params_total_M": round(total_params / 1e6, 1),
        "linear_params_quantized_M": round(quantized_params / 1e6, 1),
        "quant_time_s": round(quant_time, 1),
        "vram_gb_before": round(vram_before, 2),
        "vram_gb_after": round(vram_after, 2),
        "vram_gb_peak_during_quant": round(peak_vram, 2),
    }
    meta_path = Path(OUTPUT_PATH).parent / "quant_config_cubepart.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f">> Wrote {meta_path}")
    print(json.dumps(meta, indent=2))

    return pipe


if __name__ == "__main__":
    main()
