# CubePart INT4 Quantization

## Scripts
- `quantize_cubepart_rtn.py` — quantizes CubePart's `diffusion_model` in-place using `torchao.int4_weight_only(use_hqq=True, group_size=128)`. Uses HQQ instead of pure RTN due to a torchao dtype bug in the default `TensorCoreTiledLayout` packing path (reproduced across versions 0.10-0.13). Same W4A16 scheme.
- `benchmark_spot_check.py` — runs BF16 vs INT4 CubePart on 2 baseline meshes (hat, yarn ball), computes per-part Chamfer distance.

## Results (spot check, 2 meshes)
See `spot_check_summary.json` for numeric results, `spot_check/` for output `.glb` files per part.

Key numbers:
- BF16 peak VRAM: 28.33 GB
- INT4 peak VRAM: 21.23 GB (inflated by residual BF16 allocator; INT4-only from separate run was 11.68 GB)
- CD median across all parts: 0.539
- Latency: BF16 ~48-57s per mesh, INT4 ~64-70s per mesh

## Environment notes
- torch 2.8.0+cu128, torchao 0.11.0
- L40S GPU (44GB), 2× H100 also available
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` required to avoid Warp/marching_cubes OOM
- `TORCHDYNAMO_DISABLE=1` required to avoid Dynamo error in `hijack_qwenimage_multi.py` — likely why INT4 latency isn't seeing expected speedup

## Open questions
- More baseline meshes to run a bigger sample?
- Chamfer distance metric — sampled point-cloud CD, or latent-space?
- Would resolving the `torch.compile` incompatibility enable proper INT4 latency comparison?