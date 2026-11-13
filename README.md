# 🚀 First INT4 Quantized Efficient-CubePart3D - 83% Smaller Part Decomposition Model
![](https://img.shields.io/badge/Model_Size-1.48_GB_(82.8%25↓)-blueviolet)
![](https://img.shields.io/badge/Peak_VRAM-27.09_GiB_(19.6%25↓)-brightgreen)
![](https://img.shields.io/badge/Part_CD_median-0.0114_(near_lossless)-blue)
![](https://img.shields.io/badge/Latency-66.5s_(eager_mode,_1.9x_slower)-orange)

This repo has the code for INT4 quantization of the [CubePart](https://huggingface.co/Roblox/cubepart) diffusion transformer and the BF16 vs INT4 benchmark.
Weights and model card are at [TrNi/efficient-cubepart3d](https://huggingface.co/TrNi/efficient-cubepart3d).

We use HQQ (Half-Quadratic Quantization) through `torchao.quantization.int4_weight_only(group_size=128, use_hqq=True)` at W4A16.
It needs no calibration data. Only `transformer_blocks` and `multi_transformer_blocks` inside `diffusion_model` are quantized.

## Results

250 prompts, 12 categories and 828 parts (SLURM job 1159996). The pipeline is fully seeded, so the chamfer distance of a configuration against itself is 0.

| | min | median | mean | max |
|---|---|---|---|---|
| CD(A,B): BF16 Cube3D input, BF16 vs INT4 CubePart | 0.0011 | 0.0114 | 0.0193 | 0.8206 |
| CD(C,D): INT4 Cube3D input, BF16 vs INT4 CubePart | 0.0014 | 0.0118 | 0.0239 | 1.6261 |

| | BF16 | **INT4 (HQQ W4A16)** |
|---|:-:|:-:|
| 💾 Model size (diffusion transformer) | 8.58 GB | **1.48 GB (82.8%↓)** |
| 🎮 Peak VRAM | 33.71 GiB | **27.09 GiB (19.6%↓)** ✨ |
| 📦 Setup time | 161.8 s | **166.1 s** (BF16 load + 4.3 s quantize) |
| ⏱️ Latency | 34.7 s | **66.5 s** (eager mode) |
| 🎯 Part chamfer distance vs BF16, median | n/a | **0.0114** |

Per prompt, per category and dataset statistics for both pairs are in `cd_stats.json`.
File sizes are in decimal GB and VRAM in GiB from `torch.cuda.max_memory_allocated()`. All numbers are measured on one L40S.

## BF16 vs INT4 renders

BFloat16 Cube+CubePart on the left, INT4 Cube+CubePart on the right. Full mesh on top, parts below.

We strived for grayscale renders with high details without injecting color bias. The model output is textureless, so we employ spherical harmonic lighting with Blinn-Phong specular term and ambient occlusion to accurately depict geometry and curves of each part with light rays.

![A pineapple with a crown of 8 spiky leaves and a hexagonal surface pattern](assets/sh_renders/191_figure.png)

![steampunk boombox with 2 circular speakers and 6 brass dials](assets/sh_renders/305_figure.png)

![A sitting cat with pointed ears and a long tail](assets/sh_renders/020_figure.png)

### Cube3D: BF16 vs INT4

| Prompt | BF16 | **INT4** |
|---|:-:|:-:|
| A pineapple with a crown of 8 spiky leaves and a hexagonal surface pattern | ![BF16 191](assets/sh_renders/191_cube3d_bf16.gif) | ![INT4 191](assets/sh_renders/191_cube3d_int4.gif) |
| steampunk boombox with 2 circular speakers and 6 brass dials | ![BF16 305](assets/sh_renders/305_cube3d_bf16.gif) | ![INT4 305](assets/sh_renders/305_cube3d_int4.gif) |
| A sitting cat with pointed ears and a long tail | ![BF16 020](assets/sh_renders/020_cube3d_bf16.gif) | ![INT4 020](assets/sh_renders/020_cube3d_int4.gif) |
| A lace doily pattern extruded to a thin 3D surface | ![BF16 118](assets/sh_renders/118_cube3d_bf16.gif) | ![INT4 118](assets/sh_renders/118_cube3d_int4.gif) |

The lace doily is a failure case, INT4 loses the lace pattern.

## Scripts

Run all scripts from the CubePart repo root (`cube/cubepart`) with:

```bash
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # Warp marching cubes OOM otherwise
export TORCHDYNAMO_DISABLE=1                               # Dynamo error in hijack_qwenimage_multi.py otherwise
```

Benchmark
- `benchmark_4config_sweep.py`: 4-config sweep (A/B/C/D, see below). Writes `<idx>_ab.json` and `<idx>_cd.json` per mesh and can resume.
- `full_sweep_4config.slurm`: SLURM job (`gpu-l40s`, 23 h) for the sweep and the stats.
- `compute_4config_cd_stats.py`: per prompt, per category and dataset statistics from the per-mesh results.
- `measure_efficiency.py`: latency, setup time and peak VRAM for BF16 and INT4.
- `migrate_cd_sum_to_average.py`: one-time fix for old results where chamfer distance was the sum of both directions. Not needed for new runs.

Parts lists
- `parts_override.json`: the 250 parts lists used for all numbers above.
- `rebuild_parts_override.py`: has the 250 lists, validates them and writes `parts_override.json` with `--apply`.
- `spot_check_parts_override.py`: flags part names that do not fit the `master_suite.json` prompts.
- `audit_alignment.py`: word overlap between prompts and parts per category. A good file has 0 to 15% zero overlap, the shuffled one had 55 to 85%.
- `diagnose_shuffle.py`: checks if the lists are offset by a constant (fixable) or shuffled (needs a rebuild).
- `inspect_defects.py`: prints flagged entries with two neighbours on each side.
- These read `nisarg_shared/parts_override.json` and `master_suite.json`.

Checkpoint
- `test_int4_checkpoint_save.py`: tries 3 save and reload patterns on a small module.
- `export_int4_checkpoint.py`: saves `multi_part_dit_int4_hqq_g128.pt` and checks the reload is bit exact on a real generation.
- `export_int4_safetensors.py`: converts the checkpoint to `.safetensors` and checks it bit for bit. Passed in job 1215174 (1,478,462,752 bytes).
- `export_safetensors.slurm`: SLURM job for the export, the check runs in a new process.
- `time_load_paths.py`, `time_load_paths.slurm`: timing of repeated generations per load path (bf16, pt, st). st and pt are the same (64.6 s).

Meshes
- `select_visual_prompts.py`, `select_3_demo_prompts.py`: pick prompts for the mesh package.
- `export_visual_assets.py`: exports Cube3D meshes and CubePart part meshes as `.glb` with a `manifest.json`.
- `load_parts_example.py`, `tutorial_part_mapping.py`: map part files back to part names.
- `render_sh_spec_ao.py`: older CPU renderer (SH, specular, AO). Replaced by the `SH_BPS_AO` renderer below.

Renders
- `sh_bps_ao_colab.ipynb`: `SH_BPS_AO` renderer (nvdiffrast and Embree AO) with 18 Fibonacci sphere views per mesh. All renders above come from it. It is Nisarg Trivedi's notebook with two changes: rows are flipped since nvdiffrast row 0 is the image bottom, and there is no upright rotation since our meshes are +y up.
- `sh_figures_colab.ipynb`: makes the BF16 vs INT4 figures and GIFs from the renders. No GPU needed.
- `render_sh_bps_ao.py`, `render_sh_bps_ao.slurm`: same renderer as a SLURM job. Not run on a GPU yet.

Older scripts: `quantize_cubepart_rtn.py`, `benchmark_spot_check.py`, `benchmark_cubepart_bf16_vs_int4.py`, `probe_cuda.py`, `test_cube3d.py` and `benchmark_full_sweep.py`.
`benchmark_full_sweep.py` is the first 2-config sweep. Its sampling was not seeded and its chamfer distance was a sum, so its numbers are not comparable. `benchmark_4config_sweep.py` replaces it.

## Experimental design

| Config | Cube3D input mesh | CubePart model |
|---|---|---|
| A | BF16 | BF16 |
| B | BF16 | INT4 |
| C | INT4 | BF16 |
| D | INT4 | INT4 |

The diffusion model is quantized once in place, after the A/C pass and before the B/D pass.
Surface sampling and the chamfer metric are seeded per mesh (`seed = idx`).
Chamfer distance is the average of the forward and backward mean nearest neighbour distance over 5000 sampled points per part.

<!-- ## Superseded: spot check, 2 meshes (August 2026)

Kept for the record; do not cite. These numbers came before the sampling seed
fix and the rebuilt parts lists, and the VRAM figures were not a true
generation-time peak.

- BF16 peak VRAM 28.33 GB; INT4 peak VRAM 21.23 GB (an INT4-only run gave 11.68 GB)
- CD median across all parts: 0.539
- Latency: BF16 about 48 to 57 s per mesh, INT4 about 64 to 70 s per mesh

See `spot_check_summary.json`.
-->
