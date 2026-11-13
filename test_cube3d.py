"""
Smoke test: downloads the Cube3D INT4 RTN checkpoint and the BF16 tokenizer from HF
and checks the quantized weights load with torchao. Does not run generation.
"""
import torch 
from huggingface_hub import hf_hub_download

REPO = "TrNi/efficient-cube3d"
import torchao

print(f">> Downloading files from {REPO}")
quant_ckpt = hf_hub_download(REPO, "shape_gpt_rtn_int4_g128.pt")
quant_config = hf_hub_download(REPO, "quant_config.json")
config_yaml = hf_hub_download(REPO, "open_model_v0.5.yaml")

print(">> Files downloaded to:")
print("   ", quant_ckpt)
print("   ", quant_config)
print("   ", config_yaml)

import json
print(">> quant_config.json contents:")
print(json.dumps(json.load(open(quant_config)), indent=2))

print(">> Loading quantized state dict (CPU first)...")
with torch.serialization.safe_globals([torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor]):
    state_dict = torch.load(quant_ckpt, map_location="cpu")
print(f"   {len(state_dict)} tensors/keys found")

# check the torchao quantized tensor type
for i, (k, v) in enumerate(state_dict.items()):
    print(f"   [{i}] {k}: {type(v)} {getattr(v, 'shape', '')} {getattr(v, 'dtype', '')}")
    if i >= 4:
        break

if torch.cuda.is_available():
    print(">> CUDA detected -- moving one sample tensor to GPU to confirm kernel compatibility")
    torch.cuda.reset_peak_memory_stats()
    sample_key = next(iter(state_dict))
    sample_val = state_dict[sample_key]
    try:
        sample_val_cuda = sample_val.to("cuda")
        print(f"   moved {sample_key} to CUDA successfully, VRAM used: {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
    except Exception as e:
        print(f"   [!] failed to move to CUDA: {e}")
else:
    print(">> No CUDA visible -- run this inside an srun/sbatch GPU session to test the CUDA path")

print(">> Smoke test complete.")
