import torch
import torch.nn.functional as F
import torchao
from huggingface_hub import hf_hub_download

REPO = "TrNi/efficient-cube3d"
quant_ckpt = hf_hub_download(REPO, "shape_gpt_rtn_int4_g128.pt")

with torch.serialization.safe_globals([torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor]):
    state_dict = torch.load(quant_ckpt, map_location="cuda:0")

weight = state_dict["text_proj.weight"]
bias = state_dict["text_proj.bias"]

print("weight device:", weight.device, type(weight))
print("bias device:", bias.device, type(bias))

x = torch.randn(4, 768, dtype=torch.bfloat16, device="cuda:0")
out = F.linear(x, weight, bias)
print("Forward pass output shape:", out.shape)
print("INT4 kernel dispatch successful")
