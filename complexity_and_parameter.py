import torch
from thop import profile
from thop import clever_format
from model.CF_UFormer_NoCAM import CF_UFormer

input = torch.randn(1, 3, 128, 128)

model_CF_UFormer = CF_UFormer(inp_channels=3, out_channels=3, dim=16, num_blocks=[2, 4, 8, 16], num_refinement_blocks=2,
                          heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
                          attention=True, skip=False)

flops, params = profile(model_CF_UFormer, inputs=(input,))
flops, params = clever_format([flops, params], '%.3f')

print('模型参数：', params)
print('每一个样本浮点运算量：', flops)


