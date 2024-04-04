import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

import einops
from einops import rearrange
import numpy as np
import pytorch_colors as colors
import math

#################################dimension transformation#####################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


#################################Transformer#####################################

# layer norm
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # input of x ：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Axis-based Multi-head Self-Attention
class SelfAttentionImplement(nn.Module):
    def __init__(self, num_dims, num_heads, bias) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

        self.factor = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)
        return

    def forward(self, x):
        # x: [n, c, h, w]
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        reshape = lambda x: einops.rearrange(x, "n (nh dh) h w -> (n nh h) w dh", nh=n_heads, dh=dim_head)

        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = map(reshape, qkv.chunk(3, dim=1))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # factor = dim_head ** -0.5
        res = k.transpose(-2, -1)
        res = torch.matmul(q, res) * self.factor
        res = torch.softmax(res, dim=-1)

        res = torch.matmul(res, v)
        res = einops.rearrange(res, "(n nh h) w dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        res = self.fin(res)

        return res


# Axis-based Multi-head Self-Attention (row and col attention)
class SelfAttention(nn.Module):
    def __init__(self, num_dims, num_heads=1, bias=True) -> None:
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.row_att = SelfAttentionImplement(num_dims, num_heads, bias)
        self.col_att = SelfAttentionImplement(num_dims, num_heads, bias)
        self.channel_attention = ChannelAttention(num_dims)
        return

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4

        x = self.row_att(x)
        x = x.transpose(-2, -1)
        x = self.col_att(x)
        x = x.transpose(-2, -1)
        return x


# 定义多层感知机模型

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

# Dual Gated Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.mlp = MLP(dim, hidden_features, dim)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x


#  Axis-based Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


################################PatchEmbedding####################################

# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


###############################Down/Up sampling###################################

# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##################################Multi-level Fusion Unit###################################

#  MFU Block
class LAM_Module_v2(nn.Module):

    def __init__(self, in_dim, bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim
        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d(self.chanel_in, self.chanel_in * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in * 3, self.chanel_in * 3, kernel_size=3, stride=1, padding=1,
                                    groups=self.chanel_in * 3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)
        # self.out_conv = nn.Conv2d(self.chanel_in, int(self.chanel_in/3), kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize, N * C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)
        # print(out_1.shape)

        out = out_1 + x
        out = out.view(m_batchsize, -1, height, width)
        # print(out.shape)
        # out = self.out_conv(out)
        # print(out.shape)
        return out


#################################CF_UFormer#########################################
class CF_UFormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=16,
                 num_blocks=[1, 2, 4, 8],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 attention=True,
                 skip=False
                 ):

        super(CF_UFormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)  # patch_embedding (3x3 conv)
        self.block_hsv_1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.block_hsv_2 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.block_hsv_3 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.block_lab_1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.block_lab_2 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.block_lab_3 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.block_rgb_1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.block_rgb_2 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.block_rgb_3 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        # self.diffusion = Diffusion(dim * 3, dim * 9)
        self.layer_fussion = LAM_Module_v2(in_dim=int(dim * 3))
        self.conv_fussion = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.down_1 = Downsample(int(dim))  # From Level 0 to Level 1
        self.encoder_1 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2)), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down_2 = Downsample(int(dim * 2))  # From Level 1 to Level 2
        self.encoder_2 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 2)), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down_3 = Downsample(int(dim * 2 * 2))  # From Level 2 to Level 3
        self.encoder_3 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 4)), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down_4 = Downsample(int(dim * 2 * 4))  # From Level 3 to Level 4

        self.channel_attention_1 = ChannelAttention(128)
        self.channel_attention = ChannelAttention(256)
        self.neckbottle = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 8)), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up_1 = Upsample(int(dim * 2 * 8))  # From Level 4 to Level 3
        self.decoder_1 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 4)), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.up_2 = Upsample(int(dim * 2 * 4))  # From Level 3 to Level 2
        self.decoder_2 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 2)), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up_3 = Upsample(int(dim * 2 * 2))  # From Level 2 to Level 1
        self.decoder_3 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2)), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.up_4 = Upsample(int(dim * 2))  # From Level 1 to Level 0

        # skip connection wit weights
        self.coefficient_1 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2 * 4))))), requires_grad=attention)
        self.coefficient_2 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2 * 2))))), requires_grad=attention)
        self.coefficient_3 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2))))), requires_grad=attention)
        self.coefficient_4 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim))))), requires_grad=attention)

        # skip then conv 1x1
        self.skip_4 = nn.Conv2d(int(int(dim * 2 * 4)), int(int(dim * 2 * 4)), kernel_size=1, bias=bias)
        self.skip_3 = nn.Conv2d(int(int(dim * 2 * 2)), int(int(dim * 2 * 2)), kernel_size=1, bias=bias)
        self.skip_2 = nn.Conv2d(int(int(dim * 2)), int(int(dim * 2)), kernel_size=1, bias=bias)
        self.skip_1 = nn.Conv2d(int(int(dim * 2)), int(int(dim * 2)), kernel_size=1, bias=bias)

        self.feature_mapping = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.refinement_1 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refinement_2 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refinement_3 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        # self.diffusion = Diffusion(dim * 3, dim * 9)
        self.layer_fussion_2 = LAM_Module_v2(in_dim=int(dim * 3))
        self.conv_fuss_2 = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.out_cov = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.skip = skip

    def forward(self, inp_img):

        img_hsv = colors.rgb_to_hsv(inp_img)
        inp_projection_hsv = self.patch_embed(img_hsv)

        out_block_hsv_1 = self.block_hsv_1(inp_projection_hsv)
        out_block_hsv_2 = self.block_hsv_2(out_block_hsv_1)
        out_block_hsv_3 = self.block_hsv_3(out_block_hsv_2)

        inp_fusion_hsv = torch.cat([out_block_hsv_1.unsqueeze(1), out_block_hsv_2.unsqueeze(1), out_block_hsv_3.unsqueeze(1)], dim=1)  # (b,3,16,h,w)
        # inp_fusion_hsv = self.diffusion(inp_fusion_hsv)
        out_fusion_hsv = self.layer_fussion(inp_fusion_hsv)  # (b,48,h,w)
        out_fusion_hsv = self.conv_fussion(out_fusion_hsv)  # (b,16,h,w)

        img_lab = colors.rgb_to_lab(inp_img)
        inp_projection_lab = self.patch_embed(img_lab)

        out_block_lab_1 = self.block_lab_1(inp_projection_lab)
        out_block_lab_2 = self.block_lab_2(out_block_lab_1)
        out_block_lab_3 = self.block_lab_3(out_block_lab_2)

        inp_fusion_lab = torch.cat([out_block_lab_1.unsqueeze(1), out_block_lab_2.unsqueeze(1), out_block_lab_3.unsqueeze(1)], dim=1)  # (b,3,16,h,w)
        # inp_fusion_lab = self.diffusion(inp_fusion_lab)
        out_fusion_lab = self.layer_fussion(inp_fusion_lab)  # (b,48,h,w)
        out_fusion_lab = self.conv_fussion(out_fusion_lab)  # (b,16,h,w)


        # input projection
        img_rgb = self.patch_embed(inp_img)  # (b,16,h,w)

        # feature extraction
        out_block_rgb_1 = self.block_rgb_1(img_rgb)
        out_block_rgb_2 = self.block_rgb_2(out_block_rgb_1)
        out_block_rgb_3 = self.block_rgb_3(out_block_rgb_2)

        # feature fusion
        inp_fusion_rgb = torch.cat([out_block_rgb_1.unsqueeze(1), out_block_rgb_2.unsqueeze(1), out_block_rgb_3.unsqueeze(1)], dim=1)  # (b,3,16,h,w)
        # inp_fusion_rgb = self.diffusion(inp_fusion_rgb)
        out_fusion_rgb = self.layer_fussion(inp_fusion_rgb)  # (b,48,h,w)
        out_fusion_rgb = self.conv_fussion(out_fusion_rgb)  # (b,16,h,w)

        inp_fusion_col = torch.cat([out_fusion_hsv, out_fusion_lab, out_fusion_rgb], dim=1)  # (b,48,h,w)
        # inp_fusion_col = self.diffusion(inp_fusion_col)
        # inp_fusion_col = self.layer_fussion(inp_fusion_col)
        out_col_fussion = self.conv_fussion(inp_fusion_col)  # (b,16,h,w)



        # encoder of U-Net
        inp_encoder_1 = self.down_1(out_col_fussion)  # (b,32,h/2,w/2)
        out_encoder_1 = self.encoder_1(inp_encoder_1)

        inp_encoder_2 = self.down_2(out_encoder_1)  # (b,64,h/4,w/4)
        out_encoder_2 = self.encoder_2(inp_encoder_2)

        inp_encoder_3 = self.down_3(out_encoder_2)  # (b,128,h/8,w/8)
        attention_value_1 = self.channel_attention_1(inp_encoder_3)
        inp_encoder_3 = inp_encoder_3.mul(attention_value_1)
        out_encoder_3 = self.encoder_3(inp_encoder_3)

        inp_encoder_4 = self.down_4(out_encoder_3)  # (b,256,h/16,w/16)
        attention_value = self.channel_attention(inp_encoder_4)
        inp_encoder_4 = inp_encoder_4.mul(attention_value)
        inp_encoder_4 = self.neckbottle(inp_encoder_4)
        out_encoder_4 = self.up_1(inp_encoder_4)  # (b,128,h/8,w/8)

        # skip and decoder of U-Net
        inp_decoder_1 = self.coefficient_1[0, :][None, :, None, None] * out_encoder_3 + self.coefficient_1[1, :][None, :, None, None] * out_encoder_4
        inp_decoder_1 = self.skip_4(inp_decoder_1)  # conv 1x1 (b,128,h/8,w/8)
        out_decoder_1 = self.decoder_1(inp_decoder_1)
        out_decoder_1 = self.up_2(out_decoder_1)  # (b,64,h/4,w/4)

        inp_decoder_2 = self.coefficient_2[0, :][None, :, None, None] * out_encoder_2 + self.coefficient_2[1, :][None, :, None, None] * out_decoder_1
        inp_decoder_2 = self.skip_3(inp_decoder_2)
        out_decoder_2 = self.decoder_2(inp_decoder_2)
        out_decoder_2 = self.up_3(out_decoder_2)  # (b,32,h/2,w/2)

        inp_decoder_3 = self.coefficient_3[0, :][None, :, None, None] * out_encoder_1 + self.coefficient_3[1, :][None, :, None, None] * out_decoder_2
        inp_decoder_3 = self.skip_1(inp_decoder_3)
        out_decoder_3 = self.decoder_3(inp_decoder_3)
        out_decoder_3 = self.up_4(out_decoder_3)  # (b,16,h,w)

        out_fusion_123 = self.feature_mapping(out_col_fussion)
        out = self.coefficient_4[0, :][None, :, None, None] * out_fusion_123 + self.coefficient_4[1, :][None, :, None, None] * out_decoder_3

        # refinement
        out_1 = self.refinement_1(out)
        out_2 = self.refinement_2(out_1)
        out_3 = self.refinement_3(out_2)

        # feature fusion
        inp_fusion = torch.cat([out_1.unsqueeze(1), out_2.unsqueeze(1), out_3.unsqueeze(1)], dim=1)  # (b,3,16,h,w)
        # inp_fusion = self.diffusion(inp_fusion)
        out_fusion_123 = self.layer_fussion_2(inp_fusion)  # (b,48,h,w)
        out = self.conv_fuss_2(out_fusion_123)  # (b,16,h,w)

        if self.skip:
            out = self.out_cov(out) + inp_img
        else:
            out = self.out_cov(out)
        return out  # (b,3,h,w)



