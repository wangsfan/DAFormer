# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

'''These modules are adapted from those of timm, see
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''

import copy
import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.parameter import Parameter
import continual.utils as cutils

import torch


def min_max_normalize(tensor):
    # 获取最小值和最大值
    min_value = tensor.min()
    max_value = tensor.max()

    # 进行 Min-Max 归一化
    normalized_tensor = (tensor - min_value) / (max_value - min_value)

    return normalized_tensor


def z_score_normalize(tensor):
    # 计算均值和标准差
    mean_value = tensor.mean()
    std_value = tensor.std()

    # 进行 Z-Score 归一化
    normalized_tensor = (tensor - mean_value) / std_value

    return normalized_tensor


class BatchEnsemble(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.out_features, self.in_features = out_features, in_features
        self.bias = bias

        self.r = nn.Parameter(torch.randn(self.out_features))
        self.s = nn.Parameter(torch.randn(self.in_features))

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls, self.in_features, self.out_features, self.bias)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        result.linear.weight = self.linear.weight
        return result

    def reset_parameters(self):
        device = self.linear.weight.device
        self.r = nn.Parameter(torch.randn(self.out_features).to(device))
        self.s = nn.Parameter(torch.randn(self.in_features).to(device))

        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x):
        w = torch.outer(self.r, self.s)
        w = w * self.linear.weight
        return F.linear(x, w, self.linear.bias)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., fc=nn.Linear):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = fc(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = fc(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, BatchEnsemble):
            trunc_normal_(m.linear.weight, std=.02)
            if isinstance(m.linear, nn.Linear) and m.linear.bias is not None:
                nn.init.constant_(m.linear.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True, fc=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.apply(self._init_weights)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1) != N:
            self.get_rel_indices(N)

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, v

    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1, -1)
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map=False):

        attn_map = self.get_attention(x).mean(0)  # average over batch
        distances = self.rel_indices.squeeze()[:, :, -1] ** .5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self, locality_strength=1.):
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1  # max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads ** .5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight.data[position, 2] = -1
                self.pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches ** .5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        self.rel_indices = rel_indices.to(device)


class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., fc=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N ** .5)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        distances = indd ** .5
        distances = distances.to('cuda')

        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N

        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class ScaleNorm(nn.Module):
    """See
    https://github.com/lucidrains/reformer-pytorch/blob/a751fe2eb939dcdd81b736b2f67e745dc8472a09/reformer_pytorch/reformer_pytorch.py#L143
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type=GPSA,
                 fc=nn.Linear, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention_type(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                   proj_drop=drop, fc=fc, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, fc=fc)

    def reset_parameters(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.reset_parameters()
        self.mlp.apply(self.mlp._init_weights)

    def forward(self, x, z=None, hyperFC=None):
        xx = self.norm1(x)
        if isinstance(self.attn, HyperDecoder):
            xx, attn, v = self.attn(xx, z, hyperFC)

        else:
            xx, attn, v = self.attn(xx)
        x = self.drop_path(xx) + x
        x = self.drop_path(self.mlp(self.norm2(x))) + x

        return x, attn, v


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding, from timm
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding, from timm
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)
        self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


## add HyperNetworks
class HyperNetwork(nn.Module):

    def __init__(self, z_dim=128, fc_in=384, fc_out=384, num_wights=4, num_bias=4):
        super(HyperNetwork, self).__init__()
        self.num_weights = num_wights
        self.num_bias = num_bias

        self.z_dim = z_dim
        self.fc_out = fc_in
        self.fc_out = fc_out

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.fc_out*self.num_weights)).cuda(), 2))
        self.b1 = Parameter(torch.fmod(torch.randn(self.fc_out * self.num_weights).cuda(), 2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.z_dim * self.z_dim)).cuda(), 2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.z_dim * self.z_dim)).cuda(), 2))

        self.w_bias = Parameter(torch.fmod(torch.randn((fc_in, self.fc_out * self.num_bias)).cuda(), 2))
        self.b_bias = Parameter(torch.fmod(torch.randn((self.fc_out * self.num_bias)).cuda(), 2))


    def forward(self, z):
        h1 = torch.matmul(z[:self.z_dim], self.w2) + self.b2
        h2 = torch.matmul(z[self.z_dim:self.z_dim*2], self.w2) + self.b2
        h3 = torch.matmul(z[self.z_dim*2:self.z_dim*3], self.w2) + self.b2
        h_in = torch.cat([h1, h2, h3], dim=0)

        h_in = h_in.view(self.fc_out, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        weight = h_final.view(self.fc_out, self.fc_out, self.num_weights).permute(2, 0, 1)

        bias = torch.matmul(z, self.w_bias) + self.b_bias
        bias = bias.view(self.fc_out, self.num_bias).permute(1, 0)

        return weight, bias


class HyperDecoder(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None, qkv_bias=None, fc=None):
        super(HyperDecoder, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # self.HyperFC = HyperNetwork(z_dim=dim//3, fc_in=dim, fc_out=dim)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(64)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, z, HyperFC):
        assert z is not None, "z is NULL, please Enter parameter [z]!"

        B, N, C = x.shape
        FC_weight, FC_bias = HyperFC(z)
        q_weight, k_weight, v_weight = FC_weight[0].squeeze(), FC_weight[1].squeeze(),FC_weight[2].squeeze()
        q_bias, k_bias, v_bias = FC_bias[0].squeeze(), FC_bias[1].squeeze(), FC_bias[2].squeeze()
        proj_weight = FC_weight[3].squeeze()
        proj_bias = FC_bias[3].squeeze()


        # q_weight, k_weight = FC_weight[0].squeeze(), FC_weight[1].squeeze()
        # proj_weight = FC_weight[2].squeeze()

        # q_weight = F.normalize(q_weight, p=2, dim=None)
        # k_weight = F.normalize(k_weight, p=2, dim=None)
        # v_weight = F.normalize(v_weight, p=2, dim=None)
        # proj_weight = F.normalize(proj_weight, p=2, dim=None)

        q = self.bn1(F.linear(x, q_weight, q_bias).reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)
        k = self.bn2(F.linear(x, k_weight, k_bias).reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)
        v = self.bn3(F.linear(x, v_weight, v_bias).reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)
        # v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.bn4(F.linear(x_out, proj_weight, proj_bias))
        x_out = self.proj_drop(x_out)

        return x_out, attn, FC_weight


class Hyper_mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., mlp_ratio=4):
        super().__init__()
        self.mlp_ratio = int(mlp_ratio)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        # self.bn1 = nn.LayerNorm(self.hidden_features)
        # self.bn2 = nn.LayerNorm(self.out_features)

    def forward(self, x, z=None, hypernetwork=None):
        FC_weight, FC_bias = hypernetwork(z)

        fc1_wight = FC_weight[:self.mlp_ratio].reshape(self.hidden_features, self.in_features)
        fc2_wight = FC_weight[self.mlp_ratio:].reshape(self.out_features, self.hidden_features)

        fc1_bias = FC_bias[:self.mlp_ratio].reshape(self.hidden_features)
        fc2_bias = FC_bias[-1].squeeze()

        x = F.linear(x, fc1_wight, fc1_bias)
        x = self.act(x)
        x = self.drop(x)
        x = F.linear(x, fc2_wight, fc2_bias)
        x = self.drop(x)
        return x


class HyperVit(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=1000, embed_dim=384, depth=6,
                 num_heads=12, mlp_ratio=4.,qkv_bias=False,  qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer='layer',
                 local_up_to_layer=3, locality_strength=1., use_pos_embed=True,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.final_dim = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed
        self.HyperFC = HyperNetwork(z_dim=embed_dim // 3, fc_in=embed_dim, fc_out=embed_dim)
        self.Hyper_mlp = HyperNetwork(z_dim=embed_dim // 3, fc_in=embed_dim, fc_out=embed_dim,
                                      num_wights=int(mlp_ratio*2), num_bias=int(mlp_ratio+1))

        if norm_layer == 'layer':
            norm_layer = nn.LayerNorm
        elif norm_layer == 'scale':
            norm_layer = ScaleNorm
        else:
            raise NotImplementedError(f'Unknown normalization {norm_layer}')

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.hyper_len = depth-local_up_to_layer

        self.z = nn.Parameter(torch.fmod(torch.randn(self.embed_dim, self.hyper_len).cuda(), 2))
        trunc_normal_(self.z, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []

        for layer_index in range(depth):
            if layer_index < local_up_to_layer:
                block = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[layer_index], norm_layer=norm_layer,
                    attention_type=GPSA, locality_strength=locality_strength
                )
            else:
                block = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[layer_index], norm_layer=norm_layer,
                    attention_type=HyperDecoder
                )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, embed_dim))

        # Classifier head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def freeze(self, names):
        for name in names:
            if name == 'all':
                return cutils.freeze_parameters(self)
            elif name == 'old_heads':
                self.head.freeze(name)
            elif name == 'backbone':
                cutils.freeze_parameters(self.blocks)
                cutils.freeze_parameters(self.patch_embed)
                cutils.freeze_parameters(self.pos_embed)
                cutils.freeze_parameters(self.norm)
            else:
                raise NotImplementedError(f'Unknown name={name}.')


    def reset_classifier(self):
        self.head.apply(self._init_weights)

    def reset_parameters(self):
        for b in self.blocks:
            b.reset_parameters()
        self.norm.reset_parameters()
        self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_internal_losses(self, clf_loss):
        return {}

    def end_finetuning(self):
        pass

    def begin_finetuning(self):
        pass

    def epoch_log(self):
        return {}

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'z'}

    def get_classifier(self):
        return self.head

    def forward_sa(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks[:self.local_up_to_layer]:
            x, _ = blk(x)

        return x

    def forward_features(self, x, final_norm=True):
        B = x.shape[0]
        x = self.patch_embed(x)


        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks[:self.local_up_to_layer]:
            x, _, _ = blk(x)

        for blk in self.blocks[self.local_up_to_layer:]:
            for i in range(self.hyper_len):
                x, _, _ = blk(x, self.z[:, i], self.HyperFC, self.Hyper_mlp)
        x = self.norm(x)
        return x, None, None

    def forward(self, x):
        x = self.forward_features(x)[0]
        x = self.avg_pooling(x)
        x = self.head(x)
        return x

# from torchstat import stat
# import numpy as np
if __name__ == '__main__':
    # x = torch.randn((128, 3, 32, 32)).cuda()
    # x = torch.randn((128, 16, 384)).cuda()
    # z = torch.randn(384).cuda()
    #
    # model1 = HyperNetwork(num_wights=8, num_bias=5)
    # model2 = Hyper_mlp(in_features=384, hidden_features=384*4, mlp_ratio=4)
    #
    # out = model2(x, z, model1)
    # model3 = HyperVit(num_classes=10, embed_dim=384)
    # model3 = model3.to("cuda")
    # stat(model3, (3, 32, 32))
    #
    # para = sum([np.prod(list(p.size())) for p in model3.parameters()])
    # print('Model {} : params: {:4f}M'.format(model3._get_name(), para / 1000 / 1000))



    # a = model3(x)
    # print(a.shape)
    print()