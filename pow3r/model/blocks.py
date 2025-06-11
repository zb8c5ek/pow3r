# this is a modification of the Attention and Block class from timm (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
# with a decoder that takes an additional input and perform self-attention, cross-attention and mlp
import torch
import torch.nn as nn 

import pow3r.tools.path_to_dust3r
from croco.models.blocks import Mlp


class CoreAttention (nn.Module):
    def __init__(self, rope=None, attn_drop=0.):
        super().__init__()
        self.rope = rope
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_val = attn_drop

    def attention(self, num_cls, q, k, v, qpos=None, kpos=None, mask_for_attention=None):
        assert q.ndim == k.ndim == v.ndim == 4
        B, H, Nq, C = q.shape
        Nk = k.shape[-2]
        assert k.shape == v.shape == (B, H, Nk, C)

        # first, apply RoPE inline
        q_ini = q
        k_ini = k
        if num_cls > 0:
            q_no_cls = q[:, :, num_cls:].clone()
            k_no_cls = k[:, :, num_cls:].clone()
            qpos_no_cls = qpos[:, num_cls:].clone() if qpos is not None else None
            kpos_no_cls = kpos[:, num_cls:].clone() if kpos is not None else None
        else:
            q_no_cls = q
            k_no_cls = k
            qpos_no_cls = qpos
            kpos_no_cls = kpos

        if self.rope is not None: 
            q_no_cls = self.rope(q_no_cls, qpos_no_cls)
            k_no_cls = self.rope(k_no_cls, kpos_no_cls)

        if num_cls > 0:
            """ Computing the attention between
                        | K_cls |    K_pos   |
                |-------|-------|------------|
                | Q_cls |       | sim_qcls_k |
                |-------| sim_q |------------|
                | Q_pos |       |  sim_rope  |
                |-------|-------|------------|
            """
            # cls tokens are first
            sim = q_no_cls @ k_no_cls.transpose(-2, -1)

            sim_qcls_k = q_ini[:, :, :num_cls] @ k_ini[:, :, num_cls:].transpose(-2, -1)
            sim_q = q_ini @ k_ini[:, :, :num_cls].transpose(-2, -1)
            sim = torch.concatenate([sim_qcls_k, sim], dim=-2)
            sim = torch.concatenate([sim_q, sim], dim=-1)
        else:
            sim = q_no_cls @ k_no_cls.transpose(-2, -1)

        sim *= self.scale
        if mask_for_attention is not None:
            sim = torch.where(mask_for_attention.unsqueeze(1), sim, -float('inf'))
        attn = sim.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v) # B, H, Nq, C

        return x.transpose(1, 2).reshape(B, Nq, C*H)


class Attention (CoreAttention):
    """compared to timm.models.vision_transformer Attention class, just add pos embedding that acts within the attention"""

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., qkln=False):
        super().__init__(rope=rope, attn_drop=attn_drop)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkln = qkln
        if self.qkln:
            self.qln = nn.LayerNorm(head_dim, eps=1e-06, elementwise_affine=False)
            self.kln = nn.LayerNorm(head_dim, eps=1e-06, elementwise_affine=False)

    def forward(self, num_cls, x, xpos, mask_for_attention=None):
        B, N, C = x.shape
        
        if self.qkln: # a bit ugly to have contiguous tensor for curope
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = [qkv.select(2, i) for i in range(3)]
            q = self.qln(q).transpose(1,2)
            k = self.kln(k).transpose(1,2)
            v = v.transpose(1,2)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
            q, k, v = [qkv.select(2, i) for i in range(3)]

        x = self.attention(num_cls, q, k, v, xpos, xpos, mask_for_attention)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block (nn.Module):
    """compared to timm.models.vision_transformer Block class, just add pos embedding that acts within the attention"""

    def __init__(self, dim, num_heads, mlp_ratio=4., rope=None, qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, qkln=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, qkln=qkln)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, xpos=None, num_cls=0, **kw):
        x = x + self.attn(num_cls, self.norm1(x), xpos)
        x = x + self.mlp(self.norm2(x))
        return x


class BlockInject (Block):
    def init(self, dim):
        self.rays_embed = Mlp(dim)
        self.depth_embed = Mlp(dim)

    def forward(self, x, xpos=None, num_cls=0, rays=None, depth=None):
        x = x + self.attn(num_cls, self.norm1(x), xpos)
        if rays is not None:
            x = x + self.rays_embed(rays)
        if depth is not None:
            x = x + self.depth_embed(depth)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention (CoreAttention):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., qkln=False, 
                       kv_dim=None):
        super().__init__(rope=rope, attn_drop=attn_drop)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        if kv_dim is None: kv_dim = dim

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkln = qkln
        if self.qkln:
            self.qln = nn.LayerNorm(head_dim, eps=1e-06, elementwise_affine=False)
            self.kln = nn.LayerNorm(head_dim, eps=1e-06, elementwise_affine=False)

    def forward(self, num_cls, query, key, value, qpos, kpos):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]
        
        if self.qkln: # a bit ugly to have contiguous tensor for curope
            q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads)
            k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads)
            v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads)
            q = self.qln(q)
            k = self.kln(k)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        else:
            q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
            k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
            v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)

        x = self.attention(num_cls, q, k, v, qpos, kpos)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, rope=None, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True,
                 is_rope_for_sa_only=False, qkln=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop, qkln=qkln)
        self.cross_attn = CrossAttention(dim, rope=None if is_rope_for_sa_only else rope, 
                                         num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, qkln=qkln)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.mlp = Mlp( in_features=dim, hidden_features=int(dim * mlp_ratio), 
                        act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos, num_cls=0, mask_for_attention=None, **kw):
        x = x + self.attn(num_cls, self.norm1(x), xpos, mask_for_attention)
        y_ = self.norm_y(y)
        x = x + self.cross_attn(num_cls, self.norm2(x), y_, y_, xpos, ypos)
        
        x = x + self.mlp(self.norm3(x))
        return x, y


class DecoderBlockInject (Block):
    def init(self, dim):
        self.pose_embed = Mlp(dim)

    def forward(self, x, y, xpos, ypos, num_cls=0, mask_for_attention=None, relpose=None):
        x = x + self.attn(num_cls, self.norm1(x), xpos, mask_for_attention)
        y_ = self.norm_y(y)
        x = x + self.cross_attn(num_cls, self.norm2(x), y_, y_, xpos, ypos)

        if relpose is not None:
            if num_cls > 0:
                # only modify the CLS token
                cls = x[:,:num_cls]
                x = torch.cat((cls + self.pose_embed(relpose), x[:,num_cls:]), dim=1)
            else:
                x = x + self.pose_embed(relpose)

        x = x + self.mlp(self.norm3(x))
        return x, y

# modify croco
import dust3r.utils.path_to_croco
import models.blocks 
models.blocks.Attention = Attention
models.blocks.Block = Block
models.blocks.CrossAttention = CrossAttention
models.blocks.DecoderBlock = DecoderBlock
