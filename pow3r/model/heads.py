# Copyright (C) 2025-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torch.nn as nn
import torch.nn.functional as F


def head_factory( head_type, net ):
    """" build a prediction head for the decoder 
    """
    head, tags = head_type.split('_')
    tags = tags.split('+')

    if head_type.startswith('linear'):
        return LinearPts3d(tags, net)

    raise ValueError(f"unexpected {head_type=}")


class LinearPts3d (nn.Module):
    """ Each token outputs:
        - 16x16 3D points
    """
    def __init__(self, tags, net):
        super().__init__()
        self.tags = tags
        self.patch_size = tuple_to_int(net.patch_embed.patch_size)

        n_chan_out = postprocess(None, self.tags, ret_nchan=True)
        self.proj = nn.Linear(net.dec_embed_dim, n_chan_out * self.patch_size**2)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B,S,D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens) # B,S,D
        feat = feat.transpose(-1,-2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size) # B,3,H,W

        # permute + norm depth
        return postprocess(feat, self.tags)


def postprocess(out, tags, ret_nchan=False):
    if ret_nchan:
        fmap = torch.zeros((1,999)) # dummy
    else:
        fmap = out.permute(0,2,3,1) # B,H,W,D

    res = {}
    cur = 0

    for tag in tags:
        if tag.startswith('pts3d'):
            res[tag] = fmap[...,cur:cur+3]
            cur += 3

        if tag.startswith('conf'):
            tag, mode = split_tag(tag)
            res[tag] = reg_dense_conf(fmap[...,cur], mode=mode) 
            cur += 1

    if ret_nchan: return cur
    return res


def reg_dense_conf( x, mode ):
    if not mode:
        mode = ('exp', 1, float('inf'))
    elif isinstance(mode, str):
        mode = eval(mode)
    mode, vmin, vmax = mode

    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)

    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin

    raise ValueError(f'bad {mode=}')


def reg_desc(desc, mode):
    if 'norm' in mode:
        desc = desc / desc.norm(dim=-1,keepdim=True)
    else:
        raise ValueError(f"Unknown desc mode {mode}")
    return desc


def tuple_to_int(psize):
    assert len(psize) == 2
    assert psize[0] == psize[1]
    return psize[0]


GLOBALS = dict(
    inf = float('inf'),
    exp = 'exp',
    sigmoid = 'sigmoid',
    )

def split_tag(tag):
    # tag = "conf2(exp,1,inf)" --> returns 'conf2', ('exp',1,inf)
    p = tag.find('(')
    if p < 0: return tag, None
    q = tag.find(')',p)
    assert p < q  
    return tag[:p], eval(tag[p:q+1], GLOBALS)
