# Copyright (C) 2025-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torch.nn.functional as F

import pow3r.tools.path_to_dust3r
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import geotrf, inv


def dot(A, B): 
    return (A.unsqueeze(-2) @ B.unsqueeze(-1)).squeeze(-1)

def compute_rays(pix, K):
    return geotrf(inv(K), pix, ncol=3)

@torch.no_grad()
def estimate_scene_scale( rel_pose, pts3d_in_cam1, pix2, K2 ):
    """ we try to estimate the relative pose' scale factor in 'closed form',
        based of the current estimate of pts3d seen from camera1.
        we already have p1 = project(pts3d, eye(4), K1)
        now we want p2 = project(pts3d, relpose(scale), K2)

        so rel_pose is cam1-to-cam2
    """
    assert rel_pose.ndim == 3 and rel_pose.shape[1:] == (4,4)
    assert pts3d_in_cam1.ndim == 3 and pts3d_in_cam1.shape[-1] == 3
    assert pix2.ndim == 3  and pix2.shape[-1] == 2
    assert K2.ndim == 3  and K2.shape[1:] == (3,3)

    # w2cam_E1 = np.eye(4)
    w2cam_E2 = rel_pose
    cam2w_E2 = inv(w2cam_E2)

    # world coord camera center
    # C1 = 0
    C2 = cam2w_E2[...,None,:3,3]

    # we want ray2 to intersect with pts3d
    rays_cam2 = compute_rays(pix2, K2)
    V2 = F.normalize(geotrf(cam2w_E2, rays_cam2) - C2, dim=-1)

    # compute the scale factor
    C_VCV = C2 - V2 * dot(V2, C2)
    scene_scale = (C_VCV * pts3d_in_cam1).sum(-1) / dot(C_VCV, C2).squeeze(-1)
    scene_scale = torch.median(scene_scale, dim=-1).values

    # fix the scene scale
    # cam2w_E2[..., :3, 3] *= scene_scale[...,None]
    # w2cam_E2 = inv(cam2w_E2)
    return scene_scale
