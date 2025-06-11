# Copyright (C) 2025-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import numpy as np
from tqdm import tqdm
from copy import deepcopy

import roma
import torch
import torch.nn as nn
import torch.nn.functional as F

import pow3r.tools.path_to_dust3r
from dust3r.utils.device import todevice, to_numpy
from dust3r.utils.geometry import geotrf, inv, xy_grid
from dust3r.inference import find_opt_scaling
from dust3r.post_process import estimate_focal_knowing_depth

from pow3r.tools.geometry import estimate_scene_scale
from pow3r.datasets.utils.modalities import gen_rays, gen_sparse_depth, gen_rel_pose
from pow3r.model import * # to build the model


__all__ = "SlidingResolver1D SlidingResolver CoarseToFine1D CoarseToFine AsymmetricSliding scale_K DUMP_TMP_FOLDER".split()


class BaseResolver (nn.Module):
    """ Base class. Has convenient functions but forward() is not implemented.
    """
    def __init__(self, crop_resolution, fix_rays=False):
        super().__init__()
        if isinstance(crop_resolution, int):
            crop_resolution = (crop_resolution,crop_resolution)
        self.crop_resolution = crop_resolution
        self.fix_rays = fix_rays
        assert self.fix_rays in (False, True, 'full')

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def load_from_checkpoint(self, ckpt):
        # here we load the actual model
        model = ckpt['definition']
        epoch = ckpt['epoch']

        print(f'>> Creating model = {model}')
        self.model = eval(model)
        print(f'>> Loading model at {epoch=}:', self.model.load_state_dict(ckpt['weights']) )
        return epoch

    def print_auxiliary_info(self, *input_views):
        view_tags = []
        for i,view in enumerate(input_views, 1):
            tags = []
            if 'camera_intrinsics' in view:
                tags.append(f'K{i}')
            if 'camera_pose' in view:
                tags.append(f'P{i}')
            if 'depthmap' in view:
                tags.append(f'D{i}')
            print(f'>> Receiving {{{"+".join(tags)}}} for view{i}')
            view_tags.append(tags)
        return view_tags
        
    def inference_with_info(self, view1, view2, K1=None, K2=None, D1=None, D2=None, cam1=None, cam2=None, ret_views=False):
        view1 = dict(view1)
        view2 = dict(view2)
        add_intrinsics(view1, K1)
        add_intrinsics(view2, K2)
        add_depth(view1, D1)
        add_depth(view2, D2)
        add_relpose(view1, cam2_to_world=cam2, cam1_to_world=cam1)
        add_relpose(view2, cam2_to_world=cam2, cam1_to_world=cam1)

        preds = self.model(view1, view2)

        if self.fix_rays:
            if K1 is not None:
                pred_depth = preds[0]['pts3d'][...,2]
                gt_rays = view1['known_rays'].permute(0,2,3,1)
                preds[0]['pts3d'] = gt_rays * pred_depth.unsqueeze(-1)
            if K2 is not None:
                pred_depth = preds[1]['pts3d2'][...,2]
                gt_rays = view2['known_rays'].permute(0,2,3,1)
                preds[1]['pts3d2'] = gt_rays * pred_depth.unsqueeze(-1)
            if self.fix_rays == 'full':
                fix_pts3d_in_other_view(preds, K2=K2, cam1=cam1, cam2=cam2)

        if ret_views:
            return (view1, view2), preds
        return preds

    def auto_inference_with_info(self, view1, view2):
        # pick whatever is available in the views
        return self.inference_with_info(view1, view2, 
                                        K1 = view1.get('camera_intrinsics'),
                                        K2 = view2.get('camera_intrinsics'),
                                        cam1 = view1.get('camera_pose'),
                                        cam2 = view2.get('camera_pose'),
                                        D1 = view1.get('depthmap'),
                                        D2 = view2.get('depthmap'))

    def downscale_small_side(self, view, new_size, round16=False, **kw):
        H, W = view['img'].shape[-2:]
        round16 = round_to_nearest_multiple_of_16 if round16 else (lambda x:x)
        if W < H: # smaller side is width
            W2 = new_size
            H2 = round16(H * new_size // W)
        else:
            W2 = round16(W * new_size // H)
            H2 = new_size
        return self._downscale(view, H2, W2, **kw)

    def downscale_long_side(self, view, new_size, round16=True, **kw):
        H, W = view['img'].shape[-2:]
        round16 = round_to_nearest_multiple_of_16 if round16 else (lambda x:x)
        if W < H: # smaller side is width
            W2 = round16(W * new_size // H)
            H2 = new_size
        else:
            W2 = new_size
            H2 = round16(H * new_size // W)
        return self._downscale(view, H2, W2, **kw)

    def _downscale(self, view, H2, W2, include='KPD'):
        img = view['img']
        assert img.ndim == 4 and img.shape[1] == 3
        H, W = img.shape[-2:]

        if (H,W) == (H2,W2): # nothing to do!
            return deepcopy(view)

        small_img = F.interpolate(img, size=(H2, W2), mode='bicubic', align_corners=False)
        res = dict(
                img = small_img, 
                true_shape = torch.tensor([(H2,W2)]*len(small_img)), 
                instance = view['instance'],
            )
        if 'K' in include and 'camera_intrinsics' in view:
            new_K = scale_K(view['camera_intrinsics'], (H,W), (H2,W2))
            res['camera_intrinsics'] = new_K

        if 'P' in include and 'camera_pose' in view:
            res['camera_pose'] = view['camera_pose']

        if 'D' in include and 'depthmap' in view:
            assert view['depthmap'].ndim == 3
            res['depthmap'] = F.interpolate(view['depthmap'].unsqueeze(1), size=(H2, W2), mode='nearest').squeeze(1)
            b,y,x = view['depthmap'].nonzero().T
            x2 = x * W2 // W
            y2 = y * H2 // H
            res['depthmap'][b,y2,x2] = view['depthmap'][b,y,x]

        return res

    def lame_upscale(self, view, H2, W2, include='KPD'):
        res = {}
        for key, val in view.items():
            if key.startswith('known'):
                continue # privileged information was only valid before rescaling

            densemap = None
            if 'pts3d' in key:
                densemap = val.permute(0,3,1,2)
            elif ('conf' in key or (key == 'depthmap' and 'D' in include)):
                assert val.ndim == 3
                densemap = val.unsqueeze(1)
            elif 'K' in include and key == 'camera_intrinsics':
                val = scale_K(val, view['img'].shape[-2:], (H2, W2))
            elif 'P' in include and key == 'camera_pose':
                val = view['camera_pose']

            if densemap is not None:
                mode = 'nearest' if 'depth' in key else 'bilinear' 
                densemap = F.interpolate(densemap, size=(H2, W2), mode=mode, align_corners=False)

                if 'pts3d' in key:
                    val = densemap.permute(0,2,3,1)
                if ('conf' in key or key == 'depthmap'):
                    val = densemap.squeeze(1)

            res[key] = val
        return res


class FakeResolver (BaseResolver):
    """ Just compute the result in low resolution and then upscale it
    """
    def forward(self, fine_view1, fine_view2):
        self.print_auxiliary_info(fine_view1, fine_view2)
        # coarse pass
        coarse_view1, coarse_view2 = [self.downscale_long_side(v,max(self.crop_resolution)) for v in (fine_view1, fine_view2)]
        coarse_pred1, coarse_pred2 = self.auto_inference_with_info(coarse_view1, coarse_view2)

        fine_pred1 = self.lame_upscale(coarse_pred1, *fine_view1['img'].shape[-2:])
        fine_pred2 = self.lame_upscale(coarse_pred2, *fine_view2['img'].shape[-2:])
        # viz_pred((fine_view1, fine_view2), (fine_pred1, fine_pred2))
        return fine_pred1, fine_pred2


class SlidingResolver1D (BaseResolver):
    """ Useful for KITTI image pairs, which have crappy aspect-ratio.
    """
    def __init__(self, *args, overlap_ratio=0.5, conf_mode='blend', **kwargs ):
        super().__init__(*args, **kwargs)
        assert 0 < overlap_ratio < 1, "overlap_ratio must be between 0 and 1"
        assert conf_mode in 'blend best'.split()
        self.overlap_ratio = overlap_ratio
        self.conf_mode = conf_mode

    def _to_crops(self, pos):
        crops = []
        for x,y in pos:
            crop = (slice(y,y+self.crop_resolution[0]), slice(x,x+self.crop_resolution[1]))
            crops.append(crop)
        return crops

    def get_tile_pos(self, view1, view2, pred1=None, pred2=None):
        true_shape = view1['true_shape']
        assert (true_shape[0] == true_shape).all(), 'all images in the batch must have the same size'
        assert (true_shape == view2['true_shape']).all(), 'view1 and view2 must have the same size'
        H, W = true_shape[0].tolist()
        
        if W > self.crop_resolution[0]:
            pos = calculate_window_positions(W, max(self.crop_resolution), self.overlap_ratio)
            pos = sort_from_center(pos)
            pos = [(x,0) for x in pos.round().astype(int).tolist()]
        else:
            pos = calculate_window_positions(H, max(self.crop_resolution), self.overlap_ratio)
            pos = sort_from_center(pos)
            pos = [(0,y) for y in pos.round().astype(int).tolist()]

        return list(zip(self._to_crops(pos), self._to_crops(pos)))

    def pad(self, view, crop, pred):
        B, THREE, H, W = view['img'].shape
        assert THREE == 3
        sy, sx = crop
        res = dict()

        for pts3d, conf in [('pts3d','conf'), ('pts3d2','conf2'), ('pts3d_in_other_view','conf')]:
            if pts3d not in pred or conf not in pred: 
                continue
            
            res_pts3d = torch.zeros((B,H,W,3), device=self.device)
            res_conf = torch.zeros((B,H,W), device=self.device)

            res_pts3d[:,sy,sx] = pred[pts3d]
            res_conf[:,sy,sx] = pred[conf]

            res[pts3d] = res_pts3d
            res[conf] = res_conf

        return res

    def merge_overlapping(self, cur, pred, scale_f=None):
        if cur is None:
            return pred, 1

        res = {}
        for pts3d, conf in [('pts3d','conf'), ('pts3d_in_other_view','conf'), ('pts3d2','conf2')]:
            if pts3d not in pred or conf not in pred: 
                continue

            if scale_f is None:
                avg_conf = (zero2nan(cur[conf]) + zero2nan(pred[conf])) / 2
                conf_thr = torch.nanmedian(avg_conf.view(len(avg_conf),-1), dim=-1).values

                div = zero2nan(cur[pts3d][..., 2]) / zero2nan(pred[pts3d][..., 2])
                div[avg_conf < conf_thr.view(-1,1,1)] = np.nan # remove low confidence areas
                scale_f = torch.nanmedian(div.view(len(div),-1), dim=1).values.view(-1,1,1,1)

            if self.conf_mode == 'blend':
                # the overlap and blend
                denom = (cur[conf] + pred[conf])[...,None].clip(min=1e-8)
                blend = lambda a,b: (a * cur[conf][...,None] + b * pred[conf][...,None]) / denom
            elif self.conf_mode == 'best':
                best = cur[conf] > pred[conf]
                blend = lambda a,b: torch.where(best.unsqueeze(-1), a, b)

            res[pts3d] = blend(cur[pts3d], scale_f * pred[pts3d])
            res[conf] = blend(cur[conf][...,None], pred[conf][...,None])[...,0]
            scale_f = None # reset for pts3d2

        return res, scale_f

    def forward_tiles(self, fine1, fine2, tiles):
        depth1 = get_depth_or_known_depth(fine1)
        depth2 = get_depth_or_known_depth(fine2)
        cam1 = fine1.get('camera_pose')
        cam2 = fine2.get('camera_pose')

        result1 = result2 = None

        for crop1, crop2 in tqdm(tiles):
            # crop the two view exactly the same
            cropped_view1, crop_K1, crop_D1 = crop_view(fine1, crop1, fine1['camera_intrinsics'], depth1)
            cropped_view2, crop_K2, crop_D2 = crop_view(fine2, crop2, fine2['camera_intrinsics'], depth2)

            preds = self.inference_with_info(
                cropped_view1, cropped_view2, 
                K1 = crop_K1,
                K2 = crop_K2,
                D1 = crop_D1,
                D2 = crop_D2,
                cam1 = cam1,
                cam2 = cam2,
                )
            # viz_pred((cropped_view1, cropped_view2), preds)
            save_tmp(f'window-{crop1[0].start}-{crop1[1].start}-{crop2[0].start}-{crop2[1].start}', (cropped_view1, cropped_view2), preds, crop1=crop1, crop2=crop2)

            pred1 = self.pad(fine1, crop1, preds[0])
            pred2 = self.pad(fine2, crop2, preds[1])

            # align with the same scale
            result1, scale_f = self.merge_overlapping(result1, pred1)
            result2, scale_f = self.merge_overlapping(result2, pred2, scale_f=scale_f)
            # viz_pred((fine1, ), (result1, ))
            # viz_pred((fine2, ), (result2, ))
            # viz_pred((fine1, fine2), (result1, result2))

        if self.fix_rays == 'full':
            fix_pts3d_in_other_view((result1, result2), K2=fine2['camera_intrinsics'], cam1=cam1, cam2=cam2)
        return result1, result2

    def forward(self, fine1, fine2):
        self.print_auxiliary_info(fine1, fine2)
        assert 'camera_intrinsics' in fine1
        assert 'camera_intrinsics' in fine2
        assert fine1['img'].shape == fine2['img'].shape

        coarse1 = self.downscale_small_side(fine1, min(self.crop_resolution))
        coarse2 = self.downscale_small_side(fine2, min(self.crop_resolution))

        tiles = self.get_tile_pos(coarse1, coarse2)
        preds = self.forward_tiles(coarse1, coarse2, tiles)
        return preds


class SlidingResolver (SlidingResolver1D):
    """ Same as sliding 1D, but does slide in 2D without resizing anything
    """
    def get_tile_pos(self, view1, view2, pred1=None, pred2=None):
        true_shape = view1['true_shape']
        assert (true_shape[0] == true_shape).all(), 'all images in the batch must have the same size'
        assert (true_shape == view2['true_shape']).all(), 'view1 and view2 must have the same size'
        H, W = true_shape[0].tolist()

        posx = calculate_window_positions(W, self.crop_resolution[1], self.overlap_ratio)
        posx = sort_from_center(posx)

        posy = calculate_window_positions(H, self.crop_resolution[0], self.overlap_ratio)
        posy = sort_from_center(posy)

        pos = [(x,y) for y in posy.round().astype(int).tolist() for x in posx.round().astype(int).tolist()]
        return list(zip(self._to_crops(pos), self._to_crops(pos)))

    def forward(self, fine1, fine2):
        self.print_auxiliary_info(fine1, fine2)
        assert 'camera_intrinsics' in fine1
        assert 'camera_intrinsics' in fine2

        tiles = self.get_tile_pos(fine1, fine2, None, None)
        preds = self.forward_tiles(fine1, fine2, tiles)
        return preds


class CoarseToFine (SlidingResolver):
    """ Same as sliding window, except it relies on coarse-to-fine guidance
    """
    def __init__(self, *args, sparsify_depth=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsify_depth = sparsify_depth

    def get_depth(self, coarse):
        try:
            depth = coarse['pts3d'][...,2]
            conf = coarse['conf']
        except KeyError:
            depth = coarse['pts3d2'][...,2]
            conf = coarse['conf2']

        if self.sparsify_depth < 0:
            # dense depth
            return depth 
        else:
            # sparsify depth by computing local maxima in the confidence
            local_maxima = tuple(find_local_maxima(conf, thr=self.sparsify_depth).T)
            sp_depth = torch.zeros_like(depth)
            sp_depth[local_maxima] = depth[local_maxima]
            return sp_depth

    def transfer_guidance(self, coarse, fine):
        H, W = fine['img'].shape[-2:]
        upscaled_coarse = self.lame_upscale(coarse, H, W)
        fine.setdefault('depthmap', self.get_depth(upscaled_coarse))

        res = dict(
            img=fine['img'], 
            true_shape=fine['true_shape'],
            instance = fine['instance']
            )
        for key in 'camera_intrinsics camera_pose depthmap'.split():
            if key in fine:
                res[key] = fine[key]

        if 'camera_intrinsics' not in res:
            coarse_pts3d = upscaled_coarse['pts3d']
            B, H, W, THREE = coarse_pts3d.shape
            assert B == 1 and THREE == 3
            K = torch.eye(3, device=coarse_pts3d.device)[None]
            K[:,0,2] = W/2
            K[:,1,2] = H/2
            focal = estimate_focal_knowing_depth(coarse_pts3d, K[:,:2,2], 'weiszfeld')
            K[:,0,0] = focal
            K[:,1,1] = focal
            res['camera_intrinsics'] = K

        return res

    def forward(self, fine_view1, fine_view2):
        self.print_auxiliary_info(fine_view1, fine_view2)
        
        # coarse pass
        self.print_auxiliary_info(fine_view1, fine_view2)
        coarse_view1, coarse_view2 = [self.downscale_long_side(v,max(self.crop_resolution)) for v in (fine_view1, fine_view2)]
        coarse_preds = self.auto_inference_with_info(coarse_view1, coarse_view2)
        # viz_pred((coarse_view1, coarse_view2), coarse_preds)
        save_tmp('coarse', (coarse_view1, coarse_view2), coarse_preds)

        # transfer coarse info into fine guidance
        fine_view1 = self.transfer_guidance(coarse_preds[0], fine_view1)
        fine_view2 = self.transfer_guidance(coarse_preds[1], fine_view2)

        # fine pass
        tiles = self.get_tile_pos(fine_view1, fine_view2, *coarse_preds)
        preds = self.forward_tiles(fine_view1, fine_view2, tiles)
        save_tmp('fine', (fine_view1, fine_view2), preds)
        return preds


class CoarseToFine1D (CoarseToFine):
    def forward(self, fine_view1, fine_view2):
        self.print_auxiliary_info(fine_view1, fine_view2)

        # coarse pass
        coarse_view1, coarse_view2 = [self.downscale_long_side(v,max(self.crop_resolution)) for v in (fine_view1, fine_view2)]
        coarse_preds = self.auto_inference_with_info(coarse_view1, coarse_view2)

        # transfer coarse info into fine guidance
        assert False # TODO
        fine_view1 = self.transfer_guidance(coarse_preds[0], fine_view1)
        fine_view2 = self.transfer_guidance(coarse_preds[1], fine_view2)

        # fine pass
        preds = self.forward_tiles(fine_view1, fine_view2)
        return preds[0], None


class AsymmetricSliding (CoarseToFine):
    """ Sliding window for view1 <--> coarse view for view2
    """
    def __init__(self, *args, bootstrap_depth=True, **kwargs):
        super().__init__(*args, **kwargs)
        assert bootstrap_depth in (False, True, 'c2f_fine1', 'c2f_coarse2', 'c2f_both')
        if bootstrap_depth == 'c2f_both':
            bootstrap_depth = 'c2f_fine1+coarse2'
        self.bootstrap_depth = bootstrap_depth

    def forward_tiles(self, fine1, coarse2, tiles):
        depth1 = get_depth_or_known_depth(fine1)
        depth2 = get_depth_or_known_depth(coarse2)

        result1 = result2 = None

        for crop1, crop2 in tqdm(tiles):
            cropped_view1, crop_K1, crop_D1 = crop_view(fine1, crop1, fine1['camera_intrinsics'], depth1)

            preds = self.inference_with_info(
                cropped_view1, coarse2, 
                K1 = crop_K1,
                K2 = coarse2.get('camera_intrinsics'),
                D1 = crop_D1,
                D2 = depth2,
                cam1 = fine1.get('camera_pose'),
                cam2 = coarse2.get('camera_pose'),
                )
            # viz_pred((cropped_view1, coarse2), preds)
            save_tmp(f'window-{crop1[0].start}-{crop1[1].start}-{crop2[0].start}-{crop2[1].start}', (cropped_view1, None), preds, crop1=crop1, crop2=crop2)

            # align with the same scale
            pred1 = self.pad(fine1, crop1, preds[0])
            result1, scalef = self.merge_overlapping(result1, pred1)
            result2, scalef = self.merge_overlapping(result2, preds[1], scale_f=scalef)

            if self.bootstrap_depth is True: # update guidance!
                depth1 = result1['pts3d'][...,2]

        return result1, result2

    def forward(self, fine_view1, fine_view2):
        self.print_auxiliary_info(fine_view1, fine_view2)

        # coarse pass?
        coarse_view1, coarse_view2 = [self.downscale_long_side(v,max(self.crop_resolution)) for v in (fine_view1, fine_view2)]

        if isinstance(self.bootstrap_depth, str) and 'c2f' in self.bootstrap_depth:
            coarse_preds = self.auto_inference_with_info(coarse_view1, coarse_view2)
            # viz_pred((coarse_view1, coarse_view2),coarse_preds)
            save_tmp('coarse', (coarse_view1, coarse_view2), coarse_preds)

            if 'fine1' in self.bootstrap_depth:
                fine_view1 = self.transfer_guidance(coarse_preds[0], fine_view1)

            if 'coarse2' in self.bootstrap_depth:
                add_depth(coarse_view2, self.get_depth(coarse_preds[1]))

        # fine pass
        tiles = self.get_tile_pos(fine_view1, fine_view1)
        pred1, pred2 = self.forward_tiles(fine_view1, coarse_view2, tiles)
        pred2 = self.lame_upscale(pred2, *fine_view2['img'].shape[-2:])

        save_tmp('fine', (fine_view1, fine_view2), (pred1, pred2))
        return pred1, pred2


def scale_K( K, shape_before, shape_after ):
    res = K.clone() if isinstance(K, torch.Tensor) else K.copy()
    sy, sx = np.float32(shape_after) / shape_before
    res[...,0,:] *= sx
    res[...,1,:] *= sy
    return res


def iter_views(views, device='numpy'):
    if device:
        views = todevice(views, device)
    assert views['img'].ndim == 4
    B = len(views['img'])
    for i in range(B):
        view = {k:(v[i] if isinstance(v, (np.ndarray,torch.Tensor)) else v) for k,v in views.items()}
        yield view

def add_intrinsics(view, K):
    if K is not None:
        device = 'numpy' if isinstance(K,np.ndarray) else K.device
        view['camera_intrinsics'] = K
        if K.ndim == 2:
            known_rays = todevice(gen_rays(to_numpy(view)), device)
        else:
            known_rays = [gen_rays(v) for v in iter_views(view)] 
            known_rays = torch.stack([todevice(k,device) for k in known_rays])
        view['known_rays'] = known_rays

def add_depth(view, depthmap, npts=None):
    if depthmap is not None:
        assert depthmap.shape[-2:] == view['img'].shape[2:], breakpoint()
        depthmap = torch.nan_to_num(depthmap) # in case we have nans instead of zeros
        view['depthmap'] = depthmap
        view['valid_mask'] = (depthmap > 0)
        npts = npts or view['valid_mask'].sum().item()
        known_depth = [gen_sparse_depth(v, npts, npts) for v in iter_views(view)]
        view['known_depth'] = torch.stack([todevice(k,depthmap.device) for k in known_depth])

def add_relpose(view, cam2_to_world, cam1_to_world=None):
    if cam2_to_world is not None:
        cam1_to_world = todevice(cam1_to_world, 'numpy')
        cam2_to_world = todevice(cam2_to_world, 'numpy')
        def fake_views(i):
            return [dict(camera_pose=np.eye(4) if cam1_to_world is None else cam1_to_world[i]), 
                    dict(camera_pose=cam2_to_world[i]) ]
        if cam2_to_world.ndim == 2:
            known_pose = gen_rel_pose(fake_views(slice(None)))
        else:
            known_pose = [gen_rel_pose(fake_views(i)) for i,v in enumerate(iter_views(view))]
            known_pose = torch.stack([todevice(k, view['img'].device) for k in known_pose])
        view['known_pose'] = known_pose


def sparse_upsample(depth, H2, W2):
    assert depth.ndim == 3
    res = F.interpolate(depth.unsqueeze(1), (H2,W2), mode='bilinear', align_corners=False)
    return res.squeeze(1)


def crop_view( view, crop, K, depth=None):
    sy, sx = crop
    img_crop = view['img'][...,sy,sx]

    K_crop = K.clone()
    K_crop[...,0,2] -= sx.start
    K_crop[...,1,2] -= sy.start

    view = dict(view, # copy and modify the image
        img = img_crop, 
        camera_intrinsics = K_crop,
        )
    view['true_shape'] = torch.tensor([img_crop.shape[-2:]]*len(img_crop))

    if depth is not None:
        depth_crop = depth[...,sy,sx]
    else:
        depth_crop = None

    return view, K_crop, depth_crop


def get_depth_or_known_depth(view):
    try:
        return view['depthmap']
    except KeyError:
        try:
            return view['known_depth'][:,0,:,:]
        except KeyError:
            return None

def find_local_maxima(conf, thr=0):
    # Apply max pooling with a 3x3 kernel and stride 1 to find local maxima
    max_pool = F.max_pool2d(conf.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)

    # Find where the values in conf are equal to the max pooled values
    local_max = (conf == max_pool)
    
    # Apply threshold mask
    threshold_mask = (conf > thr)
    
    # Combine the local maxima mask and the threshold mask
    output = local_max & threshold_mask
    
    # Convert boolean mask to float for consistency with the input tensor
    return output.nonzero()

    
def calculate_window_positions(W, window_size, overlap_ratio):
    # Calculate the step size based on overlap ratio
    step_size = window_size * (1 - overlap_ratio)
    
    # Calculate the total number of windows needed to cover [0, W]
    num_windows = int(np.ceil((W - window_size) / step_size)) + 1
    
    # Generate evenly spaced positions using np.linspace
    positions = np.linspace(0, W - window_size, num_windows)
    return positions


def sort_from_center( pos ):
    center = pos.mean()
    order = np.abs(pos - center).argsort()
    return pos[order]

def zero2nan( arr ):
    arr = arr.clone()
    arr[arr==0] = np.nan
    return arr

def round_to_nearest_multiple_of_16( n ):
    assert n > 0
    return int(n/16 + 0.5)*16

def fix_pts3d_in_other_view(preds, K2=None, cam1=None, cam2=None):
    pts22 = preds[1]['pts3d2']
    conf22 = preds[1]['conf2']
    pts21 = preds[1]['pts3d_in_other_view']
    conf21 = preds[1]['conf']
    B = len(pts21)

    if cam2 is not None:
        if cam1 is None: 
            cam1 = torch.eye(4)[None].expend(B,4,4)

        # find scale of relative pose
        cam1_to_cam2 = inv(cam2) @ cam1
        H, W = pts21.shape[-3:-1]
        pix2 = xy_grid(W, H, device=pts21.device).view(1,H*W,2).expand(B,H*W,2)
        scale_f = estimate_scene_scale( cam1_to_cam2, pts21.view(B,H*W,3), pix2, K2 )
        cam1_to_cam2[..., :3, 3] *= scale_f[...,None]

        # find scale of X22
        pts21_in_cam2 = geotrf(cam1_to_cam2, pts21)
        scale_X22 = find_opt_scaling(pts22, None, pts21_in_cam2, fit_mode='median')

        # project pts22 to pts21
        cam2_to_cam1 = inv(cam1_to_cam2)
        new_pts21 = geotrf(cam2_to_cam1, pts22 * scale_X22.view(B,1,1,1))

        preds[1]['pts3d_in_other_view'] = new_pts21           
    else:
        # manual estimation by comparing X21 and X22
        R, T, s = roma.rigid_points_registration(
                    pts22.reshape(B, -1, 3),
                    pts21.reshape(B, -1, 3),
                    weights = (conf21 * conf22).sqrt().reshape(B, -1), 
                    compute_scaling = True)

        fixed = s * (pts22.reshape(B, -1, 3) @ R.permute(0,2,1)) + T
        preds[1]['pts3d_in_other_view'] = fixed.view(pts21.shape)


@torch.no_grad()
def viz_pred(views, preds, gt2=None, idx=0): #TODO remove
    from matplotlib import pyplot as pl; pl.ion()
    from recon.cloud_opt.viz import rgb, SceneViz

    if isinstance(views, dict):
        views = [views]
        preds = [preds]

    pl.figure('viz')
    viz = SceneViz()

    for i, (view, pred) in enumerate(zip(views, preds)):
        img = rgb(view['img'][idx])
        pl.subplot(len(views), 1, i+1).imshow(img)

        pred = pred.get('pts3d', pred.get('pts3d_in_other_view', pred.get('pts3d2')))[idx]
        viz.add_pointcloud(pred, img)

    viz.show()
    breakpoint()


def show_crops(view1, view2, crop1, crop2): # TODO remove
    from matplotlib import pyplot as pl; pl.ion()
    from recon.cloud_opt.viz import rgb
    def plot_rect(crop, **kw):
        (t,b),(l,r) = crop
        pl.plot([l,r,r,l,l],[t,t,b,b,t],**kw)
    pl.ion()
    pl.clf()
    pl.subplot(211).imshow(rgb(view1['img'][0]))
    plot_rect(crop1, color='r', lw=4)
    pl.subplot(212).imshow(rgb(view2['img'][0]))
    plot_rect(crop2, color='b', lw=4)
    breakpoint()


DUMP_TMP_FOLDER = None
def save_tmp(name, views, preds, **kw):
    if DUMP_TMP_FOLDER:
        np.savez_compressed(DUMP_TMP_FOLDER+'/'+name, views=to_numpy(views), preds=to_numpy(preds), **to_numpy(kw))
