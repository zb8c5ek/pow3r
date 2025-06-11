# Copyright (C) 2025-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import torch
import tempfile
import matplotlib.pyplot as pl

import pow3r.tools.path_to_dust3r
from dust3r.utils.device import todevice, to_numpy
from dust3r.utils.image import load_images
from dust3r.viz import rgb, SceneViz

from pow3r.model import inference as supr
from pow3r.tools.video_maker import (pyrender, Image, MovieMaker, CameraPose, 
    PointCloud, Text, Arrow, RoundedBoxWithText, ResizedImage, 
    ScreenShift, ScreenTrf, Rectangle, linear, smooth)


def extract_data(
            img1 = 'assets/img1.jpg',
            img2 = 'assets/img2.jpg',
            ckpt_path = "checkpoints/Pow3R_ViTLarge_BaseDecoder_512_linear.pth",
            resolution = 1280,
            device = 'cuda'):

    tmp_dir = tempfile.gettempdir()
    supr.DUMP_TMP_FOLDER = os.path.join(tmp_dir, 'pow3r_demo', img1, img2)
    os.makedirs(supr.DUMP_TMP_FOLDER, exist_ok=True)
    print('saving temporary stuff in', supr.DUMP_TMP_FOLDER)
    if osp.isfile(osp.join(supr.DUMP_TMP_FOLDER,'done')):
        return supr.DUMP_TMP_FOLDER
    
    # load images
    imgs_hd = load_images([img1, img2], size=resolution)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    crop_res = (384,512)

    view1, view2 = todevice(imgs_hd, device)

    with torch.no_grad():
        try:
            pred1, pred2 = np.load(supr.DUMP_TMP_FOLDER+'fine.npz', allow_pickle=True)['preds']

        except IOError:
            slider = supr.AsymmetricSliding(crop_res, bootstrap_depth='c2f_both', fix_rays='full', sparsify_depth=1.1)
            slider.load_from_checkpoint(ckpt)
            slider = slider.to(device)

            pred1, pred2 = to_numpy(slider(view1, view2))

    open(osp.join(supr.DUMP_TMP_FOLDER,'done'),'w')
    return supr.DUMP_TMP_FOLDER


def sliding_video(root, width=1280, height=960, generate_frames=False, video_dir='.', show_preds=False):
    # load all files
    coarse = np.load(osp.join(root,'coarse.npz'), allow_pickle=True)
    # show_preds_3d(coarse['views'], coarse['preds'])

    fine = dict(np.load(osp.join(root,'fine.npz'), allow_pickle=True))
    if show_preds:
        show_preds_3d(fine['views'], fine['preds'])

    crops = {}
    for cropf in os.listdir(root):
        if not cropf.startswith('window'): continue
        y1, x1, y2, x2 = map(int, cropf[:-4].split('-')[1:])
        crops[x1,y1] = np.load(osp.join(root,cropf), allow_pickle=True)

    # sort them so that it starts with middle crops
    middle = np.mean(list(crops))
    crops = {xy:crops[xy] for xy in sorted(crops, key=lambda xy: np.linalg.norm(xy - middle))}

    # make a moving camera
    cam = CameraPose(motion=lambda t: np.r_[np.cos(t),np.sin(t),0.5], 
            spatial_scale=0.5, time_scale=1, 
            look_at=(0,0,2), up=(0,-1,0))

    # start adding clips
    mm = MovieMaker(cam, render_flags=pyrender.RenderFlags.FLAT)

    mm.add_segment(Text(4, (0.5,0.5), 'Coarse-to-fine\nHigh-resolution Binocular\n3D reconstruction with Pow3R', fontsize=50))

    # images appear big and get smaller
    t20 = 1000
    img1 = RGB8(fine['views'][0]['img'])
    img2 = RGB8(fine['views'][1]['img'])
    t1, t2 = 1,2
    small1, small2 = np.array((0.12,0.25)), np.array((0.12,0.65))
    scale = linear(0.5, 0.2, t2, t1)
    mm.add_segment(ResizedImage(t20, img1, next=0, xy=smooth((0.5,0.25), small1, t2, t1), scale=scale, border_color='r', border_width=5))
    mm.add_segment(ResizedImage(t20, img2, next=3, xy=smooth((0.5,0.75), small2, t2, t1), scale=scale, border_color='b', border_width=5))

    # adding intrinsics and power
    x = 0.35
    hopt = dict(head_width=20, head_length=20)
    mm.add_segment(Text(t20, small1+(0.1,0.15), 'Intrinsics $K^1$', next=0.5, color='k', fontsize=25, va='top', ha='right'))
    mm.add_segment(Text(t20, small2+(0.1,0.15), 'Intrinsics $K^2$', next=0.5, color='k', fontsize=25, va='top', ha='right'))
    mm.add_segment(Arrow(t20, small1+(0.1, 0.05), (x-0.02,0.39), next=0, **hopt))
    mm.add_segment(Arrow(t20, small1+(0.1, 0.16), (x-0.02,0.44), next=0, **hopt))
    mm.add_segment(Arrow(t20, small2+(0.1,-0.05), (x-0.02,0.54), next=0, **hopt))
    mm.add_segment(Arrow(t20, small2+(0.1, 0.15), (x-0.02,0.62), next=0, **hopt))
    mm.add_segment(Text(t20, (x+0.1,0.2), 'Coarse pass', next=0, fontsize=30))
    mm.add_segment(RoundedBoxWithText(t20, (x,0.4,x+0.2,0.6), 'Pow3R', next=0.5, fontsize=40, rounding_size=30, box_pad=0, facecolor='lightgray'))

    # add coarse output
    pts11 = coarse['preds'][0]['pts3d']
    pts21 = coarse['preds'][1]['pts3d_in_other_view']
    pts22 = coarse['preds'][1]['pts3d_in_other_view']
    col1 = RGB8(coarse['views'][0]['img'])
    col2 = RGB8(coarse['views'][1]['img'])
    normed_pts11 = norm_ptcloud(pts11, 0.3, depth=2)
    normed_pts21 = norm_ptcloud(pts21, 0.3, depth=2)
    depth1 = np.uint8(255*pl.cm.viridis(between0and1(pts11[0,:,:,2])))[:,:,:3]
    depth2 = np.uint8(255*pl.cm.viridis(between0and1(pts22[0,:,:,2])))[:,:,:3]

    ss = mm.add_segment(ScreenShift(t20, (0.3,0), next=0, priority=True)) # for pushing the 3d viz to the right
    p1 = mm.add_segment(PointCloud(t20, next=0, pts=smooth(pts11/1.7 + (0,0,1.5), normed_pts11-(0,0.4,0), 8, 6), colors=linear(col1,col1//2+np.uint8((127,0,0)), 6, 3)))
    mm.add_segment(Arrow(t20, (x+0.2,0.5), smooth((x+0.25,0.5), (x+0.25,0.43), 8, 6), next=3, **hopt))
    t1 = mm.add_segment(Text((3,t20), (0.8,0.15), text="  Predicted\npointmap $X^{1,1}$", color='r', fontsize=25, fontweight='bold', next=0))
    p2 = mm.add_segment(PointCloud(t20, next=0, pts=smooth(pts21/1.7 + (0,0,1.5), normed_pts21+(0,0.4,0), 8, 6), colors=linear(col2,col2//2+np.uint8((0,0,127)), 6, 3)), prev=ss)
    mm.add_segment(Arrow(t20, (x+0.2,0.5), smooth((x+0.25,0.5), (x+0.25,0.57), 8, 6), next=3, **hopt))
    t2 = mm.add_segment(Text((3,t20), (0.8,0.85), next=5, text="      Predicted\npointmaps $X^{2,1}, X^{2,2}$", color='b', fontsize=25, fontweight='bold'))
    ss2 = mm.add_segment(ScreenShift(t20, smooth((0,0), (-0.62,0), 2), next=2))
    mm.clip_segments(ignore={p1,p2,ss,ss2,t1,t2})

    # coarse becomes depthmaps
    y1, y2 = 0.43, 0.83
    mm.add_segment(Arrow(t20, (0.4, y1), (0.6, y1), next=0))
    mm.add_segment(Text(t20, (0.5, y1), "Depthmap $D^1$", next=0.5, fontsize=20, va="bottom"))
    mm.add_segment(ResizedImage(t20, depth1, next=0.5, xy=(0.8,y1), scale=0.2))
    mm.add_segment(Arrow(t20, (0.4, y2), (0.6, y2), next=0))
    mm.add_segment(Text(t20, (0.5, y2), "Depthmap $D^2$", next=0.5, fontsize=20, va="bottom"))
    mm.add_segment(ResizedImage(t20, depth2, next=4, xy=(0.8,y2), scale=0.2))
    mm.add_segment(ScreenShift(2, smooth((0,0), (-0.68,0), 2)))
    mm.clip_segments()

    d1 = mm.add_segment(ResizedImage(t20, depth1, next=0, xy=(0.12,y1), scale=0.2))
    d2 = mm.add_segment(ResizedImage(t20, depth2, next=0, xy=(0.12,y2), scale=0.2))
    i1 = mm.add_segment(ResizedImage(t20, img1, next=0, xy=small1, scale=0.2))
    i2 = mm.add_segment(ResizedImage(t20, img2, next=0, xy=small2, scale=0.2))
    mm.add_segment(Text(t20, (0.2,0.12), 'Intrinsics K1', next=0, color='k', ha='right', fontsize=25))
    mm.add_segment(Text(t20, (0.2,0.96), 'Intrinsics K2', next=0, color='k', ha='right', fontsize=25))

    mm.add_segment(Arrow(t20, small1+(0.1,-0.1), (x-0.02,0.39), next=0, **hopt))
    mm.add_segment(Arrow(t20, small1+(0.1, 0.05), (x-0.02,0.44), next=0, **hopt))
    mm.add_segment(Arrow(t20, small1+(0.1, 0.16), (x-0.02,0.49), next=0, **hopt))
    mm.add_segment(Arrow(t20, small2+(0.1,-0.05), (x-0.02,0.54), next=0, **hopt))
    mm.add_segment(Arrow(t20, small2+(0.1, 0.15), (x-0.02,0.58), next=0, **hopt))
    mm.add_segment(Arrow(t20, small2+(0.1, 0.30), (x-0.02,0.62), next=0, **hopt))
    mm.add_segment(Text(t20, (x+0.1,0.2), 'Fine pass with\na sliding window', next=0, fontsize=30))
    mm.add_segment(RoundedBoxWithText(t20, (x,0.4,x+0.2,0.6), 'Pow3R', next=0.5, fontsize=40, rounding_size=30, box_pad=0, facecolor='lightgray'))
    
    # sliding-window output
    mm.clip_segments(4.5, ignore={d1,d2,i1,i2})
    T = 60
    mm.add_segment(ScreenTrf(T, next=0, xy=smooth((0.75,0.5), (0.6,0.5), 6, 3), scale=smooth(0.5, 1.2, 6, 3), priority=True))
    mm.add_segment(Text(T, next=0, xy=(0.995,0.995), color=(0.2,0.2,0.2), va='bottom', ha='right', text='\n'.join(l.strip() for l in """
        Pointmaps are predicted independently but conditioned by intrinsics and 
        coarse depth. Only the pointmaps' scale needs to be fixed, which we do 
        by computing the median scale factor in overlappping regions (in red).""".splitlines() if l), fontsize=18))

    def sub_bbox(seg, ltrb):
        # get outer bbox in pixels
        ref_shape = seg.img(0).shape[:2]
        bbox = seg.get_bbox(height, width, 0, ref_shape)
        bbox = np.reshape(bbox, (2,2))
        bw, bh = bbox[1] - bbox[0]

        # normalize inner bbox
        ltrb = np.reshape(ltrb, (2,2)) / (1280, 960)

        # rescale according to outer bbox
        res = ltrb * (bw, bh) + bbox[0]
        return np.int32(res.round())

    overlap = RGB8(fine['views'][0]['img'])

    for i,((x,y), crop) in enumerate(crops.items()):
        # start and end 3D points
        init_pts = crop['preds'][0]['pts3d']
        gt_pts = fine['preds'][0]['pts3d'][0][tuple(crop['crop1'])]

        # start and end 3D colors
        sy, sx = tuple(crop['crop1'])
        start_color = overlap[sy, sx].copy()
        overlap[sy, sx] = (255,0,0)
        end_color = RGB8(crop['views'][0]['img'])

        ltrb = np.float32((sx.start, sy.start, sx.stop, sy.stop))
        t = 2 - i/10
        mm.add_segment(Rectangle(t, sub_bbox(d1,ltrb), border_color='r', border_width=5, next=0))
        mm.add_segment(Rectangle(t, sub_bbox(i1,ltrb), border_color='r', border_width=5, next=0))
        mm.add_segment(Rectangle(t, sub_bbox(d2,ltrb), border_color='b', border_width=5, next=0))
        mm.add_segment(Rectangle(t, sub_bbox(i2,ltrb), border_color='b', border_width=5, next=0))
        mm.add_segment(PointCloud(T, next=t, pts=linear(init_pts, gt_pts, 1), colors=linear(start_color, end_color, 1)))

    mm.clip_segments(10)

    if generate_frames:
        os.system(f'rm -f {video_dir}/*.jpg')
        for i,frame in enumerate(mm.generate(width, height, fps=25, point_size=1)):
            Image.fromarray(frame).save(osp.join(video_dir, f'frame-{i:05d}.jpg'), quality=95)
    else:
        pl.ion()
        fig, ax = pl.subplots(1,1, num='viz')
        pl.subplots_adjust(0,0,1,1)
        for frame in mm.generate(width, height, fps=5, point_size=1, expedite_pyplot=True):
            ax.cla()
            ax.imshow(frame)
            pl.pause(0.001)


def RGB8(img): 
    img = rgb(img.squeeze())
    assert img.ndim == 3, breakpoint()
    img = np.uint8(255*img + 0.5)
    return img

def norm_ptcloud( pts, factor=1, depth=1 ): 
    scale = factor / np.sqrt(pts.reshape(-1,3).var(axis=0).sum())
    pts = pts * scale
    pts[...,2] += depth - pts[...,2].mean()
    return pts

def between0and1( x ): 
    x = x - x.min()
    x /= x.max() 
    return x


def show_preds_3d(views, preds, max_grad=0.01, **kw): 
    views = to_numpy(views)
    preds = to_numpy(preds)

    def grad(pts3d):
        depth = pts3d[...,2]
        grad = np.gradient(depth)
        grad = np.sqrt(grad[0]**2 + grad[1]**2)
        return grad / depth

    viz = SceneViz()
    for i in range(len(views)):
        img1 = rgb(views[i]['img'][0])
        pred11 = preds[i].get('pts3d', preds[i].get('pts3d_in_other_view'))[0]
        conf = preds[i]['conf']
        mask = views[i].get('valid_mask', conf>1.5)[0]
        mask &= grad(pred11) < max_grad

        viz.add_pointcloud(pred11, img1, mask=mask)
    viz.show(**kw)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Pow3R demo for high-res inference')

    parser.add_argument('--img1', type=str, default='assets/img1.jpg')
    parser.add_argument('--img2', type=str, default='assets/img2.jpg')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/Pow3R_ViTLarge_BaseDecoder_512_linear.pth')
    
    parser.add_argument('--generate_frames', action='store_true', help='flag to generate a mp4 video')
    parser.add_argument('--video_dir', type=str, default='output/')

    parser.add_argument('--show_preds', action='store_true', help='visualize 3D reconstruction before generating the video')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    root = extract_data(args.img1, args.img2, ckpt_path=args.ckpt_path)

    if args.generate_frames:
        os.makedirs(args.video_dir, exist_ok=True)

    sliding_video(root, generate_frames=args.generate_frames, video_dir=args.video_dir, show_preds=args.show_preds)

    if args.generate_frames:
        cmd = f'cd {args.video_dir}; ffmpeg -framerate 25 -i frame-%05d.jpg -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p output.mp4'
        print(cmd)
        os.system(cmd)
