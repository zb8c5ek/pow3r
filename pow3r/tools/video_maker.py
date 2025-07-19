# Copyright (C) 2025-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from PIL import Image
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation

import trimesh
import pyrender

import matplotlib.pyplot as pl
from matplotlib.colors import to_rgb as pl_to_rgb
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import pow3r.tools.path_to_dust3r
from dust3r.utils.geometry import geotrf

OPENGL_CAMERA_CONVENTION = np.eye(4,dtype=np.float32) * (1,-1,-1,1)


class MovieMaker:
    def __init__(self, camera, render_flags=0, bg_color=(255,255,255)):
        self.camera = camera
        self.segments = []
        self.render_flags = render_flags
        self.bg_color = bg_color

    def add_segment(self, segment, prev='last'):
        assert isinstance(segment, Segment)
        if prev == 'last':
            prev = self.segments[-1] if self.segments else None
        segment.prev = prev
        self.segments.append(segment)
        return segment

    def clip_segments(self, time=0, ignore=()):
        assert self.segments, "can't clip if there's no segments"
        max_t = self.segments[-1].time_to_go_to_next + time
        for seg in self.segments:
            if seg in ignore: 
                continue
            start, end = seg.timing
            assert start < max_t, 'we need to delete this segment'
            if end > max_t:
                seg.end_time -= end - max_t

    @property
    def total_time(self):
        if not self.segments: return 0
        return max(seg.timing[1] for seg in self.segments)

    def find_seg(self, abs_t):
        for seg in self.segments:
            start_t, end_t = seg.timing
            if start_t <= abs_t < end_t:
                yield seg, abs_t-start_t

    @staticmethod
    def get_intrinsics(camera, image_height, image_width):
        # Compute focal lengths
        f_y = image_height / (2 * np.tan(camera.yfov / 2))
        f_x = f_y / (camera.aspectRatio or 1)
        
        # Principal point (center of the image)
        c_x = image_width / 2
        c_y = image_height / 2
        
        # Intrinsics matrix
        intrinsics = np.array([
            [f_x, 0,   c_x],
            [0,   f_y, c_y],
            [0,   0,   1]
        ])
        return intrinsics

    def render(self, renderer, scene, cam2w, yfov):
        camera = pyrender.PerspectiveCamera(np.deg2rad(yfov), znear=0.1)
        camera_node = scene.add(camera, pose=cam2w @ OPENGL_CAMERA_CONVENTION)

        light = pyrender.PointLight(color=(1,1,1), intensity=20e3)
        scene.add(light, parent_node=camera_node)

        color, depth = renderer.render(scene, flags=self.render_flags)
        K = self.get_intrinsics(camera, *depth.shape)
        return K, color, depth

    def generate(self, width, height, fps=25, point_size=4, yfov=60, expedite_pyplot=False):
        renderer = pyrender.OffscreenRenderer(
                        width, height, 
                        point_size=point_size)

        timestamps = np.arange(0, self.total_time, 1/fps)

        for t in tqdm(timestamps):
            # gen cam
            cam2w = self.camera(t)
            
            # gen frame
            scene = pyrender.Scene(bg_color=self.bg_color, ambient_light=(0,0,0))
            priority = []
            post_proc = []
            for seg, rel_t in self.find_seg(t):
                if isinstance(seg, TrimeshSegment):
                    seg.generate(scene, rel_t)
                elif seg.priority:
                    priority.append((seg, rel_t))
                else:
                    post_proc.append((seg, rel_t))

            K, frame, depth = self.render(renderer, scene, cam2w, yfov)

            # post-processing with 2d effects
            cam2screen = K @ cam2w[:3]
            for seg, rel_t in priority+post_proc:
                frame = seg.apply(frame, rel_t, cam2screen, expedite=expedite_pyplot)

            yield frame


class CameraPose:
    def __init__(self, motion, look_at, spatial_scale=1, time_scale=1, up=(0,1,0)):
        self.motion = motion
        self.look_at = look_at
        self.spatial_scale = spatial_scale
        self.time_scale = time_scale
        self.up = up

    @staticmethod
    def make_cam_look_at(cam_pos, center=(0,0,0), up=(0,1,0)):
        z = center - cam_pos
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        y = -np.float32(up)
        y = y - np.sum(y * z, axis=-1, keepdims=True) * z
        y /= np.linalg.norm(y, axis=-1, keepdims=True)
        x = np.cross(y, z, axis=-1)

        cam2w = np.r_[np.c_[x,y,z,cam_pos],[[0,0,0,1]]]
        return cam2w

    def __call__(self, t):
        pos = self.spatial_scale * self.motion(t * self.time_scale)
        return self.make_cam_look_at(pos, self.look_at, up=self.up)


class Segment:
    def __init__(self, start_end_time, priority=False, next=None):
        if isinstance(start_end_time, (int,float)):
            start, end = 0, start_end_time
        else:
            start, end = start_end_time
        self.start_time = start
        self.end_time = end
        self._time_to_go_to_next = next
        self.prev = None
        self.priority = priority

    @property
    def total_time(self):
        return self.end_time - self.start_time

    @property
    def time_to_go_to_next(self):
        if self._time_to_go_to_next is None:
            return self.timing[1]
        else:
            return self.timing[0] + self._time_to_go_to_next 

    @property
    def timing(self):
        offset = self.prev.time_to_go_to_next if self.prev else 0
        return offset+self.start_time, offset+self.end_time


class TrimeshSegment (Segment):
    def generate(self, scene, t):
        raise NotImplementedError()


class FrameArraySegment (Segment):
    def apply(self, frame, t, cam2screen, expedite=False):
        raise NotImplementedError()

class pl_ioff:
    """Context manager to disable interactive mode."""
    def __enter__(self):
        self.was_interactive = pl.isinteractive()
        pl.ioff()  # Disable interactive mode
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.was_interactive:  # Restore previous state
            pl.ion()

class PyplotSegment (FrameArraySegment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_diff = None

    def apply(self, frame, t, cam2screen, expedite=False):
        self.shape = (height, width) = frame.shape[:2]

        if expedite:
            if self._cached_diff is None:
                frame_before = frame.copy()
            else:
                return frame ^ self._cached_diff

        with pl_ioff():
            # Create a figure with the exact dimensions of the image
            fig = pl.figure('Text',figsize=(width / 100, height / 100), dpi=100)
            pl.clf()

            ax = fig.add_axes([0, 0, 1, 1])  # Fullscreen axes (no padding)
            ax.axis('off')

            ax.imshow(frame)
            self.draw(ax, t, cam2screen)
            
            ax.set_xlim((-0.5, width-0.5))
            ax.set_ylim((height-0.5, -0.5))

            # Render the figure to a canvas 
            canvas = FigureCanvas(fig)
            canvas.draw()

            # Convert the canvas to a NumPy array
            assert (width, height) == canvas.get_width_height()
            # res = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
            res = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(height, width, 4)[:, :, :3]
            pl.close(fig)

        if expedite:
            self._cached_diff = res ^ frame_before

        if not 'check same image':
            pl.ion()
            pl.figure('debug')
            # pl.imshow(np.abs(frame - res))
            pl.imshow(res)
            breakpoint()
        return res

    def _resolve_pt(self, pt, cam2screen=None):
        if len(pt) == 2:
            x, y = pt
            if 0 < x < 1: x *= self.shape[1]
            if 0 < y < 1: y *= self.shape[0]
            return np.float32((x,y))
        elif len(pt) == 3:
            return geotrf(cam2screen, pt, ncol=2, norm=1)
        else:
            raise TypeError()

    def draw(self, ax, t, cam2screen):
        raise NotImplementedError()


class PointCloud (TrimeshSegment):
    def __init__(self, total_time, pts, colors, **kw):
        super().__init__(total_time, **kw)
        self.pts = make_temporal(pts)
        self.colors = make_temporal(colors)

    def generate(self, scene, t):
        vertices = self.pts(t).reshape(-1,3)
        colors = np.asarray(self.colors(t))
        assert colors.dtype == np.uint8
        colors = np.broadcast_to(colors.reshape(-1,3), vertices.shape)
        obj = pyrender.Mesh.from_points(vertices, colors=colors)
        scene.add( obj )


class CameraCone (TrimeshSegment):
    def __init__(self, total_time, cam2w, focal, color=(0,0,0), imsize=None, cam_size=0.1, **kw):
        super().__init__(total_time, **kw)
        self.cam2w = make_temporal(cam2w)
        self.focal = make_temporal(focal)
        self.color = np.uint8(color)
        self.imsize = imsize
        self.cam_size = make_temporal(cam_size)

    def generate(self, scene, t):
        focal = self.focal(t)
        focal = float(focal[0,0] if isinstance(focal, np.ndarray) else focal)

        # cone height and width in the 3d space
        W, H = self.imsize
        height = focal * self.cam_size(t) / H
        width = self.cam_size(t) * 0.5**0.5

        rot45 = np.eye(4)
        rot45[:3,:3] = Rotation.from_euler('z',np.deg2rad(45)).as_matrix()
        rot45[2,3] = -height # set the tip of the cone = optical center
        aspect_ratio = np.eye(4)
        aspect_ratio[0,0] = W/H
        cam2w = self.cam2w(t) @ OPENGL_CAMERA_CONVENTION
        transform = cam2w @ aspect_ratio @ rot45

        cam = trimesh.creation.cone(width, height, sections=4, transform=transform)
        cam.visual.vertex_colors = np.broadcast_to(self.color, cam.vertices.shape)
        obj = pyrender.Mesh.from_trimesh(cam)
        scene.add( obj )


class ScreenShift (FrameArraySegment):
    """ shift_xy is either pixels or relative screen coords 
    """
    def __init__(self, total_time, shift_xy, bg_color=255, **kw):
        super().__init__(total_time, **kw)
        self.shift_xy = make_temporal(shift_xy)
        self.bg_color = make_temporal(bg_color)

    def apply(self, frame, t, cam2screen, expedite=False):
        h, w = frame.shape[:2]

        dx, dy = self.shift_xy(t)
        if not(isinstance(dx, int) and isinstance(dy, int)):
            dx = round(int(dx * w))
            dy = round(int(dy * h))

        # Create an empty image with the same shape, filled with the background color
        shifted_img = np.full_like(frame, self.bg_color(t))

        # Compute slicing indices for the input and output images
        x_start = max(0, dx)
        x_end = w if dx >= 0 else w + dx
        y_start = max(0, dy)
        y_end = h if dy >= 0 else h + dy

        src_x_start = max(0, -dx)
        src_x_end = w if dx <= 0 else w - dx
        src_y_start = max(0, -dy)
        src_y_end = h if dy <= 0 else h - dy

        # Copy the overlapping region from the original image to the shifted image
        shifted_img[y_start:y_end, x_start:x_end] = frame[src_y_start:src_y_end, src_x_start:src_x_end]

        return shifted_img


class Rectangle (FrameArraySegment):
    def __init__(self, total_time, bbox_ltrb=None, xy=None, scale=None, ref_shape=None,
                 border_color='k', border_width=None, **kw):
        super().__init__(total_time, **kw)
        self.bbox_ltrb = make_temporal(bbox_ltrb)
        self.ref_shape = make_temporal(ref_shape)
        self.xy = make_temporal(xy)
        self.scale = make_temporal(scale)
        self.border_color = make_temporal(border_color)
        self.border_width = make_temporal(border_width)

    def get_bbox(self, height, width, time, ref_shape):
        if self.bbox_ltrb is None:
            x,y = self.xy(time)
            scale = self.scale(time)
            if 0 <= x <= 1 and 0 <= y <= 1:
                x = int(round(x * width))
                y = int(round(y * height))

            assert ref_shape, breakpoint()
            ih, iw = ref_shape
            if 0 <= scale <= 1.5:
                scale = int(round(scale * (width if iw>ih else height)))
            assert isinstance(x, int) and isinstance(y, int) and isinstance(scale, int), breakpoint()
            scale /= max(iw,ih)
            l, t, r, b = map(lambda v: int(round(v)), (x-scale*iw/2, y-scale*ih/2, x+scale*iw/2, y+scale*ih/2))

        else:
            l, t, r, b = np.ravel(self.bbox_ltrb(time))

        return l, t, r, b

    def apply(self, frame, time, cam2screen, expedite=False):
        if self.border_width and (bw := self.border_width(time)) > 0:
            l, t, r, b = self.get_bbox(*frame.shape[:2], time, self.ref_shape)
            rgb = tuple(int(255*v) for v in pl_to_rgb(self.border_color(time)))

            bw0 = (bw + 0) // 2
            bw1 = (bw + 1) // 2
            frame = frame.copy()
            frame[max(0,t-bw0):t+bw1,max(0,l-bw0):r+bw1] = rgb
            frame[max(0,b-bw0):b+bw1,max(0,l-bw0):r+bw1] = rgb
            frame[max(0,t-bw0):b+bw1,max(0,l-bw0):l+bw1] = rgb
            frame[max(0,t-bw0):b+bw1,max(0,r-bw0):r+bw1] = rgb
        return frame


class ResizedImage (Rectangle):
    """
        either provide:
         - bbox_ltrb: bounding box in pixels
         - xy and scale: center and scale

    If xy >= 1 is an integer: 
        center in pixels 
        scale in pixels (largest dim)
    otherwise: 
        center in [0,1], relative coordinates of (W,H)
        scale in [0,1], relative of largest image dim 
    """
    def __init__(self, total_time, img, resample=Image.LANCZOS, **kw):
        super().__init__(total_time, **kw)
        self.img = make_temporal(img)
        self.resample = resample

    def paste_img_in_frame(self, img, frame, time):
        height, width = frame.shape[:2]
        img = Image.fromarray(img)
        iw, ih = img.size
        l, t, r, b = self.get_bbox(height, width, time, (ih, iw))
        
        img = img.resize((r-l, b-t), resample=self.resample)
        img = np.asarray(img)

        frame = frame.copy()
        t2, b2 = max(0, t), min(b, height)
        l2, r2 = max(0, l), min(r, width)
        frame[t2:b2, l2:r2] = img[t2-t:b2-b or None, l2-l:r2-r or None]

        return frame

    def apply(self, frame, time, cam2screen, expedite=False):
        img = self.img(time)
        frame = self.paste_img_in_frame(img, frame, time)
        self.ref_shape = img.shape[:2]
        frame = super().apply(frame, time, cam2screen, expedite=expedite)
        return frame


class ScreenTrf (ResizedImage):
    """ Transform the whole screen 
    """
    def __init__(self, *args, bg_color=255, **kwargs):
        super().__init__(*args, img=None, **kwargs)
        self.bg_color = make_temporal(bg_color)

    def apply(self, frame, time, cam2screen, expedite=False):
        return self.paste_img_in_frame(frame, np.full_like(frame, self.bg_color(time)), time)


class Arrow (PyplotSegment):
    def __init__(self, total_time, start, end, color='k', head_width=10, head_length=None, **kw):
        super().__init__(total_time, **kw)
        self.start = make_temporal(start)
        self.end = make_temporal(end)
        self.arrow_kw = dict(color=color, head_width=head_width, head_length=head_length)

    def draw(self, ax, t, cam2screen):
        p1 = self._resolve_pt(self.start(t), cam2screen)
        p2 = self._resolve_pt(self.end(t), cam2screen)
        ax.arrow(*p1,*(p2-p1), **self.arrow_kw)


class Text (PyplotSegment):
    def __init__(self, total_time, xy, text, color='k', fontsize=12, 
                 va='center', ha='center', fontweight='normal', **kw):
        super().__init__(total_time, **kw)
        self.xy = make_temporal(xy)
        self.text = text
        self.text_kw = dict(color=color, fontsize=fontsize, va=va, ha=ha, fontweight=fontweight)

    def draw(self, ax, t, cam2screen):
        x, y = self._resolve_pt(self.xy(t))
        ax.text(x, y, self.text, **self.text_kw)


class RoundedBox (PyplotSegment):
    def __init__(self, total_time, bbox_ltrb, 
                 box_pad=0, rounding_size=5,
                 edgecolor="black",  # Border color
                 facecolor="lightblue",  # Fill color
                 linewidth=2, **kw):
        super().__init__(total_time, **kw)
        self.bbox_ltrb = make_temporal(bbox_ltrb)
        self.box_pad = box_pad
        self.rounding_size = rounding_size
        self.facecolor = make_temporal(facecolor)
        self.box_kw = dict(edgecolor=edgecolor, linewidth=linewidth)

    def draw(self, ax, time, cam2screen):
        l, t, r, b = np.ravel(self.bbox_ltrb(time))

        l, t = self._resolve_pt((l,t))
        r, b = self._resolve_pt((r,b))

        box = FancyBboxPatch(
            (l, t),  # Bottom-left corner (x, y)
            r-l,         # Width
            b-t,         # Height
            boxstyle = f"round,pad={self.box_pad},rounding_size={self.rounding_size}", 
            facecolor = self.facecolor(time),
            **self.box_kw)
        ax.add_patch(box)


class RoundedBoxWithText (RoundedBox, Text):
    def __init__(self, total_time, bbox_ltrb, text, **kw):
        bbox_ltrb = make_temporal(bbox_ltrb)
        xy = lambda t: np.asarray(bbox_ltrb(t)).reshape(2,2).mean(0)
        super().__init__(total_time, bbox_ltrb=bbox_ltrb, xy=xy, text=text, va='center_baseline', ha='center', **kw)

    def draw(self, ax, t, cam2screen):
        RoundedBox.draw(self, ax, t, cam2screen)
        Text.draw(self, ax, t, cam2screen)


def make_temporal( obj ):
    " the object should become a function of relative time "
    if obj is None:
        return None
    elif callable(obj):
        return obj
    else:
        # make a constant
        return lambda t: obj


def linear(start, end, max_time=1, min_time=0):
    start = np.asarray(start)
    end = np.asarray(end)
    max_time -= min_time

    def morph_func(t):
        t -= min_time
        if t <= 0:
            return start
        elif t < max_time:
            res = (t/max_time)*end + (1 - t/max_time)*start
            return res.astype(end.dtype)
        else:
            return end
    return morph_func


def smooth(start, end, max_time=1, min_time=0):
    assert 0 <= min_time < max_time
    max_time -= min_time
    start = np.asarray(start)
    end = np.asarray(end)

    def morph_func(t):
        t -= min_time
        if t <= 0:
            return start
        elif t < max_time:
            x = t / max_time
            y = -2*x**3 + 3*x**2
            res = (y)*end + (1 - y)*start
            return res.astype(end.dtype)
        else:
            return end
    return morph_func



if __name__ == '__main__':
    cam = CameraPose(motion=lambda t: np.r_[np.cos(t),np.sin(t),0], spatial_scale=2, time_scale=1, look_at=(0,0,10))

    pts3d = 4*np.random.rand(1000,3) - 2 + (0,0,5)
    color = np.uint8(255 * np.random.rand(1000,3))

    mm = MovieMaker(cam, render_flags=pyrender.RenderFlags.FLAT)
    # mm.add_segment(RoundedBox(10, (300,200,400, 220)), prev=None)
    # mm.add_segment(Text(10, (300, 200),'SOME TEXT',color='r',fontsize=20,va='top'), prev=None)
    mm.add_segment(RoundedBoxWithText(2, (300,200,500, 220), 'SOME TEXT', color='r',fontsize=20), prev=None)
    # mm.add_segment(Arrow(10, (300,200), pts3d.max(axis=0)), prev=None)
    prev = mm.add_segment(ScreenShift((1,2), linear((0,0),(-200,0),1)), prev=None)
    mm.add_segment(RoundedBoxWithText(10, (100,200,300, 220), 'SOME TEXT', color='r',fontsize=20), prev=prev)
    mm.add_segment(PointCloud(10,pts3d,color), prev=prev)

    for frame in mm.generate(700, 500, fps=25, point_size=4):
        pl.clf()
        pl.imshow(frame)
        pl.pause(0.001)
        # breakpoint()
