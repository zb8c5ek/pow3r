# Copyright (C) 2025-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from copy import deepcopy
import torch
import torch.nn as nn

import pow3r.tools.path_to_dust3r
from dust3r.model import CroCoNet
from dust3r.utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from dust3r.patch_embed import get_patch_embed as dust3r_patch_embed

from pow3r.model.patch_embed import get_patch_embed
from pow3r.model.blocks import Mlp, BlockInject, DecoderBlockInject
from pow3r.model.heads import head_factory


class Pow3R (CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """
    def __init__(self, 
                 mode = 'embed',
                 head_type = 'linear',

                 patch_embed_cls = 'PatchEmbedDust3R', 
                 freeze = 'none',
                 landscape_only = True,
                 
                 **croco_kwargs):

        # retrieve all default arguments using python magic
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)
        del self.mask_token # useless
        del self.prediction_head

        dec_dim, enc_dim = self.decoder_embed.weight.shape
        self.enc_embed_dim = enc_dim
        self.dec_embed_dim = dec_dim

        self.mode = mode
        # additional parameters in the encoder
        img_size = self.patch_embed.img_size
        patch_size = self.patch_embed.patch_size[0]
        self.patch_embed = dust3r_patch_embed(patch_embed_cls, img_size, patch_size, self.enc_embed_dim)
        self.patch_embed_rays = get_patch_embed(patch_embed_cls+'_Mlp', img_size, patch_size, self.enc_embed_dim, in_chans=3)
        self.patch_embed_depth = get_patch_embed(patch_embed_cls+'_Mlp', img_size, patch_size, self.enc_embed_dim, in_chans=2)
        self.pose_embed = Mlp(12, 4*dec_dim, dec_dim)

        # additional parameters in the decoder
        self.dec_cls = ('_cls' in self.mode)
        self.dec_num_cls = 0
        if self.dec_cls:
            # use a CLS token in the decoder only
            self.mode = self.mode.replace('_cls','')
            self.cls_token1 = nn.Parameter(torch.zeros((dec_dim,)))
            self.cls_token2 = nn.Parameter(torch.zeros((dec_dim,)))
            self.dec_num_cls = 1 # affects all blocks

        use_ln = '_ln' in self.mode # TODO remove?
        self.patch_ln = nn.LayerNorm(enc_dim) if use_ln else nn.Identity()
        self.dec1_pre_ln = nn.LayerNorm(dec_dim) if use_ln else nn.Identity()
        self.dec2_pre_ln = nn.LayerNorm(dec_dim) if use_ln else nn.Identity()

        self.dec_blocks2 = deepcopy(self.dec_blocks)

        # here we modify some of the blocks
        self.replace_some_blocks()

        self.set_downstream_head( head_type, landscape_only, **croco_kwargs )
        self.set_freeze(freeze)

    def replace_some_blocks(self):
        assert self.mode.startswith('inject') #inject[0,0.5]
        NewBlock = BlockInject
        DecoderNewBlock = DecoderBlockInject

        all_layers = {i/n for i in range(len(self.enc_blocks)) for n in [len(self.enc_blocks), len(self.dec_blocks)]}
        which_layers = eval(self.mode[self.mode.find('['):]) or all_layers
        assert isinstance(which_layers, (set,list))

        n = 0
        for i, block in enumerate(self.enc_blocks):
            if i/len(self.enc_blocks) in which_layers: 
                print('modifying encoder block',i)
                block.__class__ = NewBlock
                block.init(self.enc_embed_dim)
                n += 1
        assert n == len(which_layers), breakpoint()

        n = 0
        for i in range(len(self.dec_blocks)):
            for blocks in [self.dec_blocks, self.dec_blocks2]:
                block = blocks[i]
                if i/len(self.dec_blocks) in which_layers: 
                    print('modifying decoder block',i)
                    block.__class__ = DecoderNewBlock
                    block.init(self.dec_embed_dim)
                    n += 1
        assert n == 2*len(which_layers), breakpoint()

    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kw):
        return _load_model(pretrained_model_path, device='cpu')

    def load_state_dict(self, ckpt, **kw ):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks','dec_blocks2')] = value
        # remove layers that have different shapes
        cur_ckpt = self.state_dict()
        for key, val in ckpt.items():
            if key.startswith('downstream_head2.proj'):
                if key in cur_ckpt and cur_ckpt[key].shape != val.shape:
                    print(f' (removing ckpt[{key}] because wrong shape)')
                    del new_ckpt[key]
        return super().load_state_dict(new_ckpt, **kw )

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [], 
            'encoder':  [self.patch_embed, self.enc_blocks], 
        }
        freeze_all_params(to_be_frozen[freeze])

    def set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, head_type, landscape_only, patch_size, img_size, mlp_ratio, dec_depth, **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'

        # split heads if different
        heads = head_type.split(';')
        assert len(heads) in (1,2)
        head1_type, head2_type = (heads+heads)[:2]

        # allocate heads
        self.downstream_head1 = head_factory(head1_type, self)
        self.downstream_head2 = head_factory(head2_type, self)

        # magic wrapper 
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape, rays=None, depth=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        if rays is not None: # B,3,H,W
            rays_emb, pos2 = self.patch_embed_rays(rays, true_shape=true_shape)
            assert (pos == pos2).all()
            if self.mode.startswith('embed'): 
                x = x + rays_emb
        else:
            rays_emb = None

        if depth is not None: # B,2,H,W
            depth_emb, pos2 = self.patch_embed_depth(depth, true_shape=true_shape)
            assert (pos == pos2).all()
            if self.mode.startswith('embed'): 
                x = x + depth_emb
        else:
            depth_emb = None

        x = self.patch_ln(x)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos, rays=rays_emb, depth=depth_emb)

        x = self.enc_norm(x)
        return x, pos

    def encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B,1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B,1))
        # warning! maybe the images have different portrait/landscape orientations

        # privileged information
        rays1 = view1.get('known_rays', None)
        rays2 = view2.get('known_rays', None)
        depth1 = view1.get('known_depth', None)
        depth2 = view2.get('known_depth', None)
        
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            def hsub(x): return None if x is None else x[::2]
            feat1, pos1 = self._encode_image(img1[::2], shape1[::2], rays=hsub(rays1), depth=hsub(depth1))
            feat2, pos2 = self._encode_image(img2[::2], shape2[::2], rays=hsub(rays2), depth=hsub(depth2))

            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, pos1 = self._encode_image(img1, shape1, rays=rays1, depth=depth1)
            feat2, pos2 = self._encode_image(img2, shape2, rays=rays2, depth=depth2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, relpose1=None, relpose2=None):
        final_output = [(f1, f2)] # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        # add CLS token for the decoder
        if self.dec_cls:
           cls1 = self.cls_token1[None,None].expand(len(f1),1,-1).clone()
           cls2 = self.cls_token2[None,None].expand(len(f2),1,-1).clone()

        if relpose1 is not None: # shape = (B, 4, 4)
            pose_emb1 = self.pose_embed(relpose1[:,:3].flatten(1)).unsqueeze(1)
            if self.mode.startswith('embed'): 
                if self.dec_cls:
                    cls1 = cls1 + pose_emb1
                else:
                    f1 = f1 + pose_emb1
        else:
            pose_emb1 = None

        if relpose2 is not None: # shape = (B, 4, 4)
            pose_emb2 = self.pose_embed(relpose2[:,:3].flatten(1)).unsqueeze(1)
            if self.mode.startswith('embed'): 
                if self.dec_cls:
                    cls2 = cls2 + pose_emb2
                else:
                    f2 = f2 + pose_emb2
        else:
            pose_emb2 = None

        if self.dec_cls:
            f1, pos1 = cat_cls(cls1, f1, pos1)
            f2, pos2 = cat_cls(cls2, f2, pos2)

        f1 = self.dec1_pre_ln(f1)
        f2 = self.dec2_pre_ln(f2)

        final_output.append((f1, f2)) # to be removed later
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2, relpose=pose_emb1, num_cls=self.dec_num_cls)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1, relpose=pose_emb2, num_cls=self.dec_num_cls)
            # store the result
            final_output.append((f1,f2))

        del final_output[1] # duplicate with final_output[0] (after decoder proj)
        if self.dec_cls: # remove cls token for decoder layers
            final_output[1:] = [(f1[:,1:],f2[:,1:]) for f1, f2 in final_output[1:]]
        # normalize last output
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2 ):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, 
                            relpose1=view1.get('known_pose'), relpose2=view2.get('known_pose'))
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d') # predict view2's pts3d in view1's frame
        return res1, res2


def convert_release_dust3r_args(args): # TODO remove
    args.model = args.model.replace("patch_embed_cls", "patch_embed") \
        .replace("AsymmetricMASt3R", "AsymmetricCroCo3DStereo") \
        .replace("PatchEmbedDust3R", "convManyAR") \
        .replace("pos_embed='RoPE100'", "enc_pos_embed='cuRoPE100', dec_pos_embed='cuRoPE100'", )
    return args


def _load_model(model_path, device): # TODO remove
    print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    try:
        net = eval(ckpt['args'].model[:-1].replace("convManyAR","convP")+', landscape_only=False)')
    except Exception:
        args = convert_release_dust3r_args(ckpt['args'])
        net = eval(args.model[:-1].replace("convManyAR","convP")+', landscape_only=False)')
    ckpt['model'] = {k.replace('_downstream_head','downstream_head'):v for k,v in ckpt['model'].items()}
    print(net.load_state_dict(ckpt['model'], strict=False))
    return net.to(device)


def cat_cls(cls, tokens, pos):
    tokens = torch.cat((cls, tokens), dim=1)
    pos = torch.cat((-pos.new_ones(len(cls), 1, 2), pos), dim=1)
    return tokens, pos


if __name__ == '__main__': # TODO remove
    import numpy as np
    import roma
    from dust3r.utils.device import to_cpu
    from dust3r.losses import ConfLoss, Regr3D, L21
    from pow3r.datasets.utils import modalities

    B = 2
    H = 144
    W = 160

    def gen_one_view( num ):
        img = torch.rand((3, H, W))
        from scipy.ndimage import zoom
        K = np.float32([(100,0,50), (0,100,40), (0,0,1)])
        depthmap = 2 * zoom(np.random.rand(H//16,W//16), 16)
        valid_mask = zoom(np.random.rand(H//16,W//16), 16) < 0.5
        pose = np.r_[np.random.randn(3,4), [(0,0,0,1)]]
        pose[:3, :3] = roma.special_procrustes(torch.from_numpy(pose[:3, :3])).numpy()

        view = dict(
            img = img,
            true_shape = np.int32((H, W)), 
            pts3d = np.random.rand(H,W,3),
            is_metric_scale = np.array(False), # np.array([False])
            camera_intrinsics = K, 
            depthmap = depthmap.astype(np.float32),
            valid_mask = valid_mask,
            camera_pose = pose.astype(np.float32),
            instance = torch.tensor([num]),
            )
        view['modalities'] = 'K+D+RT'
        view['known_rays'] = modalities.gen_rays(view)
        view['known_depth'] = modalities.gen_sparse_depth(view, 64, 0, np.random)
        view['known_pose'] = modalities.gen_rel_pose([view]*2)
        # add sky_mask
        view['sky_mask'] = view['depthmap'] < 0
        return to_cpu(view)

    from torch.utils.data.dataloader import default_collate as collate
    view1 = collate([gen_one_view(1) for _ in range(B)])
    view2 = collate([gen_one_view(2) for _ in range(B)])

    rope100 = 'RoPE100'

    # loss = 
    criterion = ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)

    for mode in ['inject[0]_cls', 'inject[0,0.5]']:
      # Predicting X22 : linear_pts3d+conf+ptd3d2+conf2 -> we need 2 confidence level for both X21 (conf) and X22(conf2)
      for head_type in [
            'linear_pts3d+conf;linear_pts3d+conf+pts3d2+conf2',
            ]:
        print(mode, head_type)

        net = Pow3R(
                mode = mode,
                patch_embed_cls = 'ManyAR_PatchEmbed',
                head_type = head_type,
                enc_depth=24, enc_embed_dim = 1024, enc_num_heads=16, 
                dec_depth=12, dec_embed_dim = 768, dec_num_heads=12, 
                pos_embed = rope100, 
                img_size = (H, W),
            )
        # print(net)
        print('trainable parameters:', sum([p.numel() for n,p in net.named_parameters() if p.requires_grad]))
        print('trainable head1 parameters:', sum([p.numel() for n,p in net.downstream_head1.named_parameters() if p.requires_grad]))
        print('trainable head2 parameters:', sum([p.numel() for n,p in net.downstream_head2.named_parameters() if p.requires_grad]))

        # cp = 'checkpoints/Pow3R_ViTLarge_BaseDecoder_512_linear.pth'
        # cp = torch.load(cp, map_location='cpu', weights_only=False)
        # print(net.load_state_dict(cp['model']))

        res1, res2 = net(view1, view2)
        loss = criterion(view1, view2, res1, res2)
        print(loss)

        assert res1['pts3d'].shape == (B,H,W,3), breakpoint()
        assert res2['pts3d_in_other_view'].shape == (B,H,W,3), breakpoint()
        assert res2['pts3d2'].shape == (B,H,W,3), breakpoint()
        assert res2['conf2'].shape == (B,H,W), breakpoint()
        for res in [res1, res2]:
            assert res['conf'].shape == (B,H,W), breakpoint()

    print('>> passed!')
