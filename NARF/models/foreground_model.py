import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .nerf_model import NeRF, PoseConditionalNeRF
from .enarf_model import ENARFGenerator
from .stylegan import EqualConv2d, EqualLinear

from .model_utils import flex_grid_ray_sampler, random_or_patch_sampler
from einops import rearrange


class ForegroundGenerator(nn.Module):
    def __init__(self, config, size, intrinsics=None, num_bone=1, ray_sampler=flex_grid_ray_sampler,
                 parent_id=None):
        super().__init__()
        self.config = config
        self.size = size
        self.intrinsics = intrinsics
        self.inv_intrinsics = np.linalg.inv(intrinsics)
        normalized_intrinsics = np.concatenate([intrinsics[:2] / size, np.array([[0, 0, 1]])], axis=0)
        self.normalized_inv_intrinsics = np.linalg.inv(normalized_intrinsics)
        self.num_bone = num_bone
        self.ray_sampler = ray_sampler
        
        self.enarf = ENARFGenerator(config.nerf_params, rank=0, size=128, channels=96, 
                 num_bone=num_bone, parent=parent_id)

    @property
    def memory_cost(self):
        return self.enarf.memory_cost

    @property
    def flops(self):
        return self.enarf.flops

    def forward(self, pose_to_camera, pose_to_world, bone_length, background=None, z=None, inv_intrinsics=None,
                z_foreground=None, z_background=None):
        """
        generate image from 3d bone mask
        :param pose_to_camera: camera coordinate of joint
        :param pose_to_world: wold coordinate of joint
        :param bone_length:
        :param background:
        :param z: latent vector
        :param inv_intrinsics:
        :return:
        """

        assert self.num_bone == 1 or (bone_length is not None and pose_to_camera is not None)
        batchsize = pose_to_camera.shape[0]
        patch_size = self.config.patch_size

        grid, img_coord = self.ray_sampler(self.size, patch_size, batchsize)

        # sparse rendering
        if inv_intrinsics is None:
            inv_intrinsics = self.inv_intrinsics
        inv_intrinsics = torch.tensor(inv_intrinsics).float().cuda(img_coord.device)
        rendered_color, rendered_mask = self.enarf(z_foreground, batchsize, patch_size ** 2, img_coord,
                                                  pose_to_camera, inv_intrinsics, 
                                                  pose_to_world, bone_length, thres=0.0,
                                                  Nc=self.config.nerf_params.Nc,
                                                  Nf=self.config.nerf_params.Nf)
        if self.ray_sampler in [flex_grid_ray_sampler]:  # TODO unify the way to sample
            rendered_color = rendered_color.reshape(batchsize, 3, patch_size, patch_size)
            rendered_mask = rendered_mask.reshape(batchsize, patch_size, patch_size)

        if background is None:
            rendered_color = rendered_color + (-1) * (1 - rendered_mask[:, None])  # background is black
        else:
            if np.isscalar(background):
                sampled_background = background
            else:
                if self.ray_sampler in [flex_grid_ray_sampler]:  # TODO unify the way to sample
                    sampled_background = torch.nn.functional.grid_sample(background,
                                                                         grid, mode='bilinear')
                else:
                    sampled_background = torch.gather(background, dim=2, index=grid[:, None].repeat(1, 3, 1))

            rendered_color = rendered_color + sampled_background * (1 - rendered_mask[:, None])

        return rendered_color, rendered_mask, grid