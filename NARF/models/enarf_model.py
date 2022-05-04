import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .activation import MyReLU
from .stylegan import EqualLinear, EqualConv1d, NormalizedConv1d, ModulatedConv2d
from .networks_stylegan2 import SynthesisNetwork, MappingNetwork
from .model_utils import in_cube
from .utils_3d import SE3
        


class ENARFGenerator(nn.Module):
    def __init__(self, config, rank=0, size=128, channels=96, 
                 num_bone=1, parent=None,
                 mapping_kwargs={}, nerf_decoder_kwargs = {},
                 ):
        '''
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        size,                       # Output resolution.
        img_channels                = 96,  # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        nerf_decoder_kwargs      = {},
        rank = None,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
        '''
        super(ENARFGenerator, self).__init__()
        
        self.config = config
        self.rank = rank
        self.z_dim = 512
        self.c_dim = num_bone * (1+12)
        self.w_dim = 512
        self.size = 128
        self.channels = 32
        self.save_mask = False  # flag to save mask
        self.groups = 1
        self.density_scale = config.density_scale
        self.num_bone = num_bone
        self.t = torch.new_ones((bs,20))            # TODO: 实现 t modulate G_dec
        self.parent_id = parent

        if self.config.use_gan:
            #synthesis_kwargs['use_noise'] = True
            self.synthesis = SynthesisNetwork(
                                w_dim = self.w_dim, 
                                img_resolution = size, 
                                img_channels = (self.channels + self.num_bone) * 3,
                                num_fp16_res=0,
                                use_noise = True)
            self.mapping = MappingNetwork(z_dim=self.z_dim, c_dim=self.c_dim, w_dim=self.w_dim, 
                                        num_ws=self.synthesis.num_ws, **mapping_kwargs)
        if self.config.dynamic:
            self.c_dim = 20 + 9 * (self.num_bone-1)
            self.deform_synthesis = SynthesisNetwork(
                                w_dim = self.w_dim, 
                                img_resolution = size, 
                                img_channels = 2 * 3,
                                num_fp16_res=0,
                                use_noise = True)
            self.deform_mapping = MappingNetwork(z_dim=self.z_dim, c_dim=self.c_dim, w_dim=self.w_dim, 
                                        num_ws=self.deform_synthesis.num_ws, **mapping_kwargs)        
            self.decoder = ModulatedConv2d(in_channel=32, out_channel=4, kernel_size=3, style_dim=self.c_dim)
        else:
            self.decoder = ModulatedConv2d(**decoder_kwargs)
        


    def update_selector_tmp(self):
        gamma = self.config.selector_adaptive_tmp.gamma
        if gamma != 1:
            self.selector_tmp[()] = torch.clamp_min(self.selector_tmp * gamma,
                                                    self.config.selector_adaptive_tmp.min).float()

    def encode(self, value, num_frequency: int, num_bone=None):
        """
        positional encoding for group conv
        :param value: b x -1 x n
        :param num_frequency: L in NeRF paper
        :param num_bone: num_bone for positional encoding
        :return:
        """
        b, _, n = value.shape
        num_bone = num_bone or self.num_bone  # replace if None
        values = [2 ** i * value.reshape(b, num_bone, -1, n) * np.pi for i in range(num_frequency)]
        values = torch.cat(values, dim=2)
        gamma_p = torch.cat([torch.sin(values), torch.cos(values)], dim=2)
        gamma_p = gamma_p.reshape(b, -1, n)
        # mask outsize [-1, 1]
        mask = (value.reshape(b, num_bone, -1, n).abs() > 1).float().sum(dim=2, keepdim=True) >= 1
        mask = mask.float().repeat(1, 1, gamma_p.shape[1] // num_bone, 1)
        mask = mask.reshape(gamma_p.shape)
        return gamma_p * (1 - mask)  # B x (groups * ? * L * 2) x n

    def backbone_(self, p, j=None, bone_length=None, ray_direction=None):
        '''
        Params:
        p: sampled points: b x groups * 3 x n (n = num_of_ray * points_on_ray)
        j:  pose_world:  b x groups x 4 x 4
        bone_length: b x groups x 1
        :param ray_direction: b x groups * 3 x m (m = number of ray)

        Returns: 
        b x groups x 4 x n
        '''
        bs, _, n = p.shape  
        feat_xy, feat_yz, feat_xz = self.triplane_features.chunk(3, dim=1)  # b x 32 x 128 x 128
        prob_xy, prob_yz, prob_xz = self.triplane_probs.chunk(3, dim=1)  # b x bones x 128 x 128

        feats = self.bilinear_sample_tri_plane(p, feat_xy, feat_yz, feat_xz)  # b, [32,32,32], num_bones, num_sampling
        features = (feats[0] + feats[1] + feats[2]).permute(0,3,2,1)
        #nerf_feat = self.nerf_decoder(nerf_feat).reshape(batchsize, self.num_bone, 4, -1)  # bs, 32+1, num_bones, num_sampling
        probs = self.bilinear_sample_tri_plane(p, prob_xy, prob_yz, prob_xz)    # b, 19, num_bones, num_sampling
        tri_probs = (probs[0] * probs[1] * probs[2]).permute(0,3,2,1)           # b, num_sampling, num_bones, 19 
        #tri_probs = F.normalize(tri_probs, dim=-1)
        mask_probs = tri_probs.new_zeros((bs, n, self.num_bone, 1))
        for i in range(self.num_bone):
            mask_probs[ :, : , i, 0] = tri_probs[ :, : , i, i]
        masked_features = mask_probs * features

        if self.config.dynamic:
            deform_xy, deform_yz, deform_xz = self.deform_features.chunk(3, dim=1)
            deforms = self.bilinear_sample_tri_plane(p, deform_xy, deform_yz, deform_xz)
            deforms = (deforms[0] + deforms[1] + deforms[2]).permute(0,3,2,1)
            deform_probs = tri_probs.new_ones((bs, n, self.num_bone, 1)) / self.num_bone

        inter_features = masked_features.sum(2).permute(0, 2, 1).reshape(bs, self.channels, ray_direction.shape[-1], -1)        # 4 x n x 32 -> 4 x 32 x m x numsample
        out = self.decoder(inter_features.contiguous(),self.t).reshape(bs, 4, -1)       # b x 4 x m x numsample
        density = out[:, 0:1, :]
        color = out[:, None, 1:, :]
        return density, color       # b x 1 x n    b x 1 x 3 x n

    def backbone(self, p, j=None, bone_length=None, ray_direction=None):
        num_pixels = ray_direction.shape[2]  # number of sampled pixels
        chunk_size = self.config.max_chunk_size // p.shape[0]
        if num_pixels > chunk_size:
            num_points_on_a_ray = p.shape[2] // ray_direction.shape[2]
            density, color = [], []
            for i in range(0, num_pixels, chunk_size):
                p_chunk = p[:, :, i * num_points_on_a_ray:(i + chunk_size) * num_points_on_a_ray]
                ray_direction_chunk = ray_direction[:, :, i:i + chunk_size]
                bone_length.requires_grad = True
                density_i, color_i = torch.utils.checkpoint.checkpoint(self.backbone_, p_chunk, j,
                                                                       bone_length, ray_direction_chunk)
                density.append(density_i)
                color.append(color_i)

            density = torch.cat(density, dim=2)
            color = torch.cat(color, dim=3)
            return density, color

        else:
            return self.backbone_(p, j, bone_length, ray_direction)

    def calc_color_and_density(self, p, pose_world=None, bone_length=None, ray_direction=None):
        """
        forward func of ImplicitField
        :param pose_world:
        :param p: b x groups * 3 x n (n = num_of_ray * points_on_ray)
        :param z: b x dim
        :param pose_world: b x groups x 4 x 4
        :param bone_length: b x groups x 1
        :param ray_direction: b x groups * 3 x m (m = number of ray)
        :return: b x groups x 4 x n
        """
        density, color = self.backbone(p, pose_world, bone_length=bone_length, ray_direction=ray_direction)
        if not self.config.concat:
            # density is zero if p is outside the cube
            density *= in_cube(p)
        return density, color  # B x groups x 1 x n, B x groups x 3 x n

    @staticmethod
    def coord_transform(p: torch.tensor, rotation: torch.tensor, translation: torch.tensor) -> torch.tensor:
        # 'world coordinate' -> 'bone coordinate'
        return torch.matmul(rotation.permute(0, 2, 1), p - translation)

    def sum_density(self, density: torch.tensor, semantic_map: bool = False) -> (torch.tensor, ) * 2:
        """

        :param density: B x num_bone x 1 x n x N
        :param semantic_map:
        :return:
        """
        temperature = 100 if semantic_map else self.arf_temperature
        alpha = torch.softmax(density * temperature, dim=1)  # B x num_bone x 1 x n x Nc-1

        if self.config.detach_alpha:
            alpha = alpha.detach()

        # sum density across bone
        if self.config.sum_density:
            density = density.sum(dim=1, keepdim=True)
        else:
            density = (density * alpha).sum(dim=1, keepdim=True)
        return density, alpha

    def coarse_to_fine_sample(self, image_coord: torch.tensor, pose_to_camera: torch.tensor,
                              inv_intrinsics: torch.tensor, world_pose: torch.tensor = None,
                              bone_length: torch.tensor = None, near_plane: float = 0.3, far_plane: float = 5,
                              Nc: int = 64, Nf: int = 128, render_scale: float = 1) -> (torch.tensor,) * 3:
        n_samples_to_decide_depth_range = 16
        batchsize, _, _, n = image_coord.shape
        num_bone = 1 if self.config.concat_pose else self.num_bone  # PoseConditionalNeRF or other
        with torch.no_grad():
            #if self.config.concat_pose:  # recompute pose of camera
            #    pose_to_camera = torch.matmul(pose_to_camera, torch.inverse(world_pose))[:, :1]
            # 每个part 相对 camera coord 的rotation & translation
            R = pose_to_camera[:, :, :3, :3].reshape(batchsize * num_bone, 3, 3)  # B*num_bone x 3 x 3
            t = pose_to_camera[:, :, :3, 3].reshape(batchsize * num_bone, 3, 1)  # B*num_bone x 3 x 1

            if True:
                image_coord = image_coord.reshape(batchsize * num_bone, 3, n)
                # img coord -> camera coord
                sampled_camera_coord = torch.matmul(inv_intrinsics, image_coord)
            else:
                # reshape for multiplying inv_intrinsics
                image_coord = image_coord.reshape(batchsize, num_bone, 3, n)
                image_coord = image_coord.permute(0, 2, 1, 3)  # B x 3 x num_bone x n
                image_coord = image_coord.reshape(batchsize, 3, num_bone * n)

                # img coord -> camera coord
                sampled_camera_coord = torch.matmul(inv_intrinsics, image_coord)
                sampled_camera_coord = sampled_camera_coord.reshape(batchsize, 3, num_bone, n)
                sampled_camera_coord = sampled_camera_coord.permute(0, 2, 1, 3)
                sampled_camera_coord = sampled_camera_coord.reshape(batchsize * num_bone, 3, n)

            # camera coord -> bone coord
            sampled_bone_coord = self.coord_transform(sampled_camera_coord, R, t)  # B*num_bone x 3 x n

            # camera origin (bone coord)
            camera_origin = self.coord_transform(torch.zeros_like(sampled_camera_coord), R, t)  # B*num_bone x 3 x n

            def get_canonical_Rt(batchsize, num_bone, pose):  # B x num_bone x 4 x 4     # TODO: 写得很可能有问题!! 求canonical pose把所有part都转换到canonical part
                R = pose[:, self.parent_id[1:], :3, :3]
                t_pa = pose[:, self.parent_id[1:], :3, 3]
                R_pa = pose[:, self.parent_id[1:], :3, :3]
                t_ch = pose[:, 1:, :3, 3]
                R_ch = pose[:, 1:, :3, :3]
                t_diff = torch.matmul(R.permute(0, 1, 3, 2), (t_ch - t_pa)[:, :, :, None])
                R_diff = torch.matmul(R, R_ch)
                canonical_t = [pose.new_zeros((batchsize, 1, 3, 1))]
                canonical_R = [pose.new_ones((batchsize, 1, 3, 3))]
                for i in range(1, pose.shape[1]):
                    canonical_t.append(canonical_t[self.parent_id[i]] + t_diff[:, i-1, ...].unsqueeze(1))
                    canonical_R.append(torch.matmul(canonical_R[self.parent_id[i]], R_diff[:, i-1, ...].unsqueeze(1)))
                canonical_t = torch.cat(canonical_t, dim=1).reshape(batchsize*num_bone,3,1)
                canonical_R = torch.cat(canonical_R, dim=1).reshape(batchsize*num_bone,3,3)
                return canonical_R, canonical_t
            
            canonical_R, canonical_t = get_canonical_Rt(batchsize, num_bone, pose_to_camera)

            def canonical_transform(p, can_R, can_t):
                return torch.matmul(can_R.permute(0, 1, 2), p) + can_t

            # bone coord -> canonical coord
            sampled_bone_coord = canonical_transform(sampled_bone_coord, canonical_R, canonical_t)  # B*num_bone x 3 x n
            # camera origin (canonical coord)
            camera_origin = canonical_transform(camera_origin, canonical_R, canonical_t)  # B*num_bone x 3 x n
            # ray direction (canonical coord)
            ray_direction = sampled_bone_coord - camera_origin  # B*num_bone x 3 x n

            # unit ray direction (canonical coord)
            ray_direction = F.normalize(ray_direction, dim=1)  # B*num_bone x 3 x n

            # sample points to decide depth range
            sampled_depth = torch.linspace(near_plane, far_plane, n_samples_to_decide_depth_range, device="cuda")
            sampled_points_on_rays = camera_origin[:, :, :, None] + ray_direction[:, :, :, None] * sampled_depth

            # inside the cube [-1, 1]^3?
            inside = in_cube(sampled_points_on_rays)  # B*num_bone x 1 x n x n_samples_to_decide_depth_range

            # minimum-maximum depth
            depth_min = torch.where(inside, sampled_depth * inside,
                                    torch.full_like(inside.float(), 1e3)).min(dim=3)[0]
            depth_max = torch.where(inside, sampled_depth * inside,
                                    torch.full_like(inside.float(), -1e3)).max(dim=3)[0]
            # # replace values if no intersection
            depth_min = torch.where(inside.sum(dim=3) > 0, depth_min, torch.full_like(depth_min, near_plane))
            depth_max = torch.where(inside.sum(dim=3) > 0, depth_max, torch.full_like(depth_max, far_plane))

            # adopt the smallest/largest values among bones
            depth_min = depth_min.reshape(batchsize, num_bone, 1, n).min(dim=1, keepdim=True)[0]  # B x 1 x 1 x n
            depth_max = depth_max.reshape(batchsize, num_bone, 1, n).max(dim=1, keepdim=True)[0]

            start = (camera_origin.reshape(batchsize, num_bone, 3, n) +
                     depth_min * ray_direction.reshape(batchsize, num_bone, 3, n))  # B x num_bone x 3 x n
            end = (camera_origin.reshape(batchsize, num_bone, 3, n) +
                   depth_max * ray_direction.reshape(batchsize, num_bone, 3, n))  # B x num_bone x 3 x n

            # coarse ray sampling
            bins = (torch.arange(Nc, dtype=torch.float, device="cuda").reshape(1, 1, 1, 1, Nc) / Nc +
                    torch.cuda.FloatTensor(batchsize, 1, 1, n, Nc).uniform_() / Nc)
            coarse_points = start.unsqueeze(4) * (1 - bins) + end.unsqueeze(4) * bins  # B x num_bone x 3 x n x Nc
            coarse_depth = (depth_min.unsqueeze(4) * (1 - bins) +
                            depth_max.unsqueeze(4) * bins)  # B x 1 x 1 x n x Nc

            ray_direction = ray_direction.reshape(batchsize, num_bone * 3, n)

            # coarse density
            coarse_density, _ = self.calc_color_and_density(coarse_points.reshape(batchsize, num_bone * 3, n * Nc),
                                                            world_pose,
                                                            bone_length,
                                                            ray_direction)    # B x groups x n*Nc

            if self.groups > 1:
                # alpha blending
                coarse_density, _ = self.sum_density(coarse_density)

            # calculate weight for fine sampling
            coarse_density = coarse_density.reshape(batchsize, 1, 1, n, Nc)[:, :, :, :, :-1]
            # # delta = distance between adjacent samples
            delta = coarse_depth[:, :, :, :, 1:] - coarse_depth[:, :, :, :, :-1]  # B x 1 x 1 x n x Nc-1

            density_delta = coarse_density * delta * render_scale
            T_i = torch.exp(-(torch.cumsum(density_delta, dim=4) - density_delta))
            weights = T_i * (1 - torch.exp(-density_delta))  # B x 1 x 1 x n x Nc-1
            weights = weights.reshape(batchsize * n, Nc - 1)
            # fine ray sampling
            bins = (torch.multinomial(torch.clamp_min(weights, 1e-8),
                                      Nf, replacement=True).reshape(batchsize, 1, 1, n, Nf).float() / Nc +
                    torch.cuda.FloatTensor(batchsize, 1, 1, n, Nf).uniform_() / Nc)
            fine_points = start.unsqueeze(4) * (1 - bins) + end.unsqueeze(4) * bins  # B x num_bone x 3 x n x Nf
            fine_depth = (depth_min.unsqueeze(4) * (1 - bins) +
                          depth_max.unsqueeze(4) * bins)  # B x 1 x 1 x n x Nc

            # sort points
            fine_points = torch.cat([coarse_points, fine_points], dim=4)
            fine_depth = torch.cat([coarse_depth, fine_depth], dim=4)
            arg = torch.argsort(fine_depth, dim=4)

            fine_points = torch.gather(fine_points, dim=4,
                                       index=arg.repeat(1, num_bone, 3, 1, 1))  # B x num_bone x 3 x n x Nc+Nf
            fine_depth = torch.gather(fine_depth, dim=4, index=arg)  # B x 1 x 1 x n x Nc+Nf

            fine_points = fine_points.reshape(batchsize, num_bone * 3, n * (Nc + Nf))

        if pose_to_camera.requires_grad:
            R = pose_to_camera[:, :, :3, :3]
            t = pose_to_camera[:, :, :3, 3:]

            with torch.no_grad():
                fine_points = fine_points.reshape(batchsize, num_bone, 3, n * (Nc + Nf))
                fine_points = torch.matmul(R, fine_points) + t
            fine_points = torch.matmul(R.permute(0, 1, 3, 2), fine_points - t).reshape(batchsize, num_bone * 3,
                                                                                       n * (Nc + Nf))
        return (
            fine_depth,  # B x 1 x 1 x n x Nc+Nf
            fine_points,  # B x num_bone*3 x n*Nc+Nf
            ray_direction  # B x num_bone*3 x n
        )

    def render(self, image_coord: torch.tensor, pose_to_camera: torch.tensor, inv_intrinsics: torch.tensor,
               world_pose: torch.tensor = None, bone_length: torch.tensor = None,
               thres: float = 0.9, render_scale: float = 1, Nc: int = 64, Nf: int = 128,
               semantic_map: bool = False) -> (torch.tensor,) * 3:
        near_plane = 0.3
        # n <- number of sampled pixels
        # image_coord: B x groups x 3 x n
        # camera_extrinsics: B x 4 x 4
        # camera_intrinsics: 3 x 3

        batchsize, num_bone, _, n = image_coord.shape

        fine_depth, fine_points, ray_direction = self.coarse_to_fine_sample(image_coord, pose_to_camera,
                                                                            inv_intrinsics,
                                                                            world_pose=world_pose,
                                                                            bone_length=bone_length,
                                                                            near_plane=near_plane, Nc=Nc, Nf=Nf,
                                                                            render_scale=render_scale)
        # fine density & color # B x groups x 1 x n*(Nc+Nf), B x groups x 3 x n*(Nc+Nf)
        if semantic_map and self.config.mask_input:
            self.save_mask = True

        fine_density, fine_color = self.calc_color_and_density(fine_points, world_pose,
                                                               bone_length, ray_direction)
        #if semantic_map and self.config.mask_input:
        #    self.save_mask = False

        # semantic map
        if semantic_map and not self.config.concat:
            bone_idx = torch.arange(num_bone).cuda()
            fine_color = torch.stack([bone_idx // 9, (bone_idx // 3) % 3, bone_idx % 3], dim=1) - 1  # num_bone x 3
            fine_color[::2] = fine_color.flip(dims=(0,))[1 - num_bone % 2::2]
            fine_color = fine_color[None, :, :, None, None]
        elif semantic_map and self.config.mask_input:
            bone_idx = torch.arange(num_bone).cuda()
            seg_color = torch.stack([bone_idx // 9, (bone_idx // 3) % 3, bone_idx % 3], dim=1) - 1  # num_bone x 3
            seg_color[::2] = seg_color.flip(dims=(0,))[1 - num_bone % 2::2]  # num_bone x 3
            fine_color = seg_color[self.mask_prob.reshape(-1)]
            fine_color = fine_color.reshape(batchsize, 1, -1, 3).permute(0, 1, 3, 2)
            fine_color = fine_color.reshape(batchsize, 1, 3, n, Nc + Nf)[:, :, :, :, :-1]
        else:
            fine_color = fine_color.reshape(batchsize, self.groups, 3, n, Nc + Nf)[:, :, :, :, :-1]

        fine_density = fine_density.reshape(batchsize, self.groups, 1, n, Nc + Nf)[:, :, :, :, :-1]

        sum_fine_density = fine_density

        if thres > 0:
            # density = inf if density exceeds thres
            sum_fine_density = (sum_fine_density > thres) * 100000

        delta = fine_depth[:, :, :, :, 1:] - fine_depth[:, :, :, :, :-1]  # B x 1 x 1 x n x Nc+Nf-1
        sum_density_delta = sum_fine_density * delta * render_scale  # B x 1 x 1 x n x Nc+Nf-1

        T_i = torch.exp(-(torch.cumsum(sum_density_delta, dim=4) - sum_density_delta))
        weights = T_i * (1 - torch.exp(-sum_density_delta))  # B x 1 x 1 x n x Nc+Nf-1

        fine_depth = fine_depth.reshape(batchsize, 1, 1, n, Nc + Nf)[:, :, :, :, :-1]

        rendered_color = torch.sum(weights * fine_color, dim=4).squeeze(1)  # B x 3 x n
        rendered_mask = torch.sum(weights, dim=4).reshape(batchsize, n)  # B x n
        rendered_disparity = torch.sum(weights * 1 / fine_depth, dim=4).reshape(batchsize, n)  # B x n

        return rendered_color, rendered_mask, rendered_disparity

    def forward(self, z, batchsize, num_sample, sampled_img_coord, pose_to_camera, inv_intrinsics, 
                world_pose=None, bone_length=None, thres=0.9, render_scale=1, Nc=64, Nf=128,
                truncation_psi=1, truncation_cutoff=None, update_emas=True,
                **synthesis_kwargs):
        """
        rendering function for sampled rays
        :param batchsize:
        :param num_sample:
        :param sampled_img_coord: sampled image coordinate
        :param pose_to_camera:   b x num_bone x 4 x 4
        :param inv_intrinsics:
        :param z:
        :param world_pose:
        :param bone_length:
        :param thres:
        :param render_scale:
        :param Nc:
        :param Nf:
        :return: color and mask value for sampled rays
        """

        # repeat coords along bone axis
        sampled_img_coord = sampled_img_coord.repeat(1, self.num_bone, 1, 1)    # b x num_bone x 3 x patch^2
        bs = pose_to_camera.shape[0]

        if self.config.use_gan:
            cond = torch.cat((pose_to_camera[:, :, 0:3].reshape(bs,-1), bone_length.reshape(bs,-1)), dim=1)
            ws = self.mapping(z.repeat(bs,1), c=cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            # triplanes: b x (channels+bones)*3 x 128 x 128
            triplanes = self.synthesis(ws[:, :self.synthesis.num_ws], update_emas=update_emas, **synthesis_kwargs)  
            self.triplane_features = triplanes[ : , 0:self.channels * 3, ... ]
            self.triplane_probs = triplanes[ : , self.channels * 3: , ... ]
        else:
            self.triplane_features = nn.Parameter( torch.randn(bs, 32*3, self.size, self.size).cuda(sampled_img_coord.device) )
            self.triplane_probs = nn.Parameter( torch.randn(bs, self.num_bone * 3, self.size, self.size).cuda(sampled_img_coord.device) )
        
        if self.config.dynamic:
            cond = torch.cat((pose_to_camera[:, :, 0:3, 0:3].reshape(bs,-1), self.t.reshape(bs,-1)), dim=1)
            if z.shape[0] != bs:
                z = z.repeat(bs, 1)
            ws = self.deform_mapping(z, c=cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            triplanes = self.deform_synthesis(ws[:, :self.deform_synthesis.num_ws], update_emas=update_emas, **synthesis_kwargs)
            self.deform_features = triplanes

        assert self.triplane_features.shape[1] == 96
        assert self.triplane_probs.shape[1] == 57

        if not self.config.use_gan:
            merged_color, merged_mask, _ = self.render(sampled_img_coord,
                                                    pose_to_camera,
                                                    inv_intrinsics,
                                                    world_pose=world_pose,
                                                    bone_length=bone_length,
                                                    thres=thres,
                                                    Nc=Nc,
                                                    Nf=Nf,
                                                    render_scale=render_scale)
        else:
            merged_color, merged_mask, _ = self.render_entire_img(pose_to_camera,
                                                    inv_intrinsics,
                                                    world_pose=world_pose,
                                                    bone_length=bone_length,
                                                    thres=thres,
                                                    Nc=Nc,
                                                    Nf=Nf,
                                                    render_scale=render_scale)

        return merged_color, merged_mask

    def render_entire_img(self, pose_to_camera, inv_intrinsics, world_pose=None, bone_length=None,
                          thres=0.9, render_scale=1, batchsize=1000, render_size=128, Nc=64, Nf=128,
                          semantic_map=False, use_normalized_intrinsics=False):
        #assert bone_length is None or bone_length.shape[0] == 1
        #assert world_pose is None or world_pose.shape[0] == 1
        batchsize = self.config.render_bs or batchsize
        if use_normalized_intrinsics:
            img_coord = torch.stack([(torch.arange(render_size * render_size) % render_size + 0.5) / render_size,
                                     (torch.arange(render_size * render_size) // render_size + 0.5) / render_size,
                                     torch.ones(render_size * render_size).long()], dim=0).float()
        else:
            img_coord = torch.stack([torch.arange(render_size * render_size) % render_size + 0.5,
                                     torch.arange(render_size * render_size) // render_size + 0.5,
                                     torch.ones(render_size * render_size).long()], dim=0).float()

        img_coord = img_coord[None, None].cuda()

        if not self.config.concat_pose:
            img_coord = img_coord.repeat(1, self.num_bone, 1, 1)

        rendered_color = []
        rendered_mask = []
        rendered_disparity = []

        with torch.no_grad():
            for i in range(0, render_size ** 2, batchsize):
                (rendered_color_i, rendered_mask_i,
                 rendered_disparity_i) = self.render(img_coord[:, :, :, i:i + batchsize],
                                                     pose_to_camera[:1],
                                                     inv_intrinsics,
                                                     bone_length=bone_length,
                                                     world_pose=world_pose,
                                                     thres=thres,
                                                     render_scale=render_scale, Nc=Nc, Nf=Nf,
                                                     semantic_map=semantic_map)
                rendered_color.append(rendered_color_i)
                rendered_mask.append(rendered_mask_i)
                rendered_disparity.append(rendered_disparity_i)

            rendered_color = torch.cat(rendered_color, dim=2)
            rendered_mask = torch.cat(rendered_mask, dim=1)
            rendered_disparity = torch.cat(rendered_disparity, dim=1)

        return (rendered_color.reshape(3, render_size, render_size),  # 3 x size x size
                rendered_mask.reshape(render_size, render_size),  # size x size
                rendered_disparity.reshape(render_size, render_size))  # size x size

    def bilinear_sample_tri_plane(self, points, feat_xy, feat_yz, feat_xz):
        batchsize, _, n = points.shape      # (b, h x w x num_steps, 3)     b x num_bone*3 x num_sampling
        points = points.reshape(batchsize, self.num_bone, 3, -1).permute(0, 1, 3, 2)
        points = F.normalize(points, dim=-1)        # b, bones, n, 3
        
        x = points[ ... , 0:1]  # b, hw, n
        y = points[ ... , 1:2] 
        z = points[ ... , 2:3] 
        xy = torch.cat([x, y], dim=-1)  # b, bones, n, 2
        xz = torch.cat([x, z], dim=-1)
        yz = torch.cat([y, z], dim=-1)

        xy_f = F.grid_sample(feat_xy, grid=xy, mode='bilinear', align_corners=True)  # b, c, h, w   # padding_mode='border') 
        xz_f = F.grid_sample(feat_xz, grid=xz, mode='bilinear', align_corners=True)
        yz_f = F.grid_sample(feat_yz, grid=yz, mode='bilinear', align_corners=True)

        xyz_f = [xy_f, yz_f, xz_f]

        #xyz_f = xyz_f.reshape(batchsize, 32, self.num_bone, -1)
        return xyz_f
    


class Decoder(nn.Module):
    def __init__(self, 
        in_c: int, 
        mid_c: int, 
        out_c: int, 
        num_layers = 3,
        activation ='relu', 
    ):
        super().__init__()
        num_layers = num_layers
        self.num_layers = num_layers
        self.out_c = out_c
        self.fc0 = self.create_block(in_c, mid_c, activation=activation)
        for idx in range(1, num_layers - 1):
            layer = self.create_block(mid_c, mid_c, activation=activation)
            setattr(self, f'fc{idx}', layer)
        setattr(self, f'fc{num_layers - 1}', self.create_block(mid_c, out_c, activation='none'))

    def create_block(self, in_features: int, out_features: int, activation: str):
        if activation == 'relu':
            return torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features), 
                torch.nn.ReLU()
            )
        elif activation == 'softmax':
            return torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features), 
                torch.nn.Softmax(dim=-1)
            )
        elif activation == 'softplus':
            return torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features), 
                torch.nn.Softplus()
            )
        elif activation == 'none':
            return torch.nn.Linear(in_features, out_features)
        else:
            raise NotImplementedError()
    
    def forward(self, feature: torch.Tensor):
        x = feature

        bs_n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, c)

        # Main layers
        for idx in range(self.num_layers - 1):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)
        x = getattr(self, f'fc{self.num_layers - 1}')(x)
        o = x
        o = o.reshape(bs_n, h, w, self.out_c).permute(0, 3, 1, 2) # bs_n, c ,h, w
        return o

class LightweightDecoder(nn.Module):
    def __init__(self):
        super().__init__(in_c: int, 
                        mid_c: int, 
                        out_c: int, 
                        num_layers = 3,
                        activation ='relu', )
        self.decoder = ModulatedConv2d(in_channel=32, out_channel=4, kernel_size=3, style_dim=self.c_dim)

    def forward(self):
        return
