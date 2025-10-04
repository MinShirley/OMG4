#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import math
import torch
import tinycudann as tcnn
import cupy as cp
import os
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_rotation_4d, build_scaling_rotation_4d
from torch import nn
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import sh_channels_4d
from collections import defaultdict
from utils.compress_utils import *
from cuml.cluster import KMeans
from utils.gpcc_utils import compress_gpcc, decompress_gpcc, calculate_morton_order, float16_to_uint16, uint16_to_float16
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from tqdm import tqdm

def _flatten_groups_to_ragged(groups):
    device = groups[0].device if len(groups) else torch.device('cuda')
    G = len(groups)
    if G == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, 0, empty, empty
    members = torch.cat(groups, dim=0)
    counts = torch.tensor([g.numel() for g in groups], device=device, dtype=torch.long)
    starts = torch.cat([torch.tensor([0], device=device, dtype=torch.long), counts.cumsum(0)[:-1]], dim=0)
    group_ids = torch.repeat_interleave(torch.arange(G, device=device, dtype=torch.long), counts)
    return members, group_ids, G, starts, counts


def _segment_sum(x, group_ids, G):
    out = torch.zeros((G,) + x.shape[1:], device=x.device, dtype=x.dtype)
    out.index_add_(0, group_ids, x)
    return out

def representatives_by_argmax(members, group_ids, G, w_xyz, eps=1e-12):
    device = w_xyz.device
    if members.numel() == 0 or G == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    neg_inf = torch.finfo(torch.float32).min
    max_w = torch.full((G,), neg_inf, device=device, dtype=torch.float32)
    max_w.scatter_reduce_(0, group_ids, w_xyz.to(torch.float32), reduce="amax", include_self=True)
    is_max = w_xyz >= (max_w[group_ids] - eps)
    neg_inf64 = torch.finfo(torch.float64).min
    cand = torch.where(is_max, -members.to(torch.float64), torch.full((members.numel(),), neg_inf64, device=device, dtype=torch.float64))
    best = torch.full((G,), neg_inf64, device=device, dtype=torch.float64)
    best.scatter_reduce_(0, group_ids, cand, reduce="amax", include_self=True)
    reps = (-best).to(torch.long)
    return reps


def representatives(self, eps=1e-12):
    members = self._merge_members
    group_ids = self._merge_group_ids
    G = self._merge_G
    starts = self._merge_starts
    raw = torch.sigmoid(self.xyz_w_logits) + 1e-12
    denom = _segment_sum(raw, group_ids, G) + 1e-12
    w_xyz = raw / denom[group_ids]
    return representatives_by_argmax(members, group_ids, G, w_xyz, eps)

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L.transpose(1, 2) @ L
            symm = strip_symmetric(actual_covariance)
            return symm
        
        def build_covariance_from_scaling_rotation_4d(scaling, scaling_modifier, rotation_l, rotation_r, dt=0.0):
            L = build_scaling_rotation_4d(scaling_modifier * scaling, rotation_l, rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            cov_11 = actual_covariance[:,:3,:3]
            cov_12 = actual_covariance[:,0:3,3:4]
            cov_t = actual_covariance[:,3:4,3:4]
            current_covariance = cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t
            symm = strip_symmetric(current_covariance)
            if dt.shape[1] > 1:
                mean_offset = (cov_12.squeeze(-1) / cov_t.squeeze(-1))[:, None, :] * dt[..., None]
                mean_offset = mean_offset[..., None]  # [num_pts, num_time, 3, 1]
            else:
                mean_offset = cov_12.squeeze(-1) / cov_t.squeeze(-1) * dt
            return symm, mean_offset.squeeze(-1)
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        if not self.rot_4d:
            self.covariance_activation = build_covariance_from_scaling_rotation
        else:
            self.covariance_activation = build_covariance_from_scaling_rotation_4d

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, gaussian_dim : int = 3, time_duration: list = [-0.5, 0.5], rot_4d: bool = False, force_sh_3d: bool = False, sh_degree_t : int = 0):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self.gaussian_dim = gaussian_dim
        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self._rotation_r = torch.empty(0)
        self.force_sh_3d = force_sh_3d
        self.t_gradient_accum = torch.empty(0)
        if self.rot_4d or self.force_sh_3d:
            assert self.gaussian_dim == 4
        self.env_map = torch.empty(0)
        
        self.active_sh_degree_t = 0
        self.max_sh_degree_t = sh_degree_t
        
        self.setup_functions()
        self.training_alpha = False

        self.max_sh_rest = (sh_degree+1)**2 - 1
        self.net_enabled = False
        self.temp_net = False
        self.vq_enabled = False

    def capture(self):
        if self.gaussian_dim == 3:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        elif self.gaussian_dim == 4:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.t_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self._t,
                self._scaling_t,
                self._rotation_r,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t
            )
    
    def restore(self, model_args, training_args):
        if self.gaussian_dim == 3:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        elif self.gaussian_dim == 4:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            t_gradient_accum,
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            self._t,
            self._scaling_t,
            self._rotation_r,
            self.rot_4d,
            self.env_map,
            self.active_sh_degree_t) = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.t_gradient_accum = t_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)
    
    @property
    def get_scaling_xyzt(self):
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim = 1))
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_rotation_r(self):
        return self.rotation_activation(self._rotation_r)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_t(self):
        return self._t
    
    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim = 1)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.max_sh_degree+1)**2
        elif self.gaussian_dim == 4 and self.max_sh_degree_t == 0:
            return sh_channels_4d[self.max_sh_degree]
        elif self.gaussian_dim == 4 and self.max_sh_degree_t > 0:
            return (self.max_sh_degree+1)**2 * (self.max_sh_degree_t + 1)
    
    def get_cov_t(self, scaling_modifier = 1):
        if self.rot_4d:
            L = build_scaling_rotation_4d(scaling_modifier * self.get_scaling_xyzt, self._rotation, self._rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance[:,3,3].unsqueeze(1)
        else:
            return self.get_scaling_t * scaling_modifier

    def get_marginal_t(self, timestamp, scaling_modifier = 1): # Standard
        sigma = self.get_cov_t(scaling_modifier)
        return torch.exp(-0.5*(self.get_t-timestamp)**2/sigma) # / torch.sqrt(2*torch.pi*sigma)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_current_covariance_and_mean_offset(self, scaling_modifier = 1, timestamp = 0.0):
        return self.covariance_activation(self.get_scaling_xyzt, scaling_modifier, 
                                                              self._rotation, 
                                                              self._rotation_r,
                                                              dt = timestamp - self.get_t)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        elif self.max_sh_degree_t and self.active_sh_degree_t < self.max_sh_degree_t:
            self.active_sh_degree_t += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(pcd.time).cuda().float()
            
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
            dist_t = torch.zeros_like(fused_times, device="cuda") + (self.time_duration[1] - self.time_duration[0]) / 5
            scales_t = torch.log(torch.sqrt(dist_t))
            if self.rot_4d:
                rots_r = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots_r[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def create_from_pth(self, path, spatial_lr_scale):
        assert self.gaussian_dim == 4 and self.rot_4d
        self.spatial_lr_scale = spatial_lr_scale
        init_4d_gaussian = torch.load(path)
        fused_point_cloud = init_4d_gaussian['xyz'].cuda()
        features_dc = init_4d_gaussian['features_dc'].cuda()
        features_rest = init_4d_gaussian['features_rest'].cuda()
        fused_times = init_4d_gaussian['t'].cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = init_4d_gaussian['scaling'].cuda()
        rots = init_4d_gaussian['rotation'].cuda()
        scales_t = init_4d_gaussian['scaling_t'].cuda()
        rots_r = init_4d_gaussian['rotation_r'].cuda()

        opacities = init_4d_gaussian['opacity'].cuda()
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.transpose(1, 2).requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        self._t = nn.Parameter(fused_times.requires_grad_(True))
        self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def training_setup(self, training_args):

        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.gaussian_dim == 4: # TODO: tune time_lr_scale
            if training_args.position_t_lr_init < 0:
                training_args.position_t_lr_init = training_args.position_lr_init
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            l.append({'params': [self._t], 'lr': training_args.position_t_lr_init * self.spatial_lr_scale, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': training_args.scaling_lr, "name": "scaling_t"})
            if self.rot_4d:
                l.append({'params': [self._rotation_r], 'lr': training_args.rotation_lr, "name": "rotation_r"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }
        if self.gaussian_dim == 4:
            d["t"] = new_t
            d["scaling_t"] = new_scaling_t
            if self.rot_4d:
                d["rotation_r"] = new_rotation_r

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")
        
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        if not self.rot_4d:
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            if self.gaussian_dim == 4:
                stds_t = self.get_scaling_t[selected_pts_mask].repeat(N,1)
                means_t = torch.zeros((stds_t.size(0), 1),device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        else:
            stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 4),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation_4d(self._rotation[selected_pts_mask], self._rotation_r[selected_pts_mask]).repeat(N,1,1)
            new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyzt[selected_pts_mask].repeat(N, 1)
            new_xyz = new_xyzt[...,0:3]
            new_t = new_xyzt[...,3:4]
            new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation_r = self._rotation_r[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(grads >= grad_threshold, True, False).sum()}, num_to_clone_pos: {selected_pts_mask.sum()}")
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None, prune_only=False):
        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            if self.gaussian_dim == 4:
                grads_t = self.t_gradient_accum / self.denom
                grads_t[grads_t.isnan()] = 0.0
            else:
                grads_t = None

            self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t)
            self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def scaling_t_pruning(self, min_opacity, static_mask):
        prune_mask = (self.get_opacity < min_opacity).squeeze() & static_mask
        self.prune_points(prune_mask)


    def add_densification_stats(self, viewspace_point_tensor, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]
        
    def add_densification_stats_grad(self, viewspace_point_grad, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]



    def gradient_sampling(self, tau_GS=0.2, view_grad=None, t_grad=None, args=None): 
        view_grad=  np.load(os.path.join(args.grad,'view_grad.npy'))
        t_grad =  np.load(os.path.join(args.grad,'t_grad.npy'))
        view_grad = torch.from_numpy(view_grad).to("cuda")
        t_grad = torch.from_numpy(t_grad).to("cuda")

        view_grad_norm = (view_grad - view_grad.min()) / (view_grad.max() - view_grad.min() + 1e-8)
        t_grad_norm = (t_grad - t_grad.min()) / (t_grad.max() - t_grad.min() + 1e-8)

        st_score = view_grad_norm * t_grad_norm

        N = st_score.shape[0]
        K = int(N * tau_GS)

        topk = torch.topk(st_score, K, largest=True)
        keep_indices = topk.indices  
        keep_mask = torch.zeros(N, dtype=torch.bool).cuda()
        keep_mask[keep_indices] = True
        remove_mask = ~keep_mask

        self.prune_points(remove_mask)
        return ~remove_mask

    def gradient_pruning(self, view_grad=None, t_grad=None, v_cutoff=0.8, t_cutoff=0.8, args = None, mask=None): 
        view_grad = torch.from_numpy(view_grad).to("cuda")
        t_grad = torch.from_numpy(t_grad).to("cuda")

        if mask is not None:
            view_grad = view_grad[mask]
            t_grad = t_grad[mask]

        view_threshold = torch.quantile(view_grad, v_cutoff)
        t_threshold =  torch.quantile(t_grad, t_cutoff)

        low_grad_view = view_grad <  view_threshold 
        not_low_grad_view = ~low_grad_view

        t_high_grad = t_grad > t_threshold 

        keep_low_view_grad = low_grad_view & t_high_grad 


        remove_mask = torch.ones_like(view_grad, dtype=torch.bool)
        remove_mask[keep_low_view_grad] = False
        remove_mask[not_low_grad_view] = False        
        self.prune_points(remove_mask)


    def sort_morton(self):
        with torch.no_grad():
            xyz_q = (
                (2**21 - 1)
                * (self._xyz - self._xyz.min(0).values)
                / (self._xyz.max(0).values - self._xyz.min(0).values)
            ).long()
            order = mortonEncode(xyz_q).sort().indices
        return order


    @torch.no_grad()
    def calc_clusters(self,
        grid_size: float = 0.1,
        tau_sim: float = -0.1,
        sim_cutoff: float = 0.7,
        min_cluster_size: int = 2,
        t_grid_size: float = 0.0
    ):

        print("Merging start (unique, non-subset clusters)")
        N, device = self._xyz.shape[0], self._xyz.device
        keep_mask = torch.ones(N, dtype=torch.bool, device=device)

        # Grid cells
        coords_int = torch.floor(self._xyz / grid_size).to(torch.int64)
        cell_map = defaultdict(list)

        if t_grid_size:
            coords_int_t = torch.floor(self._t / t_grid_size).to(torch.int64)
            coords_final_int = torch.cat([coords_int, coords_int_t], dim=-1)
        else:
            coords_final_int = coords_int

        for idx, cell in enumerate(coords_final_int):
            cell_map[tuple(cell.tolist())].append(idx)

        num_points, device = self._xyz.shape[0], self._xyz.device
        sorted_xyz, _ = torch.sort(self._xyz, dim=0)
        st = int(num_points * 0.001)
        ed = int(num_points * 0.999)

        xyz_min = sorted_xyz[st]
        xyz_max = sorted_xyz[ed]
        xyz_range = xyz_max - xyz_min
        sh_min, sh_max = self._features_dc.min(dim=0).values, self._features_dc.max(dim=0).values
        sh_range = sh_max - sh_min

        cluster_sets = []  

        for idx_list in tqdm(cell_map.values()):
            if len(idx_list) < 2:
                continue
            idxs_all = torch.tensor(idx_list, device=device, dtype=torch.long)
            idxs = idxs_all[keep_mask[idxs_all]]
            m = idxs.numel()
            if m < 2:
                continue

            xyz = self._xyz[idxs]
            sh  = self._features_dc[idxs]

            xyz_n = (xyz - xyz_min) / (xyz_range + 1e-6)
            sh_n  = (sh  - sh_min) / (sh_range + 1e-6)

            pos_d2 = ((xyz_n[:, None, :] - xyz_n[None, :, :]) ** 2).sum(-1)  # [m,m]
            sh_d2  = ((sh_n[:,  None, :] - sh_n[ None, :, :]) ** 2).sum(-1).squeeze(-1)  # [m,m]
            sims   = (-pos_d2) - 4.0 * sh_d2

            sims.fill_diagonal_(float("-inf"))
            tril_indices = torch.tril_indices(m, m, offset=-1, device=sims.device)
            sims[tril_indices[0], tril_indices[1]] = float("-inf")

            valid = sims[~torch.isinf(sims)]
            if valid.numel() == 0:
                continue
            try:
                thr = max(tau_sim, torch.quantile(valid, sim_cutoff).item())
            except:
                thr = tau_sim

            for i in range(m):
                nbr = sims[i] >= thr
                nbr[i] = True
                members_local = torch.nonzero(nbr, as_tuple=False).flatten()
                if members_local.numel() < min_cluster_size:
                    continue
                members_global = idxs[members_local].tolist()
                cluster_sets.append(frozenset(members_global))

        print("Whole clusters found:", len(cluster_sets))

        unique_sets = list(set(cluster_sets))
        unique_sets.sort(key=len, reverse=True)
        maximal_sets = []
        for S in unique_sets:
            if any(S.issubset(T) for T in maximal_sets):
                continue
            maximal_sets.append(S)

        self.total_groups = [torch.tensor(sorted(list(S)), device=device, dtype=torch.long)
                            for S in maximal_sets if len(S) >= min_cluster_size]


        cell_to_group_indices = defaultdict(list)
        group_to_cell = []
        for gi, g in enumerate(self.total_groups):
            if g.numel() == 0:
                continue
            cell = tuple(torch.floor(self._xyz[g[0]] / grid_size).to(torch.int64).tolist())
            group_to_cell.append(cell)
            cell_to_group_indices[cell].append(gi)


        self._clusters_per_cell = {cell: len(glist) for cell, glist in cell_to_group_indices.items()}   
        self._cell_to_group_indices = dict(cell_to_group_indices)
        self._group_to_cell = group_to_cell

        if len(self.total_groups) > 0:
            all_members = torch.cat(self.total_groups, dim=0)
        else:
            all_members = torch.empty(0, dtype=torch.long, device=device)
        in_any_cluster = torch.zeros(N, dtype=torch.bool, device=device)
        if all_members.numel() > 0:
            in_any_cluster[all_members] = True
        self._nonclustered = (~in_any_cluster).nonzero(as_tuple=False).flatten()
        self._num_nonclustered = int(self._nonclustered.numel())

        num_cells_with_clusters = len(self._clusters_per_cell)
        counts = list(self._clusters_per_cell.values()) 
        avg_per_cell = (sum(counts) / max(num_cells_with_clusters, 1)) if counts else 0.0
        print(f"clusters ready: {len(self.total_groups)} groups | cells with clusters: {num_cells_with_clusters} | avg/occupied cell: {avg_per_cell:.2f} | unclustered: {self._num_nonclustered}/{N} ({(self._num_nonclustered/max(N,1)):.1%})")

    def set_alpha_groups(self, alpha_lr=0.01):
        if not hasattr(self, "total_groups") or len(self.total_groups) == 0:
            raise ValueError("total_groups is empty. Run calc_clusters() first.")
        device = self._xyz.device

        members, group_ids, G, starts, counts = _flatten_groups_to_ragged(self.total_groups)
        M = members.numel()
        if M == 0:
            raise ValueError("No members in clusters.")

        # save ragged state
        self._merge_members = members
        self._merge_group_ids = group_ids
        self._merge_G = G
        self._merge_starts = starts
        self._merge_counts = counts

        # learnable logits
        self.xyz_w_logits  = torch.nn.Parameter(torch.zeros(M, device=device))
        self.dc_w_logits   = torch.nn.Parameter(torch.zeros(M, device=device))
        self.rest_w_logits = torch.nn.Parameter(torch.zeros(M, device=device))


        self.optimizer.add_param_group({'params': [self.xyz_w_logits],  'lr': alpha_lr, 'name': 'xyz_w_logits'})
        self.optimizer.add_param_group({'params': [self.dc_w_logits],   'lr': alpha_lr, 'name': 'dc_w_logits'})
        self.optimizer.add_param_group({'params': [self.rest_w_logits], 'lr': alpha_lr, 'name': 'rest_w_logits'})

        self.training_alpha = True

        # Set representatives
        with torch.no_grad():
            raw = torch.sigmoid(self.xyz_w_logits) + 1e-12
            denom = _segment_sum(raw, group_ids, G) + 1e-12
            w_xyz0 = raw / denom[group_ids]
        self._merge_reps_fixed = representatives_by_argmax(members, group_ids, G, w_xyz0)

    def alpha_pruning_groups(self):
        if not hasattr(self, '_merge_members'):
            raise RuntimeError('No cluster state. Call set_alpha_groups() first.')

        device = self._xyz.device
        members = self._merge_members
        group_ids = self._merge_group_ids
        G = self._merge_G
        starts = self._merge_starts

        reps = getattr(self, '_merge_reps_fixed', None)
        if reps is None:
            reps = self.representatives()

        def norm_weights(logits: torch.Tensor):
            raw = torch.sigmoid(logits) + 1e-12
            denom = _segment_sum(raw, group_ids, G) + 1e-12
            return raw / denom[group_ids]

        w_xyz  = norm_weights(self.xyz_w_logits)
        w_dc   = norm_weights(self.dc_w_logits)
        w_rest = norm_weights(self.rest_w_logits)
        self._merge_w_xyz, self._merge_w_dc, self._merge_w_rest = w_xyz, w_dc, w_rest

        xyz_mem = self._xyz[members]
        dc_mem  = self._features_dc[members]
        rest_mem= self._features_rest[members]

        xyz_bar = _segment_sum(w_xyz[:, None] * xyz_mem, group_ids, G)
        dc_bar = _segment_sum(w_dc[:, None]  * dc_mem.reshape(dc_mem.shape[0], -1), group_ids, G).view(len(reps), *dc_mem.shape[1:])
        rest_bar = _segment_sum(w_rest[:, None]* rest_mem.reshape(rest_mem.shape[0], -1), group_ids, G).view(len(reps), *rest_mem.shape[1:])


        self._xyz[reps] = xyz_bar
        self._features_dc[reps] = dc_bar
        self._features_rest[reps] = rest_bar

        remove = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device=device)
        remove[members] = True
        remove[reps] = False

        self.optimizer.param_groups = [
            group for group in self.optimizer.param_groups
            if group.get("name") != "xyz_w_logits"
            if group.get("name") != "dc_w_logits"
            if group.get("name") != "rest_w_logits"
        ]

        for key in list(self.optimizer.state.keys()):
            if key is self.xyz_w_logits:
                del self.optimizer.state[key]
            elif key is self.dc_w_logits:
                del self.optimizer.state[key]
            elif key is self.rest_w_logits:
                del self.optimizer.state[key]

        self.prune_points(remove)

        self.training_alpha = False


    def construct_net(self, train=True):
        # Default hyperparameter from OMG (https://github.com/maincold2/OMG)
        self.mlp_cont = tcnn.NetworkWithInputEncoding(
            n_input_dims=4,
            n_output_dims=13,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": 16,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        self.mlp_view = tcnn.Network(
            n_input_dims=16,
            n_output_dims=3*47,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        self.mlp_dc = tcnn.Network(
            n_input_dims=16,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        self.mlp_opacity = tcnn.Network(
            n_input_dims=16,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        if train:
            self.net_enabled = True
            self._features_static = nn.Parameter(self._features_dc[:, 0].clone().detach(),requires_grad=True) #(N,)
            self._features_view = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 3), device="cuda").requires_grad_(True)) #(N,3)
        

            if not hasattr(self, 'optimizer') or self.optimizer is None:
                raise RuntimeError("Optimizer must be initialized before calling set_alpha().")
            feature_lr = 0.0025

            self.optimizer.add_param_group({
                'params': [self._features_static],
                'lr': feature_lr,
                'name': 'f_static' 
            })

            self.optimizer.add_param_group({
                'params': [self._features_view],
                'lr': feature_lr,
                'name': 'f_view' 
            })

            mlp_params = []
            for params in self.mlp_cont.parameters():
                mlp_params.append(params)
            for params in self.mlp_view.parameters():
                mlp_params.append(params)
            for params in self.mlp_dc.parameters():
                mlp_params.append(params)
            for params in self.mlp_opacity.parameters():
                mlp_params.append(params)
                
            self.optimizer_net = torch.optim.Adam(mlp_params, lr=0.005, eps=1e-15) # TODO
            self.scheduler_net = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                self.optimizer_net, start_factor=0.01, total_iters=100
            ),
                torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_net,
                milestones=[1_000, 3_500, 6_000],
                gamma=0.33,
            ),
            ]
            )

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        dim = 3
        aabb_min, aabb_max = torch.split(aabb, dim, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5 
            return x

    def apply_svq_3d(self, args):
        self.scale_codes, self.rotation_codes, self.appearance_codes = [], [], []
        self.scale_indices, self.rotation_indices, self.appearance_indices = [], [], []
        code_params = []
        self.kmeans(self._scaling, self.scale_codes, self.scale_indices, args.slice_scale, eval(args.cluster_scale), code_params)
        self.kmeans(self._rotation, self.rotation_codes, self.rotation_indices, args.slice_rot, eval(args.cluster_rot), code_params)

        self.kmeans(torch.cat([self._features_static, self._features_view], dim=-1),
                    self.appearance_codes, self.appearance_indices, args.slice_app, eval(args.cluster_app), code_params)
        self.optimizer_code = torch.optim.Adam(code_params, lr=1e-4, eps=1e-8)
        self.vq_enabled = True

    def apply_svq_4d(self, args):
        self.scaling_t_codes, self.rotation_r_codes , self.t_codes= [], [], []
        self.scaling_t_indices, self.rotation_r_indices, self.t_indices = [], [], []

        code_params_4d = []
        self.kmeans(self._scaling_t, self.scaling_t_codes, self.scaling_t_indices, args.slice_scale_t, eval(args.cluster_scale_t), code_params_4d)
        self.kmeans(self._rotation_r, self.rotation_r_codes, self.rotation_r_indices, args.slice_rot_r, eval(args.cluster_rot_r), code_params_4d)

        self.optimizer_code_4d = torch.optim.Adam(code_params_4d, lr=1e-4, eps=1e-8)

    def kmeans(self, param_data, code_list, index_list, svq_len, n_clusters, code_params):
        assert param_data.shape[1] % svq_len == 0, "invalid sub-vector length"
        for i in range(param_data.shape[1]//svq_len):
            input_cp = cp.asarray(param_data[:, i*svq_len:(i+1)*svq_len].detach().cpu())
            kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, n_init=1)
            labels = kmeans.fit_predict(input_cp)
            cluster_centers = kmeans.cluster_centers_

            codebook = torch.nn.Parameter(torch.from_dlpack(cluster_centers)).cuda()
            index = torch.from_dlpack(labels).cuda().long()

            code_list.append(codebook)
            index_list.append(index)
            code_params.append(codebook) 

    @property    
    def get_svq_t(self):
        t = []
        for i in range(len(self.t_codes)):
            t.append(self.t_codes[i][self.t_indices[i]])
        return torch.cat(t, dim=-1)

    @property
    def get_svq_scale(self):
        scale = []
        for i in range(len(self.scale_codes)):
            scale.append(self.scale_codes[i][self.scale_indices[i]])
        return self.scaling_activation(torch.cat(scale, dim=-1))

    @property    
    def get_svq_scale_t(self):
        scale_t = []
        for i in range(len(self.scaling_t_codes)):
            scale_t.append(self.scaling_t_codes[i][self.scaling_t_indices[i]])
        return self.scaling_activation(torch.cat(scale_t, dim=-1))

    @property
    def get_svq_rotation(self):
        rotation = []
        for i in range(len(self.rotation_codes)):
            rotation.append(self.rotation_codes[i][self.rotation_indices[i]])
        return self.rotation_activation(torch.cat(rotation, dim=-1))

    @property
    def get_svq_rotation_r(self):
        rotation_r = []
        for i in range(len(self.rotation_r_codes)):
            rotation_r.append(self.rotation_r_codes[i][self.rotation_r_indices[i]])
        return self.rotation_activation(torch.cat(rotation_r, dim=-1))
    
    @property
    def get_svq_appearance(self):
        appearance = []
        for i in range(len(self.appearance_codes)):
            appearance.append(self.appearance_codes[i][self.appearance_indices[i]])
        return torch.cat(appearance, dim=-1)

    def sort_attribute(self, order, xyz_only=False):
        self._xyz = nn.Parameter(self._xyz[order], requires_grad=True)
        if not xyz_only:
            self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
            self._rotation = nn.Parameter(self._rotation[order], requires_grad=True)

            self._features_static = nn.Parameter(self._features_static[order], requires_grad=True)
            self._features_view = nn.Parameter(self._features_view[order], requires_grad=True)
            for i in range(len(self.scale_indices)):
                self.scale_indices[i] = self.scale_indices[i][order]
            for i in range(len(self.rotation_indices)):
                self.rotation_indices[i] = self.rotation_indices[i][order]
            for i in range(len(self.appearance_indices)):
                self.appearance_indices[i] = self.appearance_indices[i][order]
        return

    def encode(self):
        save_dict = dict()
        save_dict['xyz'] = self.get_xyz.half()

        save_dict['scale_code'] = []
        save_dict['scale_index'] = []
        save_dict['scale_htable'] = []
        for i in range(len(self.scale_codes)):
            save_dict['scale_code'].append(self.scale_codes[i].half().cpu().numpy())
            huf_idx, huf_tab = huffman_encode(self.scale_indices[i].cpu().numpy())
            save_dict['scale_index'].append(huf_idx)
            save_dict['scale_htable'].append(huf_tab)

        save_dict['rotation_code'] = []
        save_dict['rotation_index'] = []
        save_dict['rotation_htable'] = []
        for i in range(len(self.rotation_codes)):
            save_dict['rotation_code'].append(self.rotation_codes[i].half().cpu().numpy())
            huf_idx, huf_tab = huffman_encode(self.rotation_indices[i].cpu().numpy())
            save_dict['rotation_index'].append(huf_idx)
            save_dict['rotation_htable'].append(huf_tab)

        save_dict['app_code'] = []
        save_dict['app_index'] = []
        save_dict['app_htable'] = []
        for i in range(len(self.appearance_codes)):
            save_dict['app_code'].append(self.appearance_codes[i].half().cpu().numpy())
            huf_idx, huf_tab = huffman_encode(self.appearance_indices[i].cpu().numpy())
            save_dict['app_index'].append(huf_idx)
            save_dict['app_htable'].append(huf_tab)
        
        save_dict['t'] = self._t.half().cpu().numpy() 
        

        save_dict['rotation_r_code'] = []
        save_dict['rotation_r_index'] = []
        save_dict['rotation_r_htable'] = []
        for i in range(len(self.rotation_r_codes)):
            save_dict['rotation_r_code'].append(self.rotation_r_codes[i].half().cpu().numpy())
            huf_idx, huf_tab = huffman_encode(self.rotation_r_indices[i].cpu().numpy())
            save_dict['rotation_r_index'].append(huf_idx)
            save_dict['rotation_r_htable'].append(huf_tab)


        save_dict['scaling_t_code'] = []
        save_dict['scaling_t_index'] = []
        save_dict['scaling_t_htable'] = []
        for i in range(len(self.scaling_t_codes)):
            save_dict['scaling_t_code'].append(self.scaling_t_codes[i].half().cpu().numpy())
            huf_idx, huf_tab = huffman_encode(self.scaling_t_indices[i].cpu().numpy())
            save_dict['scaling_t_index'].append(huf_idx)
            save_dict['scaling_t_htable'].append(huf_tab)

                                           
        save_dict['MLP_cont'] = self.mlp_cont.params.detach().clone().half().cpu().numpy()
        save_dict['MLP_dc'] = self.mlp_dc.params.detach().clone().half().cpu().numpy()
        save_dict['MLP_sh'] = self.mlp_view.params.detach().clone().half().cpu().numpy()
        save_dict['MLP_opacity'] = self.mlp_opacity.params.detach().clone().half().cpu().numpy()     
                
        return save_dict

    def decode(self, save_dict, decompress=True):
        self.vq_enabled = False
        self.net_enabled = True
        
        self._xyz = save_dict['xyz'].cuda().float()

        scale = []
        rotation = []
        appearance = []
        scaling_t = []
        rotation_r = []

        if decompress:

            for i in range(len(save_dict['scale_code'])):
                labels = huffman_decode(save_dict['scale_index'][i], save_dict['scale_htable'][i])
                cluster_centers = save_dict['scale_code'][i]
                scale.append(torch.tensor(cluster_centers[labels]).cuda())
            self._scaling = torch.cat(scale, dim=-1).float()
            
            for i in range(len(save_dict['rotation_code'])):
                labels = huffman_decode(save_dict['rotation_index'][i], save_dict['rotation_htable'][i])
                cluster_centers = save_dict['rotation_code'][i]
                rotation.append(torch.tensor(cluster_centers[labels]).cuda())
            self._rotation = torch.cat(rotation, dim=-1).float()
            
            for i in range(len(save_dict['app_code'])):
                labels = huffman_decode(save_dict['app_index'][i], save_dict['app_htable'][i])
                cluster_centers = save_dict['app_code'][i]
                appearance.append(torch.tensor(cluster_centers[labels]).cuda())
            app_feature = torch.cat(appearance, dim=-1).float()


            for i in range(len(save_dict['scaling_t_code'])):
                labels = huffman_decode(save_dict['scaling_t_index'][i], save_dict['scaling_t_htable'][i])
                cluster_centers = save_dict['scaling_t_code'][i]
                scaling_t.append(torch.tensor(cluster_centers[labels]).cuda())
            self._scaling_t = torch.cat(scaling_t, dim=-1).float()

            for i in range(len(save_dict['rotation_r_code'])):
                labels = huffman_decode(save_dict['rotation_r_index'][i], save_dict['rotation_r_htable'][i])
                cluster_centers = save_dict['rotation_r_code'][i]
                rotation_r.append(torch.tensor(cluster_centers[labels]).cuda())
            self._rotation_r = torch.cat(rotation_r, dim=-1).float()



            if not hasattr(self, "mlp_cont"):
                self.construct_net(train=True)
            self.mlp_cont.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_cont']).cuda().half().requires_grad_(True))
            self.mlp_dc.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_dc']).cuda().half().requires_grad_(True))
            self.mlp_view.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_sh']).cuda().half().requires_grad_(True))
            self.mlp_opacity.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_opacity']).cuda().half().requires_grad_(True))

            self._t = torch.from_numpy(save_dict['t']).float().cuda()

            N = app_feature.shape[0]
            self._features_static = nn.Parameter(app_feature[:, 0:3].clone().detach().cuda().requires_grad_(True))  # [N, 3]
            self._features_view = nn.Parameter(app_feature[:, 3:6].clone().detach().cuda().requires_grad_(True))  