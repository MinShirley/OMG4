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
from torch.nn import functional as F
import math
from .diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, eval_shfs_4d

def _segment_sum(x, group_ids, G):
    """Per-group sum via index_add_. x: [M,...] -> out: [G,...]."""
    out = torch.zeros((G,) + x.shape[1:], device=x.device, dtype=x.dtype)
    out.index_add_(0, group_ids, x)
    return out  # TODO define in utils (also used in gaussian_model.py)

def norm_weights(logits, group_ids, G):
    raw = torch.sigmoid(logits) + 1e-12
    denom = _segment_sum(raw, group_ids, G) + 1e-12
    return raw / denom[group_ids]


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if not pipe.env_map_res else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sh_degree_t=pc.active_sh_degree_t,
        campos=viewpoint_camera.camera_center,
        timestamp=viewpoint_camera.timestamp,
        time_duration=pc.time_duration[1]-pc.time_duration[0],
        rot_4d=pc.rot_4d,
        gaussian_dim=pc.gaussian_dim,
        force_sh_3d=pc.force_sh_3d,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    scales_t = None
    rotations = None
    rotations_r = None
    ts = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        if pc.rot_4d:
            cov3D_precomp, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
            means3D = means3D + delta_mean
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        if pc.gaussian_dim == 4:
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
            # marginal_t = torch.clamp_max(marginal_t, 1.0) # NOTE: 这里乘完会大于1，绝对不行——marginal_t应该用个概率而非概率密度 暂时可以clamp一下，后期用积分 —— 2d 也用的clamp
            opacity = opacity * marginal_t
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if pc.gaussian_dim == 4:
            scales_t = pc.get_scaling_t
            ts = pc.get_t
            if pc.rot_4d:
                rotations_r = pc.get_rotation_r

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            if pipe.compute_cov3D_python:
                dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            else:
                _, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
                dir_pp = ((means3D + delta_mean) - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            if pc.gaussian_dim == 3 or pc.force_sh_3d:
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            elif pc.gaussian_dim == 4:
                dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
                sh2rgb = eval_shfs_4d(pc.active_sh_degree, pc.active_sh_degree_t, shs_view, dir_pp_normalized, dir_t, pc.time_duration[1] - pc.time_duration[0])
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            if pc.gaussian_dim == 4 and ts is None:
                ts = pc.get_t
    else:
        colors_precomp = override_color
    
    flow_2d = torch.zeros_like(pc.get_xyz[:,:2])

    
    # AC - network
    if pc.net_enabled == True:
        timestamp = viewpoint_camera.timestamp
        time_duration = pc.time_duration
        time_min, time_max = time_duration[0], time_duration[1]
        timestamp_norm = (timestamp - time_min) / (time_max - time_min)
        xyz = pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
        t_col = torch.full((xyz.shape[0], 1), timestamp_norm, device=xyz.device)
        xyzt = torch.cat([xyz, t_col], dim=1) #N,4

        cont_feature = pc.mlp_cont(xyzt)
        space_feature = torch.cat([cont_feature, pc._features_static],dim=-1)
        view_feature = torch.cat([cont_feature, pc._features_view],dim=-1)
        if pc.gaussian_dim == 4 and ts is None:
            ts = pc.get_t

        shs = pc.mlp_view(view_feature).reshape(-1,47,3).float()
        dc = pc.mlp_dc(space_feature).reshape(-1,1,3).float()
        opacity = pc.opacity_activation(pc.mlp_opacity(space_feature).float())                
        shs = torch.cat([dc, shs], dim=1)
 
    
    # AC - SVQ
    if pc.vq_enabled:
        scales = pc.get_svq_scale
        rotations = pc.get_svq_rotation    
        app_feature = pc.get_svq_appearance
        space_feature = torch.cat([cont_feature, app_feature[:,0:3]],dim=-1)
        view_feature = torch.cat([cont_feature, app_feature[:,3:6]],dim=-1)

        shs = pc.mlp_view(view_feature).reshape(-1,47,3).float()
        dc = pc.mlp_dc(space_feature).reshape(-1,1,3).float()
        opacity = pc.opacity_activation(pc.mlp_opacity(space_feature).float())                
        shs = torch.cat([dc, shs], dim=1)

        if hasattr(pc, "rotation_r_codes"):
            rotations_r = pc.get_svq_rotation_r
            scales_t = pc.get_svq_scale_t


        
    # Merging (temporary merge)
    if pc.training_alpha:
        N = pc._xyz.shape[0]
        remove = torch.zeros(N, dtype=torch.bool, device="cuda")

        new_means3D = pc._xyz.clone()
        new_sh_dc = pc._features_dc.clone()
        new_sh_rest = pc._features_rest.clone()
        new_opacity = pc._opacity.clone()        if pc._opacity is not None else None
        new_scales = pc._scaling.clone()         if pc._scaling is not None else None
        new_rot = pc._rotation.clone()      if pc._rotation is not None else None
        new_colors = colors_precomp.clone() if colors_precomp is not None else None
        new_scales_t = pc._scaling_t.clone()      if pc._scaling_t is not None else None
        new_rot_r = pc._rotation_r.clone()   if pc._rotation_r is not None else None
        new_ts = pc._t.clone()  if pc._t is not None else None

        members = pc._merge_members
        group_ids = pc._merge_group_ids
        G = pc._merge_G
        reps = getattr(pc, "_merge_reps_fixed", None)

        w_xyz  = norm_weights(pc.xyz_w_logits, group_ids, G)
        w_dc   = norm_weights(pc.dc_w_logits, group_ids, G)
        w_rest = norm_weights(pc.rest_w_logits, group_ids, G)

        # gather member slices
        xyz_mem = new_means3D[members] # [M,3]
        dc_mem  = new_sh_dc[members]
        rest_mem= new_sh_rest[members]

        # merge xyz
        mu_bar = _segment_sum(w_xyz[:, None] * xyz_mem, group_ids, G)  # [G,3]
        new_means3D[reps] = mu_bar

        # merge SH dc/rest (flatten -> segment -> reshape)
        dc_shape = dc_mem.shape
        dc_bar = _segment_sum(w_dc[:, None] * dc_mem.reshape(dc_mem.shape[0], -1), group_ids, G).view(G, *dc_shape[1:])
        new_sh_dc[reps] = dc_bar

        rest_shape = rest_mem.shape
        rest_bar = _segment_sum(w_rest[:, None] * rest_mem.reshape(rest_mem.shape[0], -1), group_ids, G).view(G, *rest_shape[1:])
        new_sh_rest[reps] = rest_bar

        remove[members] = True
        remove[reps] = False
        keep = ~remove  # unclustered + representatives

        means3D = new_means3D[keep]
        new_sh_dc   = new_sh_dc[keep]
        new_sh_rest = new_sh_rest[keep]
        
        shs = torch.cat((new_sh_dc, new_sh_rest), dim=1)

        if new_opacity is not None:
            opacity = pc.opacity_activation(new_opacity[keep])
        if new_scales is not None:
            scales = pc.scaling_activation(new_scales[keep])
        if new_rot is not None:
            rotations = pc.rotation_activation(new_rot[keep])
        if new_colors is not None:
            colors_precomp = new_colors[keep]
        if new_scales_t is not None:
            scales_t = pc.scaling_activation(new_scales_t[keep])
        if new_rot_r is not None:
            rotations_r = pc.rotation_activation(new_rot_r[keep])
        if new_ts is not None:
            ts = new_ts[keep]

        means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        flow_2d = torch.zeros_like(means3D[:,:2], device="cuda")


    ####
    # Prefilter
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        mask = marginal_t[:,0] > 0.05
        if means2D is not None:
            means2D = means2D[mask]
        if means3D is not None:
            means3D = means3D[mask]
        if ts is not None:
            ts = ts[mask]
        if shs is not None:
            shs = shs[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]
        if opacity is not None:
            opacity = opacity[mask]
        if scales is not None:
            scales = scales[mask]
        if scales_t is not None:
            scales_t = scales_t[mask]
        if rotations is not None:
            rotations = rotations[mask]
        if rotations_r is not None:
            rotations_r = rotations_r[mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[mask]
        if flow_2d is not None:
            flow_2d = flow_2d[mask]


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha, flow, covs_com, \
    accum_weights_ptr, accum_weights_count, accum_max_count = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        flow_2d = flow_2d,
        opacities = opacity,
        ts = ts,
        scales = scales,
        scales_t = scales_t,
        rotations = rotations,
        rotations_r = rotations_r,
        cov3D_precomp = cov3D_precomp)

    
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        radii_all = radii.new_zeros(mask.shape)
        radii_all[mask] = radii
    else:
        radii_all = radii

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii_all > 0,
            "radii": radii_all,
            "depth": depth,
            "alpha": alpha,
            "flow": flow,
            "accum_weights": accum_weights_ptr,
            "area_proj": accum_weights_count,
            "area_max": accum_max_count,
            }

