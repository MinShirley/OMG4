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

import os
import random
import torch
from torch import nn
from utils.loss_utils import l1_loss, ssim, msssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from PIL import Image
import torchvision.transforms as T

def calc_gradient(dataset, opt, pipe, scene, gaussians, batch_size, bg_color, background):

    training_dataset = scene.getTrainCameras()
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=12 if dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)
     
    
    ###grad 측정
    N = gaussians._xyz.shape[0]
    train_cameras = scene.train_cameras[1.0]
    timestamps = sorted(set(cam.timestamp for cam in train_cameras))
    T = len(timestamps)

    training_dataset = scene.getTrainCameras()
    viewspace_grad = torch.zeros((N, T), dtype=torch.float32, device='cuda')
    t_grad = torch.zeros((N, T), dtype=torch.float32, device='cuda')

    for idx in tqdm(range(len(training_dataset)), desc="Computing Gradients"):
        gt_image, viewpoint_cam = training_dataset[idx]
        gt_image= gt_image.cuda()
        viewpoint_cam = viewpoint_cam.cuda()
        timestamp = viewpoint_cam.timestamp
        index = timestamps.index(timestamp)


        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg["depth"]
        alpha = render_pkg["alpha"]

        # Loss
        Ll1 = l1_loss(image, gt_image)
        Lssim = 1.0 - ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

        loss = loss / batch_size
        loss.backward()


        batch_point_grad = (torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1)) ##batch 1일때는 grad:2 안하는데 확인
        viewspace_grad[:, index] += batch_point_grad
        t_grad[:, index]  += gaussians._t.grad.clone().detach().squeeze(1)  # abs 추가


    final_view_grad = viewspace_grad.sum(dim = 1)
    final_t_grad = t_grad.sum(dim = 1)
    
    if torch.is_tensor(final_view_grad):
        final_view_grad = final_view_grad.detach().cpu().numpy()
    if torch.is_tensor(final_t_grad):
        final_t_grad = final_t_grad.detach().cpu().numpy()

    os.makedirs(os.path.join(scene.model_path, "gradient"), exist_ok=True)

    np.save(os.path.join(scene.model_path, "gradient/view_grad.npy"), final_view_grad)
    np.save(os.path.join(scene.model_path, "gradient/t_grad.npy"), final_t_grad)

    return final_view_grad, final_t_grad

     