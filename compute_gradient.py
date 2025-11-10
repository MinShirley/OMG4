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

def main(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size, out_path = None):
    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]
    
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)
    gaussians.training_setup(opt)
    if out_path is not None:
        scene.model_path = out_path
    os.makedirs(scene.model_path, exist_ok=True)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

        print("load successfully", checkpoint)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    os.makedirs(os.path.join(dataset.model_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset.model_path, "test"), exist_ok=True)


    calc_gradient(dataset, opt, pipe, scene, gaussians, batch_size, bg_color, background)

def calc_gradient(dataset, opt, pipe, scene, gaussians, batch_size, bg_color, background):

    training_dataset = scene.getTrainCameras()
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=12 if dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)
     
    
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


        batch_point_grad = (torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1)) 
        viewspace_grad[:, index] += batch_point_grad
        t_grad[:, index]  += gaussians._t.grad.clone().detach().squeeze(1) 


    final_view_grad = viewspace_grad.sum(dim = 1)
    final_t_grad = t_grad.sum(dim = 1)
    
    if torch.is_tensor(final_view_grad):
        final_view_grad = final_view_grad.detach().cpu().numpy()
    if torch.is_tensor(final_t_grad):
        final_t_grad = final_t_grad.detach().cpu().numpy()
    import pdb; pdb.set_trace()

    os.makedirs(os.path.join(scene.model_path, "gradient"), exist_ok=True)

    np.save(os.path.join(scene.model_path, "gradient/view_grad.npy"), final_view_grad)
    np.save(os.path.join(scene.model_path, "gradient/t_grad.npy"), final_t_grad)



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--grad", type=str, default = None)
    parser.add_argument("--out_path", type=str, default = None)
        
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
        
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)


    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0,op.iterations,3000)]
    
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    main(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
             args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size, args.out_path)

    # All done
    print("\nComputing complete.")
