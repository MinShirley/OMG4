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
import numpy as np
import os
import torchvision.transforms as T
import torch
import lzma
import pickle


def test_comp(dataset, opt, pipe, gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, comp_checkpoint):
    

    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]
    
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)
    gaussians.training_setup(opt)
    
    os.makedirs(scene.model_path, exist_ok=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")



    xz_path = comp_checkpoint
    print(xz_path)

    with lzma.open(xz_path, "rb") as f:
        load_dict = pickle.load(f)
    gaussians.decode(load_dict, decompress=True)
    gaussians.active_sh_degree = 3
    gaussians.active_sh_degree_t = 2
    if gaussians.env_map.device != "cuda":
        gaussians.env_map =  gaussians.env_map.to("cuda")


    psnr_sum = 0.0
    test_dataset = scene.getTestCameras()

    import time
    secs = 0.0
    pipe.env_map_res = 0
    for idx in range(len(test_dataset)):
        gt_image, viewpoint_cam = test_dataset[idx]
        gt_image = gt_image.cuda()
        viewpoint = viewpoint_cam.cuda()

        screenspace_points = torch.zeros_like(
            scene.gaussians.get_xyz,
            dtype=scene.gaussians.get_xyz.dtype,
           requires_grad=False,
            device="cuda"
        )

        torch.cuda.synchronize()
        with torch.no_grad():
            st = time.time()
            render_pkg  = render(viewpoint, scene.gaussians, pipe =pipe, bg_color = background)
            ed = time.time()
        secs += (ed - st)

        torch.cuda.synchronize()

        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        test_psnr = psnr(image, gt_image).mean().item() 
        psnr_sum += test_psnr



    mean_psnr = psnr_sum / len(test_dataset)
    print(secs,  len(test_dataset), (( len(test_dataset)) / secs))
    print(f"[INFO] Mean PSNR: {mean_psnr:.2f} dB")
    print(f"[INFO] Avg Render Time: {secs/len(test_dataset):.4f} sec/frame")






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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")

    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--comp_checkpoint", type=str, default = None)
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
    test_comp(lp.extract(args), op.extract(args), pp.extract(args), args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.comp_checkpoint)

    # All done
    print("\nTraining complete.")
