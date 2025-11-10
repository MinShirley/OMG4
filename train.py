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
import sys
import uuid
import torch
import numpy as np
import torchvision.transforms as T
import imageio
import lpips

from torch import nn
from utils.loss_utils import l1_loss, ssim, msssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from PIL import Image
from compute_gradient import calc_gradient
from utils.compress_utils import save_comp


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size):
    
    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)
    gaussians.training_setup(opt)
    
    os.makedirs(scene.model_path, exist_ok=True)
    loss_log_path = os.path.join(scene.model_path, "loss_log.txt")
    loss_log_file = open(loss_log_path, "a")
    
    tau_sim = opt.tau_sim
    sim_cutoff = opt.sim_cutoff
    grid_size = opt.grid_size

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    first_iter = 30000
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    os.makedirs(os.path.join(dataset.model_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset.model_path, "test"), exist_ok=True)

    num_merge = opt.num_merge

    compression_start = 30000   # After 4DGS training
    grad_pruning_iter = compression_start + opt.grad_pruning_iter
    first_merge_iter = grad_pruning_iter + opt.grad_pruning_opt_iter
    svq3d_iter = first_merge_iter + opt.merge_opt_iter * num_merge + opt.net_opt_iter
    svq4d_iter = svq3d_iter + opt.svq3d_opt_iter
    encode_iter = svq4d_iter + opt.svq4d_opt_iter
    final_iteration = encode_iter
    testing_iterations.append(final_iteration)

    try:
        view_grad = np.load(os.path.join(args.grad, 'view_grad.npy'))
        t_grad = np.load(os.path.join(args.grad, 't_grad.npy'))
    except:
        print("Error: please calculate gradient first")

    #gradient sampling
    mask = gaussians.gradient_sampling(opt.tau_GS, view_grad, t_grad, args)
    torch.cuda.empty_cache() 

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0
    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') and key!='lambda_dssim']
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
        
    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3,pipe.env_map_res, pipe.env_map_res),dtype=torch.float, device="cuda").requires_grad_(True))
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None
        
    gaussians.env_map = env_map
        
    training_dataset = scene.getTrainCameras()
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=12 if dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)
     
    iteration = first_iter
    lpips_model = lpips.LPIPS(net='alex')   # vgg for Bartender
    lpips_model.eval()                   
    lpips_model.requires_grad_(False)  
    lpips_model = lpips_model.to("cuda")
    actual_storage = 0.0
    
    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            iteration += 1
            if iteration > opt.iterations:
                break

            iter_start.record()
            gaussians.update_learning_rate(iteration)
            
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()
                
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            batch_point_grad = []
            batch_visibility_filter = []
            batch_radii = []
            
            for batch_idx in range(batch_size):
                gt_image, viewpoint_cam = batch_data[batch_idx]
                gt_image = gt_image.cuda()
                viewpoint_cam = viewpoint_cam.cuda()

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
                if iteration<=first_merge_iter:
                    batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)
                
            if iteration %100 ==0 :
                img = image.detach().cpu().clamp(0, 1)
                img = T.ToPILImage()(img)

                save_dir = os.path.join(scene.model_path, "renders")
                os.makedirs(save_dir, exist_ok=True)

            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii,1).max(1)[0]
                if iteration<=first_merge_iter:
                    batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                    batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)
                    
                    if gaussians.gaussian_dim == 4:
                        batch_t_grad = gaussians._t.grad.clone()[:,0].detach()
                        batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                        batch_t_grad = batch_t_grad.unsqueeze(1)
            else:
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone().detach()
        
            iter_end.record()
            loss_dict = {"Ll1": Ll1,
                        "Lssim": Lssim}

            with torch.no_grad():
                psnr_for_log = psnr(image, gt_image).mean().double()
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                ema_ssimloss_for_log = 0.4 * Lssim.item() + 0.6 * ema_ssimloss_for_log
                
                for lambda_name in lambda_all:
                    if opt.__dict__[lambda_name] > 0:
                        ema = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                        vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.4 * vars()[f"L{lambda_name.replace('lambda_', '')}"].item() + 0.6*ema
                        loss_dict[lambda_name.replace("lambda_", "L")] = vars()[lambda_name.replace("lambda_", "L")]
                        
                if iteration % 10 == 0:
                    postfix = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "PSNR": f"{psnr_for_log:.{2}f}",
                                            "Ll1": f"{ema_l1loss_for_log:.{4}f}",
                                            "N": f"{gaussians._xyz.shape[0]:.1f}",
                                            "Lssim": f"{ema_ssimloss_for_log:.{4}f}"}
                    
                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema_loss = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            postfix[lambda_name.replace("lambda_", "L")] = f"{ema_loss:.{4}f}"
                            
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                #gradient pruning
                if iteration == grad_pruning_iter:
                    gaussians.gradient_pruning(view_grad, t_grad, opt.tau_GP, opt.tau_GP, args, mask)
                    torch.cuda.empty_cache() 

                #gaussian merging
                if iteration == first_merge_iter:
                    gaussians.calc_clusters(grid_size=grid_size, tau_sim=tau_sim, sim_cutoff=sim_cutoff, t_grid_size=opt.t_grid_size)
                    if opt.grid_exp_ratio:
                        grid_size *= opt.grid_exp_ratio
                    gaussians.set_alpha_groups()

                if num_merge and iteration % 1000 == 0 and iteration > first_merge_iter:
                    print(iteration,"Pruning merged Gaussians using learned alpha...")
                    num_merge -= 1
                    gaussians.training_alpha = False
                    N = gaussians.get_xyz.shape[0]
                    gaussians.alpha_pruning_groups()
                    loss_log_file.write(f"Merge done. {N} ->  {gaussians._xyz.shape[0]}\n")
                    print(f"Merge done. {N} ->  {gaussians._xyz.shape[0]}\n")
                    if num_merge:   # should prepare for next merge
                        gaussians.calc_clusters(grid_size=grid_size, tau_sim=tau_sim, sim_cutoff=sim_cutoff, t_grid_size=opt.t_grid_size)
                        gaussians.set_alpha_groups()
                    
                    else:   # merging is done, construct net
                        print("start training network")
                        loss_log_file.write(f"Start training network.\n")
                        gaussians.construct_net()

                #3d svq
                if iteration == svq3d_iter:
                    loss_log_file.write(f"3D svq start\n.")
                    gaussians.apply_svq_3d(args)
                #4d svq
                if iteration == svq4d_iter:
                    loss_log_file.write(f"4D svq start\n.")
                    gaussians.apply_svq_4d(args)

                if iteration == encode_iter:
                    print("comp")
                    save_dict = gaussians.encode()     
                    save_comp(scene.model_path + "/comp.xz", save_dict)
                    
                    actual_storage = os.path.getsize(scene.model_path + "/comp.xz") / 1024 / 1024   # header is included (Not 100% actual storage).
                    gaussians.decode(save_dict, decompress=True)


                # Log and save
                test_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_dict, lpips_model, loss_log_file, actual_storage)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter and (opt.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < opt.densify_until_num_points):
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if batch_size == 1:
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                    else:
                        gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                        
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.thresh_opa_prune, scene.cameras_extent, size_threshold, opt.densify_grad_t_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                        
                if iteration % 100 == 0 and loss_log_file:
                    log_line = f"[ITER {iteration}] Loss: {ema_loss_for_log:.6f} | PSNR: {psnr_for_log:.2f} | Ll1: {ema_l1loss_for_log:.4f} | xyz: {gaussians._xyz.shape[0]:.4f} | Lssim: {ema_ssimloss_for_log:.4f}"
                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema_val = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            log_line += f" | L{lambda_name.replace('lambda_', '')}: {ema_val:.4f}"
                    loss_log_file.write(log_line + "\n")

                if iteration in testing_iterations and loss_log_file:
                    loss_log_file.write(f"[ITER {iteration}] test_psnr: {test_psnr:.4f}\n")


                if loss_log_file:
                    loss_log_file.flush()

                # Optimizer step
                if iteration < final_iteration:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if pipe.env_map_res and iteration < pipe.env_optimize_until:
                        env_map_optimizer.step()
                        env_map_optimizer.zero_grad(set_to_none = True)
                    if gaussians.net_enabled:
                        gaussians.optimizer_net.step()
                        gaussians.optimizer_net.zero_grad(set_to_none = True)
                        gaussians.scheduler_net.step()
                    if gaussians.vq_enabled:
                        if hasattr(gaussians, "optimizer_code") and gaussians.optimizer_code is not None:
                            gaussians.optimizer_code.step()
                            gaussians.optimizer_code.zero_grad()
                        
                        if hasattr(gaussians, "optimizer_code_4d") and gaussians.optimizer_code_4d is not None:
                            gaussians.optimizer_code_4d.step()
                            gaussians.optimizer_code_4d.zero_grad()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_dict=None, lpips_model=None, log_file=None, actual_storage=0.0):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        if loss_dict is not None:
            if "Lrigid" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/rigid_loss', loss_dict['Lrigid'].item(), iteration)
            if "Ldepth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/depth_loss', loss_dict['Ldepth'].item(), iteration)
            if "Ltv" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/tv_loss', loss_dict['Ltv'].item(), iteration)
            if "Lopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/opa_loss', loss_dict['Lopa'].item(), iteration)
            if "Lptsopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/pts_opa_loss', loss_dict['Lptsopa'].item(), iteration)
            if "Lsmooth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/smooth_loss', loss_dict['Lsmooth'].item(), iteration)
            if "Llaplacian" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/laplacian_loss', loss_dict['Llaplacian'].item(), iteration)

    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        validation_configs = ({'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
                              {'name': 'test', 'cameras' : [scene.getTestCameras()[idx] for idx in range(len(scene.getTestCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                msssim_test = 0.0
                lpips_test = 0.0
                for idx, batch_data in enumerate(tqdm(config['cameras'])):
                    gt_image, viewpoint = batch_data
                    gt_image = gt_image.cuda()
                    viewpoint = viewpoint.cuda()
                    
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    
                    depth = easy_cmap(render_pkg['depth'][0])
                    alpha = torch.clamp(render_pkg['alpha'], 0.0, 1.0).repeat(3,1,1)
                    if tb_writer and (idx < 5):
                        grid = [gt_image, image, alpha, depth]
                        grid = make_grid(grid, nrow=2)
                        tb_writer.add_images(config['name'] + "_view_{}/gt_vs_render".format(viewpoint.image_name), grid[None], global_step=iteration)
                            
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    lpips_test += lpips_model(image[None], gt_image[None]).squeeze().item()

                    if idx < 5:
                        try:
                            imageio.imwrite(os.path.join(scene.model_path, config['name'], "render_{:05d}_{}.png".format(iteration, viewpoint.image_name)), (image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
                        except:
                            pass
                    

                    test_log_path = os.path.join(scene.model_path, "test.txt")
                    with open(test_log_path, "a") as test_log_file:
                        test_log_file.write(f"[Time {viewpoint.timestamp}] test_psnr: {psnr(image, gt_image).mean().double():.4f}\n")

                    ssim_test += ssim(image, gt_image).mean().double()
                    msssim_test += msssim(image[None].cpu(), gt_image[None].cpu())
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                ssim_test /= len(config['cameras'])     
                msssim_test /= len(config['cameras'])        
                lpips_test /= len(config['cameras'])        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} lpips {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - msssim', msssim_test, iteration)
                if config['name'] == 'test':
                    psnr_test_iter = psnr_test.item()
                    log_file.write(f"psnr: {psnr_test_iter}  ssim: {ssim_test}   lpips: {lpips_test}\n")
                    log_file.flush()
                    if iteration == testing_iterations[-1]:
                        name = "_".join(scene.model_path.split("/")[-2:])
                        with open("./res.txt", "a") as f:
                            num_pts = scene.gaussians.get_xyz.shape[0]
                            f.write("{}: PSNR {:.3f}, SSIM {:.5f}, MS-SSIM {:.5f}, LPIPS {:.5f}, num_pts {}, MB {:.2f}\n".format(name, psnr_test, ssim_test, msssim_test, lpips_test, num_pts, actual_storage))
    torch.cuda.empty_cache()
    return psnr_test_iter

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
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
             args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size)

    # All done
    print("\nTraining complete.")
