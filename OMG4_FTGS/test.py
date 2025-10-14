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

import random
import torch
import sys
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import numpy as np
import torch
import lzma
import pickle
from utils.compress_utils import *
from OMG4_FTGS.render import render
from OMG4_FTGS.utils import *
from OMG4_FTGS.dataloader import *
from OMG4_FTGS._gaussians import *

'''
python test.py --comp_checkpoint /hdd_1/lms20031/final/Bartender_0.6/comp.xz --config configs/dynerf/Bartender_0.6.yaml


python -m OMG4_FTGS.test \
    --comp_checkpoint /hdd/blee/4d/FreeTimeGS-main/fpsweights/ours_L_weight/cook_spinach.xz \
    --data_path /hdd_1/lms20031/OMG4_DATASET_REPRO/cook_spinach

'''

def test_comp(path, comp_checkpoint):
    xz_path = comp_checkpoint

    print("load checkpoint")
    with lzma.open(xz_path, "rb") as f:
        load_dict = pickle.load(f)
    gs = DynamicGaussians() 
    gs.decode(load_dict, decompress=True, nettrain=False)
    print("load")


    psnr_sum = 0.0
    test_dataset = load_test_dataset(path)
    w2c, K_s = load_caminfo(args)

    for idx in range(len(test_dataset)):
        viewpoint = test_dataset[idx]
        gt_image = viewpoint.image.to("cuda")


        with torch.no_grad():
            img, _, _  = render(viewpoint, gs, w2c, K_s)

        image = (img[0]).permute(2,0,1)
        test_psnr = psnr(image, gt_image).mean().item()
        psnr_sum += test_psnr


    mean_psnr = (psnr_sum / len(test_dataset))
    print(f"[INFO] Mean PSNR: {mean_psnr:.2f} dB")




def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="script parameters")
    parser.add_argument("--comp_checkpoint", type=str, default = None)
    parser.add_argument("--data_path", type=str, default = None)
    parser.add_argument("--resolution", type=int, default = 2)

    args = parser.parse_args(sys.argv[1:])

    
    print("Testing " + args.data_path)
    test_comp(args.data_path, args.comp_checkpoint)

    # All done
    print("\nTesting complete.")
