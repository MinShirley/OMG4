import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import numpy as np
from math import isqrt
from utils.compress_utils import *
import os


import numpy as np

def load_caminfo(args):
    pose_path = os.path.join(args.data_path, "poses_bounds.npy")
    print(pose_path)
    arr: np.ndarray = np.load(pose_path)
    cam = arr[:, :-2].reshape((-1, 3, 5))

    H,W,fl = cam[0, :, -1]
    intrinsic = np.array([[fl/2, 0, W/(2*args.resolution)], [0, fl/2, H/(2*args.resolution)], [0,0,1]])

    c2ws = cam[..., :4]
    c2ws = np.stack([c2ws[..., 1], c2ws[..., 0], -c2ws[..., 2], c2ws[..., 3]], axis=1)

    bottom = np.array([0, 0, 0, 1]).reshape((1, 4, 1)).repeat(len(c2ws), 0)
    c2ws = np.concatenate([c2ws, bottom], axis=2)
    extrinsics = np.transpose(np.linalg.inv(c2ws), (0,2,1))


    return extrinsics, intrinsic

