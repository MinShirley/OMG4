
import torch
from torch import nn
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import numpy as np
import torch
from math import isqrt
from utils.compress_utils import *

import numpy as np

class DynamicGaussians(nn.Module):
    def __init__(self):
        super().__init__()

        param_names = [
            "means", "scales", "quats", "opacities",
            "sh_0", "sh_n", "times", "durations", "velocities"
        ]

        for name in param_names:
            setattr(self, name, nn.Parameter(torch.empty((0, 3)).float()))

        self.sh_degree = 3  
        self.time_duration = [0.0, 10.0]

    def set(self):
        with torch.no_grad():
            self.means.zero_()
            self.scales.zero_()
            self.quats.zero_()
            self.opacities.zero_()
            self.sh_0.zero_()
            self.sh_n.zero_()
            self.times.zero_()
            self.durations.zero_()
            self.velocities.zero_()
        
    def construct_net(self, train=True):
        import tinycudann as tcnn
        # Default hyperparameter from OMG (https://github.com/maincold2/OMG)
        print("MLP get 4 -> 3. now xyzt -> xyz+v")
        self.mlp_cont = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
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


        self._features_static = nn.Parameter(self.sh_0[:, 0].clone().detach(),requires_grad=True) #(N,)
        self._features_view = nn.Parameter(torch.zeros((self.means.shape[0], 3), device="cuda").requires_grad_(True)) #(N,3)
    
        mlp_params = []
        for params in self.mlp_cont.parameters():
            mlp_params.append(params)
        for params in self.mlp_view.parameters():
            mlp_params.append(params)
        for params in self.mlp_dc.parameters():
            mlp_params.append(params)
        for params in self.mlp_opacity.parameters():
            mlp_params.append(params)
                
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
        x = x * 2 - 1  # aabb is at [-1, 1]
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
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x           

    def decode(self, save_dict, decompress=True, nettrain=True):
        
        self.means = torch.nn.Parameter(torch.from_numpy(save_dict['means']).cuda().float().requires_grad_(True))
        self.times = torch.nn.Parameter(torch.from_numpy(save_dict['times']).float().cuda().requires_grad_(True))
        scale = []
        rotation = []
        appearance = []
        durations = []
        velocities = []

        if decompress:
            for i in range(len(save_dict['scale_code'])):
                labels = huffman_decode(save_dict['scale_index'][i], save_dict['scale_htable'][i])
                cluster_centers = save_dict['scale_code'][i]
                scale.append(torch.tensor(cluster_centers[labels]).cuda())
            self.scales = torch.nn.Parameter(torch.cat(scale, dim=-1).float().requires_grad_(True))
            
            for i in range(len(save_dict['rotation_code'])):
                labels = huffman_decode(save_dict['rotation_index'][i], save_dict['rotation_htable'][i])
                cluster_centers = save_dict['rotation_code'][i]
                rotation.append(torch.tensor(cluster_centers[labels]).cuda())
            self.quats = torch.nn.Parameter(torch.cat(rotation, dim=-1).float().requires_grad_(True))

            for i in range(len(save_dict['durations_code'])):
                labels = huffman_decode(save_dict['durations_index'][i], save_dict['durations_htable'][i])
                cluster_centers = save_dict['durations_code'][i]
                durations.append(torch.tensor(cluster_centers[labels]).cuda())
            self.durations = torch.nn.Parameter(torch.cat(durations, dim=-1).float().requires_grad_(True))

            for i in range(len(save_dict['velocities_code'])):
                labels = huffman_decode(save_dict['velocities_index'][i], save_dict['velocities_htable'][i])
                cluster_centers = save_dict['velocities_code'][i]
                velocities.append(torch.tensor(cluster_centers[labels]).cuda())
            self.velocities = torch.nn.Parameter(torch.cat(velocities, dim=-1).float().requires_grad_(True))
            
            for i in range(len(save_dict['app_code'])):
                labels = huffman_decode(save_dict['app_index'][i], save_dict['app_htable'][i])
                cluster_centers = save_dict['app_code'][i]
                appearance.append(torch.tensor(cluster_centers[labels]).cuda())
            app_feature = torch.cat(appearance, dim=-1).float()



            if not hasattr(self, "mlp_cont"):
                self.construct_net(train=nettrain)
            self.mlp_cont.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_cont']).cuda().half().requires_grad_(True))
            self.mlp_dc.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_dc']).cuda().half().requires_grad_(True))
            self.mlp_view.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_sh']).cuda().half().requires_grad_(True))
            self.mlp_opacity.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_opacity']).cuda().half().requires_grad_(True))

            


            N = app_feature.shape[0]
            self._features_static = nn.Parameter(app_feature[:, 0:3].clone().detach().cuda().requires_grad_(True))  # [N, 3]
            self._features_view = nn.Parameter(app_feature[:, 3:6].clone().detach().cuda().requires_grad_(True))  
 
