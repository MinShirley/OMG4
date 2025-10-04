# Optimized Minimal 4D Gaussian Splatting

Minseo Lee*, Byeonghyeon Lee*, Lucas Yunkyu Lee, Eunsoo Lee, Sangmin Kim, Seunghyeon Song, Joo Chan Lee, Jong Hwan Ko, Jaesik Park, and Eunbyung Park†

[Project Page](https://minshirley.github.io/OMG4/) &nbsp; [Paper] 


Our code is built based on [4D-GS](https://github.com/fudan-zvg/4d-gaussian-splatting)


## Setup
We ran the experiments in the following environment:
```
- ubuntu: 20.04
- python: 3.11
- cuda: 12.1
- pytorch: 2.5.1  ( > 2.5.0 is required for svq)
- GPU: RTX 3090
```

###  1. Installation
```
conda create -n 4dgs_comp python=3.11
conda activate 4dgs_comp
pip install -r requirement.txt
```

Then, please download the pretrained 4D-GS weight and gradients.

### 2. Training

Gradient (2D mean, t) should be calculated in advance to sample important Gaussians.
If --grad is not designated, it will automatically compute gradients.
Once you compute gradients (or download provided gradients), please set --grad to your gradient path, not to compute them repeatedly.
```
python train.py \
  --config ./configs/dynerf/cook_spinach.yaml \
  --start_checkpoint PATH_TO_4DGS_PRETRAINED \
  --grad PATH_TO_GRADIENT \
  --out_path ./cook_spinach_comp
```
You can check the result (w/ various metrics, encoded model size, etc.) at **./res.txt**

### 3. Evaluation
At the end of training, the evaluation process is implemented. Or you can evaluate the trained model with the encoded "comp.xz" file with the following command
```
python test.py \
--config ./configs/dynerf/cook_spinach.yaml \
--comp_checkpoint ./cook_spinach_comp/comp.xz
```


