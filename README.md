# Optimized Minimal 4D Gaussian Splatting

Minseo Lee*, Byeonghyeon Lee*, Lucas Yunkyu Lee, Eunsoo Lee, Sangmin Kim, Seunghyeon Song, Joo Chan Lee, Jong Hwan Ko, Jaesik Park, and Eunbyung Park†

[Project Page](https://minshirley.github.io/OMG4/) &nbsp; [Paper](https://arxiv.org/abs/2510.03857)

![Teaser](https://github.com/MinShirley/OMG4/blob/main/assets/teaser.jpg?raw=true)

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
conda create -n OMG4 python=3.11
conda activate OMG4
pip install -r requirement.txt
```

Then, please download the pretrained 4D-GS weight and gradients.  
You can download the weights from [Google Drive](https://drive.google.com/drive/folders/1WB7WYOUlvemfYZE35lkl_WV4fiF3p68v?usp=sharing).


### 2. Data preparation
Data preprocessing follows the method used in [4D-GS](https://github.com/fudan-zvg/4d-gaussian-splatting).
Run the following command to prepare the data:
```
python scripts/n3v2blender.py data/N3V/$scene_name
```

The directory data/N3V/$scene_name should contain the following files before preprocessing:
```
data/N3V/$scene_name
├── cam00.mp4
├── cam01.mp4
├── ...
└── poses_bounds.npy
```

After running the script, the directory structure will look like this:
```
data/N3V/$scene_name
├── cam00.mp4
├── cam01.mp4
├── ...
├── poses_bounds.npy
├── transforms_train.json
├── transforms_test.json
└── images
    ├── cam00_0000.png
    ├── cam00_0001.png
    ├── ...
```

### 3. Training
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

### 4. Evaluation
At the end of training, the evaluation process is implemented. Or you can evaluate the trained model with the encoded "comp.xz" file with the following command
```
python test.py \
--config ./configs/dynerf/cook_spinach.yaml \
--comp_checkpoint ./cook_spinach_comp/comp.xz
```

The weights reported in our paper are available for download on [Google Drive](https://drive.google.com/drive/folders/1WB7WYOUlvemfYZE35lkl_WV4fiF3p68v?usp=sharing).

To evaluate OMG4-FTGS using a trained model, you can use the provided checkpoints.
The checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1WB7WYOUlvemfYZE35lkl_WV4fiF3p68v?usp=sharing).
```
python -m OMG4_FTGS.test \
    --comp_checkpoint ./OMG4-FTGS_weights/cook_spinach.xz \
    --data_path data/N3V/cook_spinach
```
```
