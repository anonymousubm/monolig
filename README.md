# Monocular 3D Object Detection with LiDAR Guided Semi-Supervised Active Learning

<p align="center"> <img src='img/method.png' align="center" height="450px"> </p>

## Introduction

This is the official implementation of MonoLiG submitted to CVPR 2023.

## Overview

- [Setup](#Setup)
- [Active Learning Experiments](#active-learning-experiments)
- [Acknowledgements](#acknowledgements)

## Setup

### Installation Steps

a. Clone this repository.

```shell
git clone https://github.com/anonymouscvpr23/monolig
```

b. Build docker for monocular detection network DD3D

```shell
cd dd3d/docker
./build.sh
```

c. Build docker for LiDAR detection network openpcdet.

```shell
cd openpcdet/docker
./build.sh
```

### Dataset Preparation
Download KITTI dataset from the [website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

We follow DD3D with the standart splits for KITTI, which can be downloaded from:
```
# download a standard splits subset of KITTI
curl -s https://tri-ml-public.s3.amazonaws.com/github/dd3d/mv3d_kitti_splits.tar | sudo tar xv -C /path/to/data/kitti/KITTI3D
```

The dataset must be organized as follows:
```
PATH_TO_DATA_FOLDER
    └──kitti
        └── KITTI3D
            ├── mv3d_kitti_splits
            │   ├── test.txt
            │   ├── train.txt
            │   ├── trainval.txt
            │   └── val.txt
            ├── testing
            │   ├── calib
            |   │   ├── 000000.txt
            |   │   ├── 000001.txt
            |   │   └── ...
            │   └── image_2
            │       ├── 000000.png
            │       ├── 000001.png
            │       └── ...
            └── training
                ├── calib
                │   ├── 000000.txt
                │   ├── 000001.txt
                │   └── ...
                ├── image_2
                │   ├── 000000.png
                │   ├── 000001.png
                │   └── ...
                └── label_2
                    ├── 000000.txt
                    ├── 000001.txt
                    └── ..
```

Set the corresponding path in constants.py for [PATH_TO_DATA_FOLDER]((https://github.com/anonymouscvpr23/monolig/blob/main/constants.py#L2))

### Setup data and experiment paths

Download depth-pretrained DD3D checkpoints from the original repo for the DLA34 backbone from [here.](https://github.com/TRI-ML/dd3d)

In constants.py change [PATH_TO_EXPERIMENTS_FOLDER](https://github.com/anonymouscvpr23/monolig/blob/main/constants.py#L1), [DD3D_DEPTH_PRETRAINED_PATH](https://github.com/anonymouscvpr23/monolig/blob/main/constants.py#L5)

## Active Learning Experiments

```shell
python al_loop.py \
--config_name random \
--dataset_name kitti \
--trial_number 0 \
--number_of_models 5 \
--start_cycle 0 \
--start_action BUILD
```

Each cycle consists of 4 actions:
- UNCERTAINTY: calculate selection score of each sample
- BUILD: sample the dataset basen on the scores and build in a format suitable for training
- TRAIN_LIDAR: train the LiDAR detector
- PL_UNCERTAINTY: calculate uncertainty of pseudo-labels
- TRAIN: train the monocular detector with combined labeled and pseudo-labeled dataset

## Acknowledgements
- https://github.com/SPengLiang/LPCG
- https://github.com/TRI-ML/dd3d