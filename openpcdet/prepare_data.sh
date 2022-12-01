#! /bin/bash

echo "Preparing data"

PATH_TO_DATASET=$1
DATASET_CONFIG_NAME=$2

echo "PATH_TO_DATASET: $PATH_TO_DATASET"
echo "DATASET_CONFIG_NAME: $DATASET_CONFIG_NAME"

cp -r "$PATH_TO_DATASET/mv3d_kitti_splits" "$PATH_TO_DATASET/ImageSets"

python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
"tools/cfgs/dataset_configs/active_learning/$DATASET_CONFIG_NAME.yaml" \
$PATH_TO_DATASET \