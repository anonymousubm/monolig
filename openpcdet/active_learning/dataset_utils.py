import os
import glob
import shutil
from tqdm import tqdm
from dd3d.active_learning.utils.training_utils import get_latest_experiment_folder
from dd3d.active_learning.utils.dataset_utils import (
    read_txt,
    write_txt,
)
from dd3d.active_learning.utils.object_utils import (
    read_labels_as_objects,
    write_objects_as_labels,
    threshold_objects,
    threshold_valid_classes,
)
from constants import PATH_TO_DATA_FOLDER, PATH_TO_EXPERIMENTS_FOLDER

def create_dataset(
    dataset_name, path_to_partition_folder, path_to_lidar_predictions, path_to_lidar_weights
):
    path_to_combined = os.path.join(path_to_partition_folder, "COMBINED", "KITTI3D")
    if os.path.exists(path_to_combined):
        return
    path_to_combined_splits = os.path.join(path_to_combined, "mv3d_kitti_splits")
    path_to_combined_training = os.path.join(path_to_combined, "training")
    path_to_combined_labels = os.path.join(path_to_combined_training, "label_2")
    path_to_combined_weights = os.path.join(path_to_combined_training, "weights")
    path_to_kitti = os.path.join(path_to_partition_folder, "KITTI3D")
    path_to_kitti_labels = os.path.join(path_to_kitti, "training", "label_2")

    os.makedirs(path_to_combined, exist_ok=True)
    os.makedirs(path_to_combined_splits, exist_ok=True)
    os.makedirs(path_to_combined_training, exist_ok=True)
    os.makedirs(path_to_combined_labels, exist_ok=True)
    os.makedirs(path_to_combined_weights, exist_ok=True)
    os.symlink(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "testing"),
        os.path.join(path_to_combined, "testing"),
    )
    os.symlink(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "training", "calib"),
        os.path.join(path_to_combined_training, "calib"),
    )
    os.symlink(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "training", "image_2"),
        os.path.join(path_to_combined_training, "image_2"),
    )
    os.symlink(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "training", "velodyne"),
        os.path.join(path_to_combined_training, "velodyne"),
    )
    shutil.copy(
        os.path.join(path_to_kitti, "mv3d_kitti_splits", "test.txt"),
        os.path.join(path_to_combined_splits, "test.txt"),
    )

    labeled_flags = {}
    ## Val files are the same. Also copy label files from ground-truth.
    path_to_kitti_val_files = os.path.join(
        path_to_kitti, "mv3d_kitti_splits", "val.txt"
    )
    kitti_val_files = read_txt(path_to_kitti_val_files)
    labeled_flags.update({k: 1 for k in kitti_val_files})
    shutil.copy(
        path_to_kitti_val_files, os.path.join(path_to_combined_splits, "val.txt")
    )
    for val_file in tqdm(kitti_val_files):
        shutil.copy(
            os.path.join(path_to_kitti_labels, "{}.txt".format(val_file)),
            os.path.join(path_to_combined_labels),
        )

    ## Copy previously used training files
    path_to_kitti_train_files = os.path.join(
        path_to_kitti, "mv3d_kitti_splits", "train.txt"
    )
    kitti_train_files = read_txt(path_to_kitti_train_files)
    labeled_flags.update({k: 1 for k in kitti_train_files})
    for train_file in tqdm(kitti_train_files):
        shutil.copy(
            os.path.join(path_to_kitti_labels, "{}.txt".format(train_file)),
            os.path.join(path_to_combined_labels),
        )

    ## Unused files are going to be train files. Copy files from lidar predicted path.
    path_to_kitti_unused_files = os.path.join(
        path_to_kitti, "mv3d_kitti_splits", "unused.txt"
    )
    kitti_unused_files = read_txt(path_to_kitti_unused_files)
    labeled_flags.update({k: 0 for k in kitti_unused_files})
    for val_file in tqdm(kitti_unused_files):
        objects = read_labels_as_objects(
            os.path.join(path_to_lidar_predictions, "{}.txt".format(val_file))
        )
        
        objects = threshold_objects(objects, 0.5)
        objects = threshold_valid_classes(objects)
        write_objects_as_labels(
            objects, os.path.join(path_to_combined_labels, "{}.txt".format(val_file))
        )

    ## Create weights
    VALID_CLASS_NAMES = ("Car", "Pedestrian", "Cyclist", "Van", "Truck")
    for sample_id in tqdm(
        sorted(kitti_train_files + kitti_unused_files + kitti_val_files)
    ):
        labeled_flag = int(labeled_flags[sample_id])
        path_to_weight_txt = os.path.join(
            path_to_combined_weights, "{}.txt".format(sample_id)
        )
        path_to_label_txt = os.path.join(
            path_to_combined_labels, "{}.txt".format(sample_id)
        )
        labels = read_labels_as_objects(path_to_label_txt)
        valid_labels = [label for label in labels if label.type in VALID_CLASS_NAMES]
        if labeled_flag == 1:
            weights = ["{} 1.0".format(labeled_flag) for _ in valid_labels]
            write_txt(weights, path_to_weight_txt)
        else:
            path_to_lidar_uncertainty_txt = os.path.join(
                path_to_lidar_weights, "{}.txt".format(sample_id)
            )
            shutil.copy(path_to_lidar_uncertainty_txt, path_to_weight_txt)
        assert len(read_txt(path_to_weight_txt)) == len(
            valid_labels
        ), "Number of lines in weights and labels not matching"

    ## Create splits
    write_txt(
        sorted(kitti_train_files + kitti_unused_files),
        os.path.join(path_to_combined_splits, "train.txt"),
    )
    write_txt(
        sorted(kitti_train_files + kitti_unused_files + kitti_val_files),
        os.path.join(path_to_combined_splits, "trainval.txt"),
    )


def build_combined_dataset(trial_number, config_name, dataset_name, current_cycle):
    path_to_partition_folder = "{}/active_learning/{}/trial_{}/{}/partition_{}".format(
        PATH_TO_DATA_FOLDER, dataset_name, trial_number, config_name, current_cycle
    )
    path_to_lidar_weights = os.path.join(
        PATH_TO_EXPERIMENTS_FOLDER,
        "{}_t{}_{}_{}_unc".format(dataset_name, trial_number, config_name, current_cycle),
        "weights",
    )
    path_to_lidar_predictions = os.path.join(
        PATH_TO_EXPERIMENTS_FOLDER,
        "{}_t{}_{}_{}_unc".format(dataset_name, trial_number, config_name, current_cycle),
        "label_2",
    )
    create_dataset(
        dataset_name, path_to_partition_folder, path_to_lidar_predictions, path_to_lidar_weights
    )