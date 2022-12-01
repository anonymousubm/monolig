import os
import random
import shutil
import operator

from constants import PATH_TO_DATA_FOLDER

def read_txt(path_to_txt):
    with open(path_to_txt) as f:
        lines = f.read().splitlines()
    return lines


def write_txt(lines, path_to_txt):
    with open(path_to_txt, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def get_most_uncertain_samples(uncertainty_file, number_of_new_samples):
    uncertainty_info = dict()
    with open(uncertainty_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            sample_name, uncertainty = line.rstrip().split(";")
            uncertainty = float(uncertainty)
            uncertainty_info[sample_name] = uncertainty
    sorted_uncertainty = sorted(
        uncertainty_info.items(), key=operator.itemgetter(1), reverse=True
    )
    return [x[0] for x in sorted_uncertainty[:number_of_new_samples]]


def create_trial_folder(path_to_trial_folder, dataset_name, initial_size_of_dataset):
    if os.path.isdir(path_to_trial_folder):
        return
    os.makedirs(path_to_trial_folder, exist_ok=True)

    path_to_new_dataset_folder = os.path.join(
        path_to_trial_folder, "partition_0", "KITTI3D"
    )
    os.makedirs(path_to_new_dataset_folder, exist_ok=True)
    path_to_new_splits_folder = os.path.join(
        path_to_new_dataset_folder, "mv3d_kitti_splits"
    )
    os.makedirs(path_to_new_splits_folder, exist_ok=True)
    os.symlink(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "training"),
        os.path.join(path_to_new_dataset_folder, "training"),
    )
    os.symlink(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "testing"),
        os.path.join(path_to_new_dataset_folder, "testing"),
    )

    path_to_full_dataset_folder = os.path.join(
        path_to_trial_folder, "partition_full", "KITTI3D"
    )
    os.makedirs(path_to_full_dataset_folder, exist_ok=True)
    path_to_full_splits_folder = os.path.join(
        path_to_full_dataset_folder, "mv3d_kitti_splits"
    )
    os.makedirs(path_to_full_splits_folder, exist_ok=True)
    os.symlink(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "training"),
        os.path.join(path_to_full_dataset_folder, "training"),
    )
    os.symlink(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "testing"),
        os.path.join(path_to_full_dataset_folder, "testing"),
    )

    original_test_files = read_txt(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "mv3d_kitti_splits", "test.txt")
    )
    original_train_files = read_txt(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "mv3d_kitti_splits", "train.txt")
    )
    original_val_files = read_txt(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "mv3d_kitti_splits", "val.txt")
    )
    original_trainval_files = sorted(original_train_files + original_val_files)
    write_txt(original_test_files, os.path.join(path_to_full_splits_folder, "test.txt"))
    write_txt(
        original_train_files, os.path.join(path_to_full_splits_folder, "train.txt")
    )
    write_txt(original_val_files, os.path.join(path_to_full_splits_folder, "val.txt"))
    write_txt(
        original_trainval_files,
        os.path.join(path_to_full_splits_folder, "trainval.txt"),
    )

    sampled_train_files = sorted(
        random.sample(original_train_files, initial_size_of_dataset)
    )
    sampled_trainval_files = sorted(sampled_train_files + original_val_files)
    unused_train_files = sorted(
        list(set(original_trainval_files) - set(sampled_trainval_files))
    )
    write_txt(original_test_files, os.path.join(path_to_new_splits_folder, "test.txt"))
    write_txt(sampled_train_files, os.path.join(path_to_new_splits_folder, "train.txt"))
    write_txt(original_val_files, os.path.join(path_to_new_splits_folder, "val.txt"))
    write_txt(
        sampled_trainval_files, os.path.join(path_to_new_splits_folder, "trainval.txt")
    )
    write_txt(unused_train_files, os.path.join(path_to_new_splits_folder, "unused.txt"))
    write_txt(
        original_train_files, os.path.join(path_to_new_splits_folder, "all_train.txt")
    )


def build_initial_splits(path_to_new_splits_folder, path_to_trial_folder, config_name):
    path_to_trial_splits_folder = os.path.join(
        path_to_trial_folder, "partition_0", "KITTI3D", "mv3d_kitti_splits"
    )
    shutil.copytree(path_to_trial_splits_folder, path_to_new_splits_folder)

    path_to_trial_pvrcnn = os.path.join(
        path_to_trial_folder, "random", "partition_0", "COMBINED"
    )
    if os.path.exists(path_to_trial_pvrcnn):
        path_to_dataset_folder = os.path.dirname(
            os.path.dirname(path_to_new_splits_folder)
        )
        os.symlink(
            path_to_trial_pvrcnn, os.path.join(path_to_dataset_folder, "COMBINED")
        )


def build_new_splits(
    path_to_new_splits_folder,
    path_to_trial_folder,
    config_name,
    current_cycle,
    number_of_new_samples,
):
    os.makedirs(path_to_new_splits_folder, exist_ok=True)

    path_to_previous_dataset_folder = os.path.join(
        path_to_trial_folder,
        config_name,
        "partition_{}".format(current_cycle - 1),
        "KITTI3D",
    )
    all_train_files = read_txt(
        os.path.join(
            path_to_previous_dataset_folder, "mv3d_kitti_splits", "all_train.txt"
        )
    )
    previous_test_files = read_txt(
        os.path.join(path_to_previous_dataset_folder, "mv3d_kitti_splits", "test.txt")
    )
    previous_val_files = read_txt(
        os.path.join(path_to_previous_dataset_folder, "mv3d_kitti_splits", "val.txt")
    )
    previous_train_files = read_txt(
        os.path.join(path_to_previous_dataset_folder, "mv3d_kitti_splits", "train.txt")
    )

    path_to_uncertainty_tfru = os.path.join(
        path_to_trial_folder,
        config_name,
        "partition_{}".format(current_cycle - 1),
        "uncertainty.tfru",
    )
    most_uncertain_samples = get_most_uncertain_samples(
        path_to_uncertainty_tfru, number_of_new_samples
    )
    new_train_files = sorted(most_uncertain_samples + previous_train_files)
    new_unused_files = sorted(list(set(all_train_files) - set(new_train_files)))
    write_txt(all_train_files, os.path.join(path_to_new_splits_folder, "all_train.txt"))
    write_txt(new_unused_files, os.path.join(path_to_new_splits_folder, "unused.txt"))
    write_txt(previous_test_files, os.path.join(path_to_new_splits_folder, "test.txt"))
    write_txt(new_train_files, os.path.join(path_to_new_splits_folder, "train.txt"))
    write_txt(previous_val_files, os.path.join(path_to_new_splits_folder, "val.txt"))
    write_txt(
        sorted(new_train_files + previous_val_files),
        os.path.join(path_to_new_splits_folder, "trainval.txt"),
    )


def build_dataset_iteration(
    path_to_trial_folder, config_name, dataset_name, current_cycle, number_of_new_samples
):
    path_to_new_dataset_folder = os.path.join(
        path_to_trial_folder,
        config_name,
        "partition_{}".format(current_cycle),
        "KITTI3D",
    )
    if os.path.exists(path_to_new_dataset_folder):
        return
    os.makedirs(path_to_new_dataset_folder)
    
    os.symlink(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "training"),
        os.path.join(path_to_new_dataset_folder, "training"),
    )
    os.symlink(
        os.path.join(PATH_TO_DATA_FOLDER, dataset_name, "KITTI3D", "testing"),
        os.path.join(path_to_new_dataset_folder, "testing"),
    )

    path_to_new_splits_folder = os.path.join(
        path_to_new_dataset_folder, "mv3d_kitti_splits"
    )

    if current_cycle == 0:
        build_initial_splits(
            path_to_new_splits_folder, path_to_trial_folder, config_name
        )
    else:
        build_new_splits(
            path_to_new_splits_folder,
            path_to_trial_folder,
            config_name,
            current_cycle,
            number_of_new_samples,
        )

def link_initial_dataset(path_to_trial_folder, config_name, initial_config_name):
    path_to_initial_dataset_folder = os.path.join(
        path_to_trial_folder,
        initial_config_name,
        "partition_0",
    )
    assert os.path.exists(path_to_initial_dataset_folder), "Initial dataset folder does not exists"
    path_to_new_dataset_folder = os.path.join(
        path_to_trial_folder,
        config_name,
        "partition_0",
    )
    if os.path.exists(path_to_new_dataset_folder):
        return
    os.makedirs(path_to_new_dataset_folder)
    
    os.symlink(
        os.path.join(path_to_initial_dataset_folder, "KITTI3D"),
        os.path.join(path_to_new_dataset_folder, "KITTI3D"),
    )
    os.symlink(
        os.path.join(path_to_initial_dataset_folder, "COMBINED"),
        os.path.join(path_to_new_dataset_folder, "COMBINED"),
    )