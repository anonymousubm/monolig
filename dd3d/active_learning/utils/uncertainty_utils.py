import os
from tqdm import tqdm
from active_learning.utils.dataset_utils import read_txt
from active_learning.utils.class_utils import initialize_class

def write_uncertainty_dict(uncertainty_dict, out_file):
    with open(out_file, "w") as f:
        for file_name, unc in uncertainty_dict.items():
            uncertainty_line = "{};{}".format(file_name, str(unc))
            f.write(uncertainty_line)
            f.write("\n")

def compute_uncertainty(trial_number, path_to_trial_folder, config, current_cycle):
    path_to_previous_unused = os.path.join(
        path_to_trial_folder,
        config["name"],
        "partition_{}".format(current_cycle - 1),
        "KITTI3D",
        "mv3d_kitti_splits",
        "unused.txt",
    )
    previous_unused_files = read_txt(path_to_previous_unused)
    uncertainty_tfru_path = os.path.join(
        path_to_trial_folder,
        config["name"],
        "partition_{}".format(current_cycle - 1),
        "uncertainty.tfru",
    )

    uncertainty_calculator = initialize_class(
        config["uncertainty_calculator"],
        dataset_name=config["dataset"]["name"],
        config_name=config["name"],
        current_cycle=current_cycle,
        trial_number=trial_number,
    )
    uncertainty_dict = {}
    for f in tqdm(previous_unused_files):
        uncertainty_dict[f] = uncertainty_calculator.calculate(f)
    write_uncertainty_dict(uncertainty_dict, uncertainty_tfru_path)
