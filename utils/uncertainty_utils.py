import os
import glob
from tqdm import tqdm

from dd3d.active_learning.utils.class_utils import initialize_class

def compute_pseudo_label_uncertainty(trial_number, config_name, current_cycle, config):
    uncertainty_calculator = initialize_class(
        config["pl_uncertainty_calculator"],
        config_name=config["name"],
        dataset_name=config["dataset"]["name"],
        current_cycle=current_cycle,
        trial_number=trial_number,
    )
    lidar_predictions = glob.glob(
        os.path.join(uncertainty_calculator.path_to_lidar_predictions, "*.txt")
    )

    for f in tqdm(lidar_predictions):
        uncertainty_calculator.calculate(f)
