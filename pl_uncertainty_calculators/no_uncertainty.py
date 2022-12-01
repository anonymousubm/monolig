import os

from dd3d.active_learning.utils.training_utils import get_latest_experiment_folder
from dd3d.active_learning.utils.object_utils import read_labels_as_objects
from dd3d.active_learning.utils.dataset_utils import write_txt
from pl_uncertainty_calculators.base_uncertainty_calculator import (
    BaseUncertaintyCalculator,
)
from constants import PATH_TO_EXPERIMENTS_FOLDER, VALID_CLASS_NAMES

class NoUncertainty(BaseUncertaintyCalculator):
    def __init__(self, config_name, dataset_name, current_cycle, trial_number):
        super().__init__()
        path_to_lidar_experiments = get_latest_experiment_folder(
            PATH_TO_EXPERIMENTS_FOLDER,
            "{}_t{}_{}_{}_l".format(dataset_name, trial_number, config_name, current_cycle),
        )
        self.path_to_lidar_predictions = os.path.join(
            path_to_lidar_experiments, "eval/epoch_20/unused/default/final_result/data/"
        )
        self.path_to_output_uncertainty = os.path.join(
            PATH_TO_EXPERIMENTS_FOLDER,
            "{}_t{}_{}_{}_unc".format(dataset_name, trial_number, config_name, current_cycle),
            "uncertainties",
        )
        os.makedirs(self.path_to_output_uncertainty, exist_ok=True)

    def calculate(self, file, debug=False):
        path_to_lidar_prediction = file
        lidar_predictions = read_labels_as_objects(path_to_lidar_prediction)

        labeled_flag = 0 
        uncertainties = [
            "{} 1.0".format(labeled_flag)
            for pred in lidar_predictions
            if pred.type in VALID_CLASS_NAMES
        ]

        path_to_output_txt = os.path.join(
            self.path_to_output_uncertainty, file.split("/")[-1]
        )
        write_txt(uncertainties, path_to_output_txt)
