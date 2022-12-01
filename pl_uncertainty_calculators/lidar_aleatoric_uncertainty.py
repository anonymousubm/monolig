import os

from dd3d.active_learning.utils.training_utils import get_latest_experiment_folder
from dd3d.active_learning.utils.object_utils import (
    read_labels_as_objects,
    threshold_objects,
    threshold_valid_classes,
    write_objects_as_labels,
)
from dd3d.active_learning.utils.dataset_utils import write_txt, read_txt
from pl_uncertainty_calculators.base_uncertainty_calculator import (
    BaseUncertaintyCalculator,
)
from constants import PATH_TO_EXPERIMENTS_FOLDER, VALID_CLASS_NAMES

class LidarAleatoricUncertainty(BaseUncertaintyCalculator):
    def __init__(self, config_name, dataset_name, current_cycle, trial_number, uncertainty_threshold=0.5):
        super().__init__()
        self.uncertainty_threshold = uncertainty_threshold
        
        path_to_lidar_experiments = get_latest_experiment_folder(
            PATH_TO_EXPERIMENTS_FOLDER,
            "{}_t{}_{}_{}_l".format(dataset_name, trial_number, config_name, current_cycle),
        )
        
        self.path_to_lidar_predictions = os.path.join(
            path_to_lidar_experiments, "eval/epoch_20/unused/default/final_result/data/"
        )
        self.path_to_lidar_aleatorics = os.path.join(
            path_to_lidar_experiments, "eval/epoch_20/unused/default/final_result/aleatorics/"
        )
        
        self.path_to_output_weights = os.path.join(
            PATH_TO_EXPERIMENTS_FOLDER,
            "{}_t{}_{}_{}_unc".format(dataset_name, trial_number, config_name, current_cycle),
            "weights",
        )
        os.makedirs(self.path_to_output_weights, exist_ok=True)
        self.path_to_output_labels = os.path.join(
            PATH_TO_EXPERIMENTS_FOLDER,
            "{}_t{}_{}_{}_unc".format(dataset_name, trial_number, config_name, current_cycle),
            "label_2",
        )
        os.makedirs(self.path_to_output_labels, exist_ok=True)

        self.path_to_output_visuals = os.path.join(
            PATH_TO_EXPERIMENTS_FOLDER,
            "{}_t{}_{}_{}_unc".format(dataset_name, trial_number, config_name, current_cycle),
            "visuals",
        )
        os.makedirs(self.path_to_output_visuals, exist_ok=True)

    def calculate(self, file, debug=False):
        path_to_lidar_prediction = file
        lidar_predictions = read_labels_as_objects(path_to_lidar_prediction)
        path_to_lidar_aleatorics = os.path.join(
            self.path_to_lidar_aleatorics, file.split("/")[-1]
        )
        lidar_aleatorics = read_txt(path_to_lidar_aleatorics)

        weights = []
        filtered_labels = []
        
        for prediction, aleatoric in zip(lidar_predictions, lidar_aleatorics):
            if (prediction.score < 0.5) or (prediction.type not in VALID_CLASS_NAMES):
                continue
            alea_x, alea_y = aleatoric.split(" ")
            alea_x = float(alea_x) * 100
            alea_y = float(alea_y) * 100            
            
            weight = 1 - (alea_x + alea_y) / 2
            
            if weight > self.uncertainty_threshold:
                weights.append(weight)
                filtered_labels.append(prediction)
            
        weights = ["0 {}".format(w) for w in weights]
        path_to_output_weight_txt = os.path.join(
            self.path_to_output_weights, file.split("/")[-1]
        )
        path_to_output_label_txt = os.path.join(
            self.path_to_output_labels, file.split("/")[-1]
        )
        write_txt(weights, path_to_output_weight_txt)
        write_objects_as_labels(filtered_labels, path_to_output_label_txt)
