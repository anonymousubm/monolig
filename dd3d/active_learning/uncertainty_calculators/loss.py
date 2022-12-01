import os

from active_learning.uncertainty_calculators.base_uncertainty_calculator import (
    BaseUncertaintyCalculator,
)

from constants import PATH_TO_EXPERIMENTS_FOLDER
from active_learning.utils.training_utils import get_latest_experiment_folder

class Loss(BaseUncertaintyCalculator):
    def __init__(
        self,
        dataset_name,
        config_name,
        current_cycle,
        trial_number,
        number_of_models,
        loss_type,
    ):
        super().__init__()
        self.model_prediction_paths = []
        self.number_of_models = number_of_models
        
        for model_num in range(number_of_models):
            path_to_predictions_folder = get_latest_experiment_folder(
                PATH_TO_EXPERIMENTS_FOLDER,
                "{}_t{}_{}_{}_p{}".format(
                    dataset_name, trial_number, config_name, current_cycle-1, model_num
                ),
            )
            self.model_prediction_paths.append(
                os.path.join(path_to_predictions_folder, loss_type)
            )

    def calculate(self, file, debug=False):
        ensembles_losses = []
        for model_prediction_path in self.model_prediction_paths:
            file_loss_path = os.path.join(model_prediction_path, "{}.txt".format(file))
            with open(file_loss_path, "r") as f:
                loss = float(f.read().splitlines()[0])
                ensembles_losses.append(loss)
        average_loss = sum(ensembles_losses)/len(ensembles_losses)
        return average_loss
