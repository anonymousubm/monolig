import os
import random
import click
import torch
import numpy as np
import dd3d.active_learning.utils.config_utils as config_utils
import dd3d.active_learning.utils.dataset_utils as dataset_utils
import dd3d.active_learning.utils.uncertainty_utils as uncertainty_utils
from dd3d.active_learning.utils.training_utils import (
    train_combined,
    link_initial_experiment,
)
from openpcdet.active_learning.training_utils import train as train_openpcdet
from openpcdet.active_learning.dataset_utils import build_combined_dataset
from utils.uncertainty_utils import compute_pseudo_label_uncertainty
from constants import PATH_TO_DATA_FOLDER

def seed_all_rng(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_action_list(current_cycle, config_name, initial_config_name):
    if current_cycle == 0:
        if config_name == initial_config_name:
            action_list = ["BUILD", "TRAIN_LIDAR", "PL_UNCERTAINTY", "TRAIN"]
        else:
            action_list = ["BUILD"]
    else:
        action_list = [
            "UNCERTAINTY",
            "BUILD",
            "TRAIN_LIDAR",
            "PL_UNCERTAINTY",
            "TRAIN",
        ]
    return action_list


@click.command()
@click.option("--trial_number", required=True, type=int)
@click.option("--config_name", required=True, type=str)
@click.option("--dataset_name", required=True, type=str)
@click.option("--number_of_models", required=False, type=int, default=1)
@click.option("--start_cycle", required=False, type=int, default=0)
@click.option("--start_action", required=False, type=str, default="BUILD")
def al_loop(
    trial_number,
    config_name,
    dataset_name,
    number_of_models,
    start_cycle,
    start_action,
):
    seed_all_rng(trial_number)
    config = config_utils.read_config(config_name, dataset_name)
    path_to_trial_folder = "{}/active_learning/{}/trial_{}".format(
        PATH_TO_DATA_FOLDER, config["dataset"]["name"], trial_number
    )
    dataset_utils.create_trial_folder(
        path_to_trial_folder,
        config["dataset"]["name"],
        config["dataset"]["num_samples"][0],
    )
    cycle_action_list = []
    for current_cycle in range(config["dataset"]["num_cycles"]):
        for action in get_action_list(
            current_cycle, config["name"], config["initial_config"]
        ):
            cycle_action_list.append((current_cycle, action))
    cycle_action_list = cycle_action_list[
        cycle_action_list.index((start_cycle, start_action)) :
    ]
    
    for current_cycle, action in cycle_action_list:
        print(current_cycle, action)
    
        if action == "UNCERTAINTY":
            uncertainty_utils.compute_uncertainty(
                trial_number, path_to_trial_folder, config, current_cycle
            )
        elif action == "BUILD":
            if (current_cycle == 0) and (config_name != config["initial_config"]):
                dataset_utils.link_initial_dataset(
                    path_to_trial_folder, config_name, config["initial_config"]
                )
                link_initial_experiment(
                    dataset_name,
                    trial_number,
                    config_name,
                    config["initial_config"],
                    number_of_models,
                )
            else:
                dataset_utils.build_dataset_iteration(
                    path_to_trial_folder,
                    config["name"],
                    config["dataset"]["name"],
                    current_cycle,
                    config["dataset"]["num_samples"][current_cycle],
                )
        elif action == "TRAIN_LIDAR":
            train_openpcdet(trial_number, config["name"], current_cycle, config)
        elif action == "PL_UNCERTAINTY":
            compute_pseudo_label_uncertainty(
                trial_number, config["name"], current_cycle, config
            )
        elif action == "TRAIN":
            build_combined_dataset(
                trial_number, config["name"], config["dataset"]["name"], current_cycle
            )
            train_combined(
                trial_number,
                config["name"],
                config["dataset"]["name"],
                current_cycle,
                config,
                number_of_models,
            )


if __name__ == "__main__":
    al_loop()
