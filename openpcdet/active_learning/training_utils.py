import os
import yaml
import subprocess
from constants import PATH_TO_EXPERIMENTS_FOLDER, PATH_TO_DATA_FOLDER

def prepare_data_create_config(trial_number, config_name, current_cycle, config):
    path_to_data_yaml_template = "openpcdet/tools/cfgs/dataset_configs/active_learning/base.yaml"
    with open(path_to_data_yaml_template, "r") as f:
        template_yaml = yaml.safe_load(f)

    template_yaml[
        "DATA_PATH"
    ] = "{}/active_learninglearning/{}/trial_{}/{}/partition_{}/KITTI3D".format(
        PATH_TO_DATA_FOLDER, config["dataset"]["name"], trial_number, config_name, current_cycle
    )
    template_yaml["DATA_SPLIT"] = {"train": "train", "test": "val", "unused": "unused"}
    template_yaml["INFO_PATH"] = {
        "train": ["kitti_infos_train.pkl"],
        "test": ["kitti_infos_val.pkl"],
        "unused": ["kitti_infos_unused.pkl"],
    }

    ## Write back yaml, add package global on top
    path_to_out_yaml = "openpcdet/tools/cfgs/dataset_configs/active_learning/{}_t{}_{}_{}_create.yaml".format(
        config["dataset"]["name"], trial_number, config_name, current_cycle
    )
    with open(path_to_out_yaml, "w", encoding="utf8") as f:
        yaml.dump(template_yaml, f, default_flow_style=False, allow_unicode=True)


def prepare_data_config(trial_number, config_name, current_cycle, config):
    path_to_data_yaml_template = "openpcdet/tools/cfgs/dataset_configs/active_learning/base.yaml"
    with open(path_to_data_yaml_template, "r") as f:
        template_yaml = yaml.safe_load(f)

    template_yaml[
        "DATA_PATH"
    ] = "{}/active_learning/{}/trial_{}/{}/partition_{}/KITTI3D".format(
        PATH_TO_DATA_FOLDER, config["dataset"]["name"], trial_number, config_name, current_cycle
    )
    template_yaml["DATA_SPLIT"] = {"train": "train", "test": "val"}
    template_yaml["INFO_PATH"] = {
        "train": ["kitti_infos_train.pkl"],
        "test": ["kitti_infos_val.pkl"],
    }

    ## Write back yaml, add package global on top
    path_to_out_yaml = "openpcdet/tools/cfgs/dataset_configs/active_learning/{}_t{}_{}_{}_l.yaml".format(
        config["dataset"]["name"], trial_number, config_name, current_cycle
    )
    with open(path_to_out_yaml, "w", encoding="utf8") as f:
        yaml.dump(template_yaml, f, default_flow_style=False, allow_unicode=True)


def prepare_data_pred_config(trial_number, config_name, current_cycle, config):
    path_to_data_yaml_template = "openpcdet/tools/cfgs/dataset_configs/active_learning/base.yaml"
    with open(path_to_data_yaml_template, "r") as f:
        template_yaml = yaml.safe_load(f)

    template_yaml[
        "DATA_PATH"
    ] = "{}/active_learning/{}/trial_{}/{}/partition_{}/KITTI3D".format(
        PATH_TO_DATA_FOLDER, config["dataset"]["name"], trial_number, config_name, current_cycle
    )
    template_yaml["DATA_SPLIT"] = {"train": "train", "test": "unused"}
    template_yaml["INFO_PATH"] = {
        "train": ["kitti_infos_train.pkl"],
        "test": ["kitti_infos_unused.pkl"],
    }

    ## Write back yaml, add package global on top
    path_to_out_yaml = "openpcdet/tools/cfgs/dataset_configs/active_learning/{}_t{}_{}_{}_l_pred.yaml".format(
        config["dataset"]["name"], trial_number, config_name, current_cycle
    )
    with open(path_to_out_yaml, "w", encoding="utf8") as f:
        yaml.dump(template_yaml, f, default_flow_style=False, allow_unicode=True)


def prepare_train_config(trial_number, config_name, current_cycle, config):
    path_to_openpcdet_yaml_template = "openpcdet/tools/cfgs/kitti_models/active_learning/{}".format(config["lidar_base_yaml"])
    with open(path_to_openpcdet_yaml_template, "r") as f:
        template_yaml = yaml.safe_load(f)

    template_yaml["DATA_CONFIG"][
        "_BASE_CONFIG_"
    ] = "cfgs/dataset_configs/active_learning/{}_t{}_{}_{}_l.yaml".format(
        config["dataset"]["name"], trial_number, config_name, current_cycle
    )

    ## Write back yaml, add package global on top
    path_to_out_yaml = "openpcdet/tools/cfgs/kitti_models/active_learning/{}_t{}_{}_{}_l.yaml".format(
        config["dataset"]["name"], trial_number, config_name, current_cycle
    )
    with open(path_to_out_yaml, "w", encoding="utf8") as f:
        yaml.dump(template_yaml, f, default_flow_style=False, allow_unicode=True)


def prepare_train_pred_config(trial_number, config_name, current_cycle, config):
    path_to_openpcdet_yaml_template = "openpcdet/tools/cfgs/kitti_models/active_learning/{}".format(config["lidar_base_yaml"])
    with open(path_to_openpcdet_yaml_template, "r") as f:
        template_yaml = yaml.safe_load(f)

    template_yaml["DATA_CONFIG"][
        "_BASE_CONFIG_"
    ] = "cfgs/dataset_configs/active_learning/{}_t{}_{}_{}_l_pred.yaml".format(
        config["dataset"]["name"], trial_number, config_name, current_cycle
    )

    ## Write back yaml, add package global on top
    path_to_out_yaml = "openpcdet/tools/cfgs/kitti_models/active_learning/{}_t{}_{}_{}_l_pred.yaml".format(
        config["dataset"]["name"], trial_number, config_name, current_cycle
    )
    with open(path_to_out_yaml, "w", encoding="utf8") as f:
        yaml.dump(template_yaml, f, default_flow_style=False, allow_unicode=True)


def prepare_training_config(trial_number, config_name, current_cycle, config):
    prepare_data_create_config(trial_number, config_name, current_cycle, config)
    prepare_data_config(trial_number, config_name, current_cycle, config)
    prepare_data_pred_config(trial_number, config_name, current_cycle, config)
    prepare_train_config(trial_number, config_name, current_cycle, config)
    prepare_train_pred_config(trial_number, config_name, current_cycle, config)


def start_training(trial_number, config_name, current_cycle, config):
    path_to_dataset = "{}/active_learning/{}/trial_{}/{}/partition_{}/KITTI3D".format(
        PATH_TO_DATA_FOLDER, config["dataset"]["name"], trial_number, config_name, current_cycle
    )
    dataset_config_name = "{}_t{}_{}_{}_create".format(
        config["dataset"]["name"], trial_number, config_name, current_cycle
    )
    experiment_name = "{}_t{}_{}_{}_l".format(config["dataset"]["name"], trial_number, config_name, current_cycle)
    
    subprocess.call([
        "docker",
        "run",
        "--ipc=host",
        "--rm",
        "--net=host",
        "-v", "{}:/root/openpcdet".format(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "-v", "{}:{}".format(PATH_TO_EXPERIMENTS_FOLDER, PATH_TO_EXPERIMENTS_FOLDER),
        "-v", "{}:{}".format(PATH_TO_DATA_FOLDER, PATH_TO_DATA_FOLDER),
        "--runtime=nvidia",
        "openpcdet-docker",
        "/bin/bash",
        "openpcdet/tools/data_train_and_predict.sh",
        str(experiment_name),
        str(path_to_dataset),
        str(dataset_config_name)
    ])

def train(trial_number, config_name, current_cycle, config):
    prepare_training_config(trial_number, config_name, current_cycle, config)
    start_training(trial_number, config_name, current_cycle, config)