import os
import glob
import yaml
import subprocess

from constants import PATH_TO_EXPERIMENTS_FOLDER, DD3D_DEPTH_PRETRAINED_PATH, PATH_TO_DATA_FOLDER

def prepare_prediction_config(
    trial_number, config_name, dataset_name, current_cycle, config, model_num
):
    with open("dd3d/configs/experiments/{}".format(config["dd3d_base_prediction_yaml"]), "r") as f:
        template_yaml = yaml.safe_load(f)

    template_yaml["MODEL"]["CKPT"] = "PATH_TO_BEST_CKPT"

    dataset_path = os.path.join(
        PATH_TO_DATA_FOLDER,
        "active_learning",
        dataset_name,
        "trial_{}/{}/partition_{}/".format(
            trial_number, config_name, current_cycle
        ),
    )
    assert os.path.exists(
        os.path.join(dataset_path, "KITTI3D")
    ), "Dataset root does not include KITTI3D: {}".format(dataset_path)
    template_yaml["DATASET_ROOT"] = dataset_path

    ## Write back yaml, add package global on top
    path_to_out_yaml = (
        "dd3d/configs/experiments/{}_t{}_{}_{}_p{}.yaml".format(
            dataset_name, trial_number, config_name, current_cycle, model_num
        )
    )
    with open(path_to_out_yaml, "w", encoding="utf8") as f:
        yaml.dump(template_yaml, f, default_flow_style=False, allow_unicode=True)
    with open(path_to_out_yaml, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write("# @package _global_" + "\n" + content)

def read_experiment_results(
    experiment_path, result_type="kitti_3d_val/kitti_box3d_r40/Car_Moderate_0.7 "
):
    log_path = os.path.join(experiment_path, "logs", "log.txt")
    with open(log_path, "r") as f:
        log_lines = f.read().splitlines()

    steps = []
    results = []
    for line in log_lines:
        if "  iter: " in line:
            try:
                step_number = int(line.split(" ")[9])
            except:
                pass
        if result_type in line:
            score = float(line.split("|")[-2])
            # result_dict[step_number] = score
            steps.append(step_number)
            results.append(score)
    steps[-1] = steps[-1] + 20
    result_dict = {steps[i]: results[i] for i in range(len(steps))}
    return result_dict

def get_latest_experiment_folder(path_to_experiments_folder, config_name):
    sub_folders = glob.glob(os.path.join(path_to_experiments_folder, config_name, "*"))
    if len(sub_folders) == 0:
        return None
    max_date = max([folder.split("/")[-1].split("_")[-2] for folder in sub_folders])
    max_date_folders = [
        folder
        for folder in sub_folders
        if folder.split("/")[-1].split("_")[-2] == max_date
    ]
    max_time = max(
        [folder.split("/")[-1].split("_")[-1] for folder in max_date_folders]
    )
    max_time_folder = [
        folder
        for folder in sub_folders
        if folder.split("/")[-1].split("_")[-1] == max_time
    ][0]
    return max_time_folder

def link_initial_experiment(dataset_name, trial_number, config_name, initial_config_name, num_models):
    intial_experiment_name = "{}_t{}_{}_{}_l".format(
        dataset_name, trial_number, initial_config_name, 0
    )
    experiment_name = "{}_t{}_{}_{}_l".format(
        dataset_name, trial_number, config_name, 0
    )
    os.symlink(
        os.path.join(PATH_TO_EXPERIMENTS_FOLDER, intial_experiment_name),
        os.path.join(PATH_TO_EXPERIMENTS_FOLDER, experiment_name),
    )
    intial_experiment_name = "{}_t{}_{}_{}_unc".format(
        dataset_name, trial_number, initial_config_name, 0
    )
    experiment_name = "{}_t{}_{}_{}_unc".format(
        dataset_name, trial_number, config_name, 0
    )
    os.symlink(
        os.path.join(PATH_TO_EXPERIMENTS_FOLDER, intial_experiment_name),
        os.path.join(PATH_TO_EXPERIMENTS_FOLDER, experiment_name),
    )
    for model_num in range(num_models):
        intial_experiment_name = "{}_t{}_{}_{}_m{}".format(
            dataset_name, trial_number, initial_config_name, 0, model_num
        )
        experiment_name = "{}_t{}_{}_{}_m{}".format(
            dataset_name, trial_number, config_name, 0, model_num
        )
        os.symlink(
            os.path.join(PATH_TO_EXPERIMENTS_FOLDER, intial_experiment_name),
            os.path.join(PATH_TO_EXPERIMENTS_FOLDER, experiment_name),
        )

        intial_experiment_name = "{}_t{}_{}_{}_p{}".format(
            dataset_name, trial_number, initial_config_name, 0, model_num
        )
        experiment_name = "{}_t{}_{}_{}_p{}".format(
            dataset_name, trial_number, config_name, 0, model_num
        )
        os.symlink(
            os.path.join(PATH_TO_EXPERIMENTS_FOLDER, intial_experiment_name),
            os.path.join(PATH_TO_EXPERIMENTS_FOLDER, experiment_name),
        )
        
def find_best_ckpt(
    trial_number, config_name, dataset_name, current_cycle, model_num
):
    if current_cycle == 0:
        return DD3D_DEPTH_PRETRAINED_PATH

    path_to_previous_experiment_folder = get_latest_experiment_folder(
        PATH_TO_EXPERIMENTS_FOLDER,
        "{}_t{}_{}_{}_m{}".format(
            dataset_name, trial_number, config_name, current_cycle - 1, model_num
        ),
    )
    results = read_experiment_results(path_to_previous_experiment_folder)
    best_ckpt = sorted(results, key=lambda k: results[k], reverse=True)[0]
    path_to_best_ckpt = os.path.join(
        path_to_previous_experiment_folder, "model_{:07d}.pth".format(best_ckpt - 1)
    )
    return path_to_best_ckpt

def prepare_combined_config(
    trial_number, config_name, dataset_name, current_cycle, config, model_num
):
    with open("dd3d/configs/experiments/{}".format(config["dd3d_base_yaml"]), "r") as f:
        template_yaml = yaml.safe_load(f)

    path_to_final_ckpt = find_best_ckpt(
        trial_number, config_name, dataset_name, current_cycle, model_num
    )

    assert os.path.exists(path_to_final_ckpt), "Path to checkpoint does not exist"
    template_yaml["MODEL"]["CKPT"] = path_to_final_ckpt

    template_yaml["MODEL"]["WEIGHT_LABELED"] = config["dataset"]["weight_labeled"]
    template_yaml["MODEL"]["WEIGHT_UNLABELED"] = config["dataset"]["weight_unlabeled"]

    dataset_path = os.path.join(
        PATH_TO_DATA_FOLDER,
        "active_learning",
        dataset_name,
        "trial_{}/{}/partition_{}/COMBINED".format(
            trial_number, config_name, current_cycle
        ),
    )
    assert os.path.exists(
        os.path.join(dataset_path, "KITTI3D")
    ), "Dataset root does not include KITTI3D: {}".format(dataset_path)
    template_yaml["DATASET_ROOT"] = dataset_path

    template_yaml["SOLVER"]["WARMUP_ITERS"] = config["dataset"]["warmup_iters"][
        current_cycle
    ]
    template_yaml["SOLVER"]["BASE_LR"] = config["dataset"]["learning_rate"][
        current_cycle
    ]
    template_yaml["SOLVER"]["CHECKPOINT_PERIOD"] = config["dataset"][
        "checkpoint_period"
    ][current_cycle]
    template_yaml["SOLVER"]["MAX_ITER"] = config["dataset"]["max_iter"][current_cycle]

    template_yaml["TEST"]["EVAL_PERIOD"] = config["dataset"]["eval_periods"][current_cycle]

    ## Write back yaml, add package global on top
    path_to_out_yaml = "dd3d/configs/experiments/{}_t{}_{}_{}_m{}.yaml".format(
        dataset_name, trial_number, config_name, current_cycle, model_num
    )
    print(path_to_out_yaml)
    with open(path_to_out_yaml, "w", encoding="utf8") as f:
        yaml.dump(template_yaml, f, default_flow_style=False, allow_unicode=True)
    with open(path_to_out_yaml, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write("# @package _global_" + "\n" + content)


def start_combined(trial_number, config_name, dataset_name, current_cycle, model_num):
    experiment_name = "{}_t{}_{}_{}_m{}".format(
        dataset_name, trial_number, config_name, current_cycle, model_num
    )
    prediction_name = "{}_t{}_{}_{}_p{}".format(
        dataset_name, trial_number, config_name, current_cycle, model_num
    )
    
    subprocess.call([
        "docker",
        "run",
        "--ipc=host",
        "--rm",
        "--net=host",
        "-v", "{}:/workdir".format(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "-v", "{}:{}".format(PATH_TO_EXPERIMENTS_FOLDER, PATH_TO_EXPERIMENTS_FOLDER),
        "-v", "{}:{}".format(PATH_TO_DATA_FOLDER, PATH_TO_DATA_FOLDER),
        "--runtime=nvidia",
        "dd3d",
        "/bin/bash",
        "/workdir/train_and_predict.sh",
        str(experiment_name),
        str(prediction_name),
    ])

def train_combined(trial_number, config_name, dataset_name, current_cycle, config, number_of_models):
    for model_num in range(number_of_models):
        prepare_prediction_config(trial_number, config_name, dataset_name, current_cycle, config, model_num)
        prepare_combined_config(
            trial_number, config_name, dataset_name, current_cycle, config, model_num
        )
        start_combined(trial_number, config_name, dataset_name, current_cycle, model_num)