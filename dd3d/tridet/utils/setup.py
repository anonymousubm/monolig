# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import json
import logging
import os
import sys
import resource
from datetime import datetime

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import detectron2.utils.comm as d2_comm
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import _CURRENT_STORAGE_STACK

from tridet.utils.comm import broadcast_from_master
from tridet.utils.events import WandbEventStorage

from active_learning.utils.training_utils import find_best_ckpt

LOG = logging.getLogger(__name__)


def setup_hydra_path():
    args = sys.argv
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    exp_config = None
    for arg in args:
        if "+experiments=" in arg:
            exp_config = arg.split("=")[1]
            break
    if exp_config is not None:
        main_folder_path = os.path.join(
            "/path/to/experiments", exp_config
        )
        exp_folder_path = os.path.join(main_folder_path, dt_string)
        os.makedirs(exp_folder_path, exist_ok=True)
        sys.argv.append("hydra.run.dir={}".format(exp_folder_path))


def setup_distributed(world_size, rank):
    """
    Adapted from detectron2:
        https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py#L85
    """
    host = os.environ["MASTER_ADDR"] if "MASTER_ADDR" in os.environ else "127.0.0.1"
    port = 12345
    dist_url = f"tcp://{host}:{port}"
    try:
        dist.init_process_group(
            backend="NCCL", init_method=dist_url, world_size=world_size, rank=rank
        )
    except Exception as e:
        logging.error("Process group URL: %s", dist_url)
        raise e
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    d2_comm.synchronize()

    # Assumption: all machines have the same number of GPUs.
    num_gpus_per_machine = torch.cuda.device_count()
    machine_rank = rank // num_gpus_per_machine

    # Setup the local process group (which contains ranks within the same machine)
    assert d2_comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            d2_comm._LOCAL_PROCESS_GROUP = pg

    # Declare GPU device.
    local_rank = rank % num_gpus_per_machine
    torch.cuda.set_device(local_rank)

    # Multi-node training often fails with "received 0 items of ancdata" error.
    # https://github.com/fastai/fastai/issues/23#issuecomment-345091054
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


@broadcast_from_master
def get_random_seed():
    for arg in sys.argv:
        if "+experiments=" in arg:
            exp_config = arg.split("=")[1]
            break
    try:
        _, trial, _, _, model = exp_config.split("_")
        seed = 10 * int(trial[1:]) + int(model[1:])
    except:
        """Adapted from d2.utils.env:seed_all_rng()"""
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
    return seed

def get_previous_experiment_checkpoint(cfg):
    previous_config = None
    path_to_prev_ckpt = None
    for arg in sys.argv:
        if "+previous_experiment=" in arg:
            previous_config = arg.split("=")[1]
            dataset_name, trial, config_name, currrent_cycle, model = previous_config.split("_")
            trial_number = int(trial[1:])
            current_cycle = int(currrent_cycle)+1
            model_num = int(model[1:])
            path_to_prev_ckpt = find_best_ckpt(
                trial_number, config_name, dataset_name, current_cycle, model_num
            )
            assert os.path.exists(path_to_prev_ckpt), "Path to checkpoint does not exist"
            print("Best ckpt found is {}".format(path_to_prev_ckpt), flush=True)
            break
    if path_to_prev_ckpt:
        cfg.MODEL.CKPT = path_to_prev_ckpt

def setup(cfg):
    assert torch.cuda.is_available(), "cuda is not available."

    # Seed random number generators. If distributed, then sync the random seed over all GPUs.
    seed = get_random_seed()
    LOG.info("Setting seed to {}".format(seed))
    seed_all_rng(seed)

    get_previous_experiment_checkpoint(cfg)
    LOG.info("Working Directory: {}".format(os.getcwd()))
    LOG.info(
        "Full config:\n{}".format(
            json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)
        )
    )

    # Set up EventStorage
    storage = WandbEventStorage()
    _CURRENT_STORAGE_STACK.append(storage)

    # After this, the cfg is immutable.
    OmegaConf.set_readonly(cfg, True)
