#!/usr/bin/env python
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os
import json
from collections import OrderedDict, defaultdict
import torch.nn as nn 

import hydra
import torch
import wandb
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from torch.cuda import amp
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from active_learning.modelling.loss_prediction import LossPrediction, build_loss_prediction_optimizer

import detectron2.utils.comm as d2_comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluators, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, get_event_storage

import tridet.modeling  # pylint: disable=unused-import
import tridet.utils.comm as comm
from tridet.data import build_test_dataloader, build_train_dataloader
from tridet.data.dataset_mappers import get_dataset_mapper
from tridet.data.datasets import random_sample_dataset_dicts, register_datasets
from tridet.evaluators import get_evaluator
from tridet.modeling import build_tta_model
from tridet.utils.s3 import sync_output_dir_s3
from tridet.utils.setup import setup, setup_hydra_path
from tridet.utils.train import get_inference_output_dir, print_test_results
from tridet.utils.visualization import mosaic, save_vis
from tridet.utils.wandb import flatten_dict, log_nested_dict
from tridet.visualizers import get_dataloader_visualizer, get_predictions_visualizer
from active_learning.utils.object_utils import GenericObject
from active_learning.utils.dataset_utils import read_txt, write_txt

LOG = logging.getLogger("tridet")

def split_predictions_to_kitti_format(path_to_experiment_folder, path_to_data_folder):
    """
    Split bbox3d_predictions.json file into kitti label format.
    Args:
        path_to_experiment_folder(str): Experiment folder containing inference folder.
        path_to_data_folder(str): Data folder contraining unused.txt splits file.
    """
    path_to_camera_predictions = os.path.join(
        path_to_experiment_folder,
        "inference",
        "final",
        "kitti_3d_unused",
        "bbox3d_predictions.json",
    )
    path_to_unused_file = os.path.join(
        path_to_data_folder, "KITTI3D", "mv3d_kitti_splits", "unused.txt"
    )
    unused_files = set(read_txt(path_to_unused_file))
    with open(path_to_camera_predictions, "r") as f:
        camera_predictions = json.load(f)

    path_to_output_label_folder = "/".join(
        path_to_camera_predictions.split("/")[:-4] + ["kitti_predictions"]
    )
    os.makedirs(path_to_output_label_folder, exist_ok=True)

    image_id_to_predictions = {image_id: [] for image_id in unused_files}
    for pred_json in tqdm(camera_predictions):
        image_id = pred_json["image_id"].split("_")[0]
        image_id_to_predictions[image_id].append(pred_json)

    for image_id, predictions in tqdm(image_id_to_predictions.items()):
        object_lines = [
            GenericObject(pred_json, input_type="prediction_json").kitti_string
            for pred_json in predictions
        ]
        write_txt(
            object_lines,
            os.path.join(path_to_output_label_folder, "{}.txt".format(image_id)),
        )

def predict_loss(cfg, dataloader, model, model_loss_pred, path_to_experiment_folder):
    model_loss_pred.eval()
    path_to_losses_gt = os.path.join(path_to_experiment_folder, "losses_gt")
    path_to_losses_predicted = os.path.join(path_to_experiment_folder, "losses_predicted")
    os.makedirs(path_to_losses_gt, exist_ok=True)
    os.makedirs(path_to_losses_predicted, exist_ok=True)

    for data in dataloader:
        gt_losses, _ = model_loss_pred.extract_features(data, model)
        pred_losses = model_loss_pred(data, model)
        pred_losses = [pred.item() for pred in pred_losses]
        gt_losses = [gt.item() for gt in gt_losses]
        for single_data, gt_loss, pred_loss in zip(data, gt_losses, pred_losses):
            path_to_single_loss_gt = os.path.join(path_to_losses_gt, "{}.txt".format(single_data["sample_id"]))
            with open(path_to_single_loss_gt, "w") as f:
                f.write("{:.5f}\n".format(gt_loss))
            
            path_to_single_loss_pred = os.path.join(path_to_losses_predicted, "{}.txt".format(single_data["sample_id"]))
            with open(path_to_single_loss_pred, "w") as f:
                f.write("{:.5f}\n".format(pred_loss))
                
@hydra.main(config_path="../configs/", config_name="defaults")
def main(cfg):
    setup(cfg)
    path_to_experiment_folder = os.getcwd()
    path_to_data_folder = cfg.DATASET_ROOT
    dataset_names = register_datasets(cfg)
    if cfg.ONLY_REGISTER_DATASETS:
        return {}, cfg
    LOG.info(
        f"Registered {len(dataset_names)} datasets:"
        + "\n\t"
        + "\n\t".join(dataset_names)
    )
    model = build_model(cfg)
    checkpointer = Checkpointer(model, "./")
    
    model_loss_pred = None
    if ("LOSS_MODULE" in cfg["MODEL"]) and cfg["MODEL"]["LOSS_MODULE"]:
        model_loss_pred = LossPrediction()
        model_loss_pred.to(torch.device(cfg.MODEL.DEVICE))
        checkpointer_loss_pred = Checkpointer(model_loss_pred, "./")
        
        checkpoint_file = cfg.MODEL.CKPT
        loss_checkpoint_file = "/"+os.path.join(*(checkpoint_file.split("/")[:-1]+["loss_model.pth"]))
        if os.path.exists(loss_checkpoint_file):
            checkpointer_loss_pred.load(loss_checkpoint_file)

    checkpoint_file = cfg.MODEL.CKPT
    if checkpoint_file:
        checkpointer.load(checkpoint_file)

    if not cfg.TEST.ENABLED:
        LOG.warning("Test is disabled.")
        return {}
    dataset_names = [
        cfg.DATASETS.TEST.NAME
    ]  # NOTE: only support single test dataset for now.

    test_results = OrderedDict()
    for dataset_name in dataset_names:
        # output directory for this dataset.
        dset_output_dir = get_inference_output_dir(
            dataset_name, is_last=True, use_tta=False
        )

        # What evaluators are used for this dataset?
        evaluator_names = MetadataCatalog.get(dataset_name).evaluators
        evaluators = []
        for evaluator_name in evaluator_names:
            evaluator = get_evaluator(
                cfg, dataset_name, evaluator_name, dset_output_dir
            )
            evaluators.append(evaluator)
        evaluator = DatasetEvaluators(evaluators)

        mapper = get_dataset_mapper(cfg, is_train=False)
        dataloader, dataset_dicts = build_test_dataloader(cfg, dataset_name, mapper)

        per_dataset_results = inference_on_dataset(model, dataloader, evaluator)
        test_results[dataset_name] = per_dataset_results

        if cfg.VIS.PREDICTIONS_ENABLED and d2_comm.is_main_process():
            visualizer_names = MetadataCatalog.get(dataset_name).pred_visualizers
            # Randomly (but deterministically) select what samples to visualize.
            # The samples are shared across all visualizers and iterations.
            sampled_dataset_dicts, inds = random_sample_dataset_dicts(
                dataset_name, num_samples=cfg.VIS.PREDICTIONS_MAX_NUM_SAMPLES
            )

            viz_images = defaultdict(dict)
            for viz_name in visualizer_names:
                LOG.info(f"Running prediction visualizer: {viz_name}")
                visualizer = get_predictions_visualizer(
                    cfg, viz_name, dataset_name, dset_output_dir
                )
                for x in tqdm(sampled_dataset_dicts):
                    sample_id = x["sample_id"]
                    viz_images[sample_id].update(visualizer.visualize(x))

            save_vis(viz_images, dset_output_dir, "visualization")

            if cfg.WANDB.ENABLED:
                LOG.info(f"Uploading prediction visualization to W&B: {dataset_name}")
                for sample_id in viz_images.keys():
                    viz_images[sample_id] = mosaic(list(viz_images[sample_id].values()))
                step = get_event_storage().iter
                wandb.log(
                    {
                        f"{dataset_name}-predictions": [
                            wandb.Image(viz, caption=f"{sample_id}")
                            for sample_id, viz in viz_images.items()
                        ]
                    },
                    step=step,
                )

    test_results = flatten_dict(test_results)
    log_nested_dict(test_results)
    if d2_comm.is_main_process():
        LOG.info("Evaluation results for {} in csv format:".format(dataset_name))
        print_test_results(test_results)

    split_predictions_to_kitti_format(path_to_experiment_folder, path_to_data_folder)

    if model_loss_pred:
        predict_loss(cfg, dataloader, model, model_loss_pred, path_to_experiment_folder)

if __name__ == "__main__":
    setup_hydra_path()
    main()  # pylint: disable=no-value-for-parameter
    LOG.info("DONE.")
