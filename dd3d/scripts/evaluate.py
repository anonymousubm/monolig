#!/usr/bin/env python
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os
from collections import OrderedDict, defaultdict

import hydra
import torch
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from torch.cuda import amp
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

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

LOG = logging.getLogger("tridet")


@hydra.main(config_path="../configs/", config_name="defaults")
def main(cfg):
    setup(cfg)
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
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpoint_file = cfg.MODEL.CKPT
    if checkpoint_file:
        checkpointer.load(checkpoint_file)

    assert (
        cfg.TEST.ENABLED
    ), "'eval-only' mode is not compatible with 'cfg.TEST.ENABLED = False'."
    test_results = do_test(cfg, model, is_last=True)
    if cfg.TEST.AUG.ENABLED:
        test_results.update(do_test(cfg, model, is_last=True, use_tta=True))
    return test_results, cfg


def do_test(cfg, model, is_last=False, use_tta=False):
    if not cfg.TEST.ENABLED:
        LOG.warning("Test is disabled.")
        return {}

    dataset_names = [
        cfg.DATASETS.TEST.NAME
    ]  # NOTE: only support single test dataset for now.

    if use_tta:
        LOG.info("Starting inference with test-time augmentation.")
        if isinstance(model, DistributedDataParallel):
            model.module.postprocess_in_inference = False
        else:
            model.postprocess_in_inference = False
        model = build_tta_model(cfg, model)

    test_results = OrderedDict()
    for dataset_name in dataset_names:
        # output directory for this dataset.
        dset_output_dir = get_inference_output_dir(
            dataset_name, is_last=is_last, use_tta=use_tta
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
        if use_tta:
            per_dataset_results = OrderedDict(
                {k + "-tta": v for k, v in per_dataset_results.items()}
            )
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

    test_results = flatten_dict(test_results)
    log_nested_dict(test_results)
    if d2_comm.is_main_process():
        LOG.info("Evaluation results for {} in csv format:".format(dataset_name))
        print_test_results(test_results)

    if use_tta:
        if isinstance(model, DistributedDataParallel):
            model.module.postprocess_in_inference = True
        else:
            model.postprocess_in_inference = True

    return test_results


if __name__ == "__main__":
    setup_hydra_path()
    main()  # pylint: disable=no-value-for-parameter
    LOG.info("DONE.")
