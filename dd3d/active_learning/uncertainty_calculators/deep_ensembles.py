import os
import numpy as np
from collections import OrderedDict

from scripts.visualize_predictions import (
    get_3d_annotation,
    get_2d_annotation,
)
from active_learning.uncertainty_calculators.base_uncertainty_calculator import (
    BaseUncertaintyCalculator,
)
from constants import PATH_TO_EXPERIMENTS_FOLDER, PATH_TO_DATA_FOLDER
from active_learning.utils.training_utils import get_latest_experiment_folder
from active_learning.utils.object_utils import (
    read_labels_as_objects,
    threshold_objects,
    threshold_class,
)
from active_learning.utils.class_utils import initialize_class


def convert_cluster_to_kitti_annotations(cluster_objects, image_id):
    sample_annotations = [obj.kitti_annotation_list for obj in cluster_objects]
    class_names = ("Car", "Pedestrian", "Cyclist", "Van", "Truck")
    name_to_id = {name: idx for idx, name in enumerate(class_names)}

    sample_id = image_id
    annotations = []
    for idx, kitti_annotation in enumerate(sample_annotations):
        class_name = kitti_annotation[0]
        if class_name not in class_names:
            continue
        annotation = OrderedDict(
            category_id=name_to_id[class_name], instance_id=f"{sample_id}_{idx}"
        )
        annotation.update(get_3d_annotation(kitti_annotation))
        annotation.update(get_2d_annotation(kitti_annotation))
        annotations.append(annotation)
    return annotations

class DeepEnsembles(BaseUncertaintyCalculator):
    def __init__(
        self,
        dataset_name,
        config_name,
        current_cycle,
        trial_number,
        number_of_models,
        clustering_strategy,
        uncertainty_functions,
        combination_logic,
    ):
        super().__init__()
        self.number_of_models = number_of_models
        self.pseudo_gt_path = "{}/active_learning/{}/trial_{}/{}/partition_{}/COMBINED/KITTI3D/training/label_2".format(
            PATH_TO_DATA_FOLDER, dataset_name, trial_number, config_name, current_cycle - 1
        )
        self.model_prediction_paths = []
        self.clustering_strategy = initialize_class(clustering_strategy)
        self.uncertainty_functions = [
            initialize_class(fn) for fn in uncertainty_functions
        ]
        self.combination_logic = initialize_class(combination_logic)

        for model_num in range(number_of_models):
            path_to_predictions_folder = get_latest_experiment_folder(
                PATH_TO_EXPERIMENTS_FOLDER,
                "{}_t{}_{}_{}_p{}".format(
                    dataset_name, trial_number, config_name, current_cycle-1, model_num
                ),
            )
            self.model_prediction_paths.append(
                os.path.join(path_to_predictions_folder, "kitti_predictions")
            )

    def calculate(self, file, debug=False):
        model_predictions = [
            read_labels_as_objects(os.path.join(path, "{}.txt".format(file)))
            for path in self.model_prediction_paths
        ]
        thresholded_predictions = [
            threshold_objects(objects) for objects in model_predictions
        ]

        gt_labels = read_labels_as_objects(
            os.path.join(self.pseudo_gt_path, "{}.txt".format(file))
        )
        gt_labels = threshold_class(gt_labels, "DontCare")

        gt_ensembles_predictions = thresholded_predictions + [gt_labels]

        clusters = self.clustering_strategy.cluster(gt_ensembles_predictions)
        extra_information = {}
        extra_information["labels"] = gt_labels
        extra_information["predictions"] = thresholded_predictions
        extra_information["pseudo_gt_path"] = self.pseudo_gt_path
        extra_information["file"] = file

        uncertainties = [
            uncertainty_fn.calculate(clusters, extra_information)
            for uncertainty_fn in self.uncertainty_functions
        ]
        uncertainties = self.combination_logic.combine(uncertainties)

        if len(uncertainties) == 0:
            uncertainties = [0]
        return max(uncertainties)