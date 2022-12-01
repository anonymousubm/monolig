import numpy as np

from active_learning.utils.object_utils import bev_center_distance


def nonzero_mean(x):
    if len(x) == 0:
        return 0
    x_nonzero = [i for i in x if i != 0]
    if len(x_nonzero) == 0:
        return 0
    return sum(x_nonzero) / len(x_nonzero)


class LocalizationBias(object):
    def __init__(self) -> None:
        pass

    def calculate(self, clusters, extra_information=None):
        """
        Calculate localization bias within clusters.
        Args:
            clusters: [[GenericObject]]
        """
        cluster_biases = []
        for cluster in clusters:
            biases = []
            label = cluster[0]
            predictions = cluster[1:]
            for pred in predictions:
                biases.append(bev_center_distance(label, pred))
            cluster_biases.append(nonzero_mean(biases))
        return cluster_biases
