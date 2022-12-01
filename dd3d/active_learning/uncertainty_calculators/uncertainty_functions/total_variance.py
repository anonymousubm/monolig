import numpy as np


def total_variance(cluster):
    predictions = [obj for obj in cluster if obj.score<1.0]
    if len(predictions)==0:
        return 0
    cluster_xy = np.array([(obj.loc_x, obj.loc_z) for obj in predictions])
    var_xy = np.var(cluster_xy, axis=0)
    total_var = np.sqrt(np.sum(var_xy))
    return max(0, total_var)


class TotalVariance(object):
    def __init__(self) -> None:
        pass

    def calculate(self, clusters, extra_information=None):
        return [total_variance(cluster) for cluster in clusters]
