import os
from active_learning.utils.object_utils import read_labels_as_objects
from active_learning.utils.dataset_utils import write_txt, read_txt

def find_index_of_label(labels, label):
    for l,lbl in enumerate(labels):
        if (label.type==lbl.type) and (abs(label.loc_x-lbl.loc_x)<1e-4) and (abs(label.loc_y-lbl.loc_y)<1e-4) and (abs(label.loc_z-lbl.loc_z)<1e-4):
            return l

class LidarAleatoric(object):
    def __init__(self) -> None:
        pass

    def read_labels(self, pseudo_gt_path, file):
        labels = read_labels_as_objects(os.path.join(pseudo_gt_path, "{}.txt".format(file)))
        return labels

    def read_aleatoric_uncertainties(self, pseudo_gt_path, file):
        weights = read_txt("/"+os.path.join(*(pseudo_gt_path.split("/")[:-1]+["weights", "{}.txt".format(file)])))
        aleatoric_uncertainties = [1-float(w.split(" ")[1]) for w in weights]
        return aleatoric_uncertainties
        
    def calculate(self, clusters, extra_information):
        """
        Calculate localization bias within clusters.
        If ground-truth available:
            - Bias is the maximum center distance of ground-truth to other objects
        If ground-truth not available:
            - Bias is the maximum of distances between objects.
        Args:
            clusters: [[GenericObject]]
        """
        pseudo_gt_path = extra_information["pseudo_gt_path"]
        file = extra_information["file"]

        lidar_aleatoric_uncertainties = []
        labels = self.read_labels(pseudo_gt_path, file)
        aleatoric_uncertainties = self.read_aleatoric_uncertainties(pseudo_gt_path, file)
        for cluster in clusters:
            if cluster[0].score == 1.0:
                label = cluster[0]
                idx = find_index_of_label(labels, label)
                lidar_aleatoric_uncertainties.append(aleatoric_uncertainties[idx])
            else:
                lidar_aleatoric_uncertainties.append(1.0)
        return lidar_aleatoric_uncertainties
