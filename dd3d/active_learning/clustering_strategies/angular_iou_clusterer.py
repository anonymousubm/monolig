import numpy as np
import active_learning.utils.object_utils as object_utils


class AngularIoUClusterer(object):
    def __init__(
        self,
        bev_iou_threshold=0.3,
        angle_threshold=1.0,
        center_distance_threshold=5.0,
    ):
        """
        Angular and IoU based clustering. Look at objects_matching for description of steps.
        Args:
            bev_iou_threshold: IoU threshold to apply (BeV space)
            param angle_threshold: Angle threshold to apply (in degrees) default ~ 0.0025 radians
            param center_distance_threshold: Distance threshold to apply (in meters)
        """
        super(AngularIoUClusterer, self).__init__()
        self.bev_iou_threshold = bev_iou_threshold
        self.angle_threshold = angle_threshold
        self.center_distance_threshold = center_distance_threshold

    def objects_matching(self, object1, object2):
        """
        There are 2 matching conditions for AngularIoU:
            - Objects are of same class
            AND
                - IoU in BeV is larger than a threshold
                OR
                - Objects angles align and they are not too distant
        Args:
            object1 : (GenericObject)
            object2 : (GenericObject)
        Returns:
            bool : matching or not
        """
        if not object_utils.similar_object_group(object1, object2):
            return False

        bev_iou = object_utils.bev_iou(object1, object2)
        if bev_iou > self.bev_iou_threshold:
            return True

        angle_difference = object_utils.angle_difference(object1, object2)
        center_distance = object_utils.bev_center_distance(object1, object2)
        if (angle_difference < self.angle_threshold) and (
            center_distance < self.center_distance_threshold
        ):
            return True
        else:
            return False

    def cluster(self, objects_list):
        """
        Cluster a list of objects together. Same logic as NMS.
        Instead of suppresing when overlap, add to the same cluster.
        Args:
            objects_list [[GenericObject]] : Each index contains predictions from a single model.
        Returns:
            [[GenericObject]] : Each index contains objects from a single cluster.
        """
        clusters = []
        all_objects = [j for i in objects_list for j in i]
        scores = [obj.score for obj in all_objects]
        idxs = np.argsort(scores)
        while len(idxs) > 0:
            last = len(idxs) - 1
            main_object = all_objects[idxs[last]]
            cluster = [main_object]
            suppress = [last]

            for pos in range(0, last):
                other_object = all_objects[idxs[pos]]
                if self.objects_matching(main_object, other_object):
                    cluster.append(other_object)
                    suppress.append(pos)

            idxs = np.delete(idxs, suppress)
            clusters.append(cluster)

        return clusters
