import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from pyquaternion import Quaternion
from tridet.structures.pose import Pose
from detectron2.structures.boxes import BoxMode
from tridet.structures.boxes3d import GenericBoxes3D

from active_learning.utils.dataset_utils import read_txt, write_txt
from active_learning.utils.iou_utils import calculate_iou_2d
from tridet.evaluators.kitti_3d_evaluator import convert_3d_box_to_kitti


class GenericObject:
    def __init__(self, input, input_type="kitti_string"):
        """
        Generic Object Format to fit other formats
        ## Taken from https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt
        #Values    Name      Description
        ----------------------------------------------------------------------------
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                            'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                            'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                            truncated refers to the object leaving image boundaries
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                            0 = fully visible, 1 = partly occluded
                            2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                            contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1    score        Only for results: Float, indicating confidence in
                            detection, needed for p/r curves, higher is better.
        """
        if input_type == "kitti_string":
            self.read_from_kitti_string(input)
        elif input_type == "prediction_json":
            self.read_from_prediction_json(input)
        else:
            assert False, "This input type not supported"

    def read_from_kitti_string(self, object_string):
        object_information = object_string.split(" ")
        self.type = object_information[0]
        self.truncated = float(object_information[1])
        self.occluded = float(object_information[2])
        self.alpha = float(object_information[3])
        self.bbox_left = float(object_information[4])
        self.bbox_top = float(object_information[5])
        self.bbox_right = float(object_information[6])
        self.bbox_bottom = float(object_information[7])
        self.dim_h = float(object_information[8])
        self.dim_w = float(object_information[9])
        self.dim_l = float(object_information[10])
        self.loc_x = float(object_information[11])
        self.loc_y = float(object_information[12])
        self.loc_z = float(object_information[13])
        self.rot_y = float(object_information[14])
        if len(object_information) == 16:
            self.score = float(object_information[15])
        else:
            self.score = 1.0

    def read_from_prediction_json(self, anno):
        class_name = anno["category"]
        box3d = GenericBoxes3D.from_vectors([anno["bbox3d"]])
        W, L, H, x, y, z, rot_y, alpha = convert_3d_box_to_kitti(box3d)
        l, t, r, b = BoxMode.convert(
            anno["bbox"], from_mode=BoxMode.XYWH_ABS, to_mode=BoxMode.XYXY_ABS
        )
        score = anno["score_3d"]

        self.type = anno["category"]
        self.truncated = float(-1)
        self.occluded = float(-1)
        self.alpha = float(alpha)
        self.bbox_left = float(l)
        self.bbox_top = float(t)
        self.bbox_right = float(r)
        self.bbox_bottom = float(b)
        self.dim_h = float(H)
        self.dim_w = float(W)
        self.dim_l = float(L)
        self.loc_x = float(x)
        self.loc_y = float(y)
        self.loc_z = float(z)
        self.rot_y = float(rot_y)
        self.score = float(score)

    @property
    def difficulty(self):
        """
        http://www.cvlibs.net/datasets/kitti/eval_object.php
        Difficulties are defined as follows:
        - Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
        - Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
        - Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %
        All methods are ranked based on the moderately difficult results.
        """
        height = abs(self.bbox_bottom - self.bbox_top)
        occ = self.occluded
        trunc = self.truncated
        # print(height)
        # print(occ)
        # print(trunc)

        if height > 40 and occ == 0 and trunc <= 0.15:
            return "Easy"
        elif height > 25 and occ <= 1 and trunc <= 0.3:
            return "Moderate"
        elif height > 25 and occ <= 2 and trunc <= 0.5:
            return "Hard"
        else:
            return "Ignored"

    @property
    def kitti_annotation_list(self):
        annotation_list = []
        annotation_list.append(self.type)
        annotation_list.append(self.truncated)
        annotation_list.append(self.occluded)
        annotation_list.append(self.alpha)
        annotation_list.append(self.bbox_left)
        annotation_list.append(self.bbox_top)
        annotation_list.append(self.bbox_right)
        annotation_list.append(self.bbox_bottom)
        annotation_list.append(self.dim_h)
        annotation_list.append(self.dim_w)
        annotation_list.append(self.dim_l)
        annotation_list.append(self.loc_x)
        annotation_list.append(self.loc_y)
        annotation_list.append(self.loc_z)
        annotation_list.append(self.rot_y)
        annotation_list.append(self.score)
        return annotation_list

    @property
    def generic_boxes_3d(self):
        box_pose = Pose(
            wxyz=Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
            * Quaternion(axis=[0, 0, 1], radians=-self.rot_y),
            tvec=np.float64([self.loc_x, self.loc_y - self.dim_h / 2, self.loc_z]),
        )
        box3d = GenericBoxes3D(
            box_pose.quat.elements, box_pose.tvec, [self.dim_w, self.dim_l, self.dim_h]
        )
        return box3d

    @property
    def kitti_string(self):
        return "{} {} {:d} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
            self.type,
            self.truncated,
            int(self.occluded),
            self.alpha,
            self.bbox_left,
            self.bbox_top,
            self.bbox_right,
            self.bbox_bottom,
            self.dim_h,
            self.dim_w,
            self.dim_l,
            self.loc_x,
            self.loc_y,
            self.loc_z,
            self.rot_y,
            self.score,
        )

    @property
    def corners(self):
        """
        Returns corners in GenericBoxes3D.corners format (8,3) array
        (x,y,z) corresponds to (width, height, length).
        """
        return self.generic_boxes_3d.corners.cpu().numpy()[0]

    @property
    def bev_corners(self):
        """
        Return corners from BeV space.
            corners are in (x,y,z) order with (width, height, length) numpy array of shape (8,3)
            corners_bev are in (x,z) numpy array of shape (4,2)
        """
        corners_bev = self.corners[:, [0, 2]][[0, 1, 5, 4]]
        return corners_bev

    @property
    def image_corners(self):
        """
        Return corners from image space (4,2) array. Start from top left, go clockwise
        """
        corners = []
        corners.append([self.bbox_left, self.bbox_top])
        corners.append([self.bbox_right, self.bbox_top])
        corners.append([self.bbox_right, self.bbox_bottom])
        corners.append([self.bbox_left, self.bbox_bottom])
        return np.array(corners)

    @property
    def angle(self):
        return np.arctan2(self.loc_z, self.loc_x) * 180 / np.pi

    def __repr__(self) -> str:
        repr_str = "{}({:.1f}%) - Loc:({:.2f},{:.2f},{:.2f}) - Dim:({:.2f},{:.2f},{:.2f})".format(
            self.type,
            self.score * 100,
            self.loc_x,
            self.loc_y,
            self.loc_z,
            self.dim_h,
            self.dim_w,
            self.dim_l,
        )
        return repr_str

    def draw_on_image(self, image, color=(255, 0, 0)):
        image = cv2.rectangle(
            image,
            (int(self.bbox_left), int(self.bbox_top)),
            (int(self.bbox_right), int(self.bbox_bottom)),
            color,
            1,
        )
        return image


def read_labels_as_objects(path_to_label_txt):
    labels_txt = read_txt(path_to_label_txt)
    objects = []
    for label_line in labels_txt:
        objects.append(GenericObject(label_line))
    return objects


def write_objects_as_labels(objects, path_to_label_txt):
    kitti_strings = [obj.kitti_string for obj in objects]
    labels_txt = [
        " ".join(kitti_string.split(" ")[:-1]) for kitti_string in kitti_strings
    ]
    write_txt(labels_txt, path_to_label_txt)
    return objects


def match_objects(objects_1, objects_2, distance_fn, matching_threshold=5.0):
    """
    Match a set of objects based on a distance function
    Args:
        objects_1 (list[GenericObject]): First list of objects
        objects_2 (list[GenericObject]): Other list of objects
        distance_fn (func(GenericObject, GenericObject) -> float) : Distance function for matching
        matching_threshold (float) : Threshold for matching
    Returns:
        a dict of matches {index_for_detections_1: index_for_detections_2}
    """
    if len(objects_1) == 0 or len(objects_2) == 0:
        return {}
    distances = []
    for object_1 in objects_1:
        row = []
        for object_2 in objects_2:
            if object_1.type != object_2.type:
                row.append(1e6)
            else:
                row.append(distance_fn(object_1, object_2))
        distances.append(row)
    distances = np.array(distances)
    row_ind, col_ind = linear_sum_assignment(distances)
    matches = {}
    for r, c in zip(row_ind, col_ind):
        if distances[r, c] < matching_threshold:
            matches[r] = c
    return matches


SIMLAR_OBJECT_GROUPS = [
    set(["Pedestrian", "Cyclist", "Person_sitting"]),
    set(["Car", "Van", "Truck"]),
]

def similar_object_group(object1, object2):
    group1 = None
    for group in SIMLAR_OBJECT_GROUPS:
        if object1.type in group:
            group1 = group
            break

    if group1 is None:
        return same_class(object1, object2)

    if object2.type in group1:
        return True
    else:
        return False


def same_class(object1, object2):
    return object1.type == object2.type


def threshold_valid_classes(
    objects, valid_class_names=("Car", "Pedestrian", "Cyclist", "Van", "Truck")
):
    return [obj for obj in objects if obj.type in valid_class_names]


def threshold_class(objects, class_name="DontCare"):
    return [obj for obj in objects if obj.type != class_name]


def threshold_objects(objects, threshold=0.3):
    return [obj for obj in objects if obj.score > threshold]


def angle_difference(object1, object2):
    return abs(object1.angle - object2.angle)


def bev_iou(object1, object2):
    return calculate_iou_2d(object1.bev_corners, object2.bev_corners)


def bev_center_distance(object1, object2):
    return np.sqrt(
        (object1.loc_x - object2.loc_x) ** 2 + (object1.loc_z - object2.loc_z) ** 2
    )


def center_distance(object1, object2):
    return np.sqrt(
        (object1.loc_x - object2.loc_x) ** 2
        + (object1.loc_y - object2.loc_y) ** 2
        + (object1.loc_z - object2.loc_z) ** 2
    )
