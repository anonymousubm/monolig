import numpy as np
from collections import OrderedDict
from pyquaternion import Quaternion

from tridet.structures.boxes3d import GenericBoxes3D
from tridet.structures.pose import Pose
from detectron2.structures.boxes import BoxMode

def get_3d_annotation(label):
    """Convert KITTI annotation data frame to 3D bounding box annotations.
    Labels are provided in the reference frame of camera_2.
    NOTE: Annotations are returned in the reference of the requested sensor
    """
    height, width, length = label[8:11]
    x, y, z = label[11:14]
    rotation = label[14]

    # We modify the KITTI annotation axes to our convention  of x-length, y-width, z-height.
    # Additionally, KITTI3D refers to the center of the bottom face of the cuboid, and our convention
    # refers to the center of the 3d cuboid which is offset by height/2. To get a bounding box
    # back in KITTI coordinates, i.e. for evaluation, see `self.convert_to_kitti`.
    box_pose = Pose(
        wxyz=Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
        * Quaternion(axis=[0, 0, 1], radians=-rotation),
        tvec=np.float64([x, y - height / 2, z]),
    )

    box3d = GenericBoxes3D(
        box_pose.quat.elements, box_pose.tvec, [width, length, height]
    )
    vec = box3d.vectorize().tolist()[0]
    distance = float(np.linalg.norm(vec[4:7]))

    return OrderedDict([("bbox3d", vec), ("distance", distance)])

def get_2d_annotation(label):
    l, t, r, b = label[4:8]
    return OrderedDict(bbox=[l, t, r, b], bbox_mode=BoxMode.XYXY_ABS)