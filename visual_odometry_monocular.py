import cv2
import os
import matplotlib.pyplot as plt
from feature_matcher import *
from utils import *
import numpy as np

from feature_tracker import *
from feature_detector import *

DATA_DIR = "kitti_dataset"


class visual_odometry_monocular:
    def __init__(self, sequence_id=0):
        self.sequence_id = sequence_id
        self.pose_file_path = os.path.join(
            DATA_DIR, "poses", str(sequence_id).zfill(2) + ".txt"
        )
        self.img_file_path = os.path.join(
            DATA_DIR, "sequences", str(sequence_id).zfill(2), "image_2"
        )
        try:
            with open(self.pose_file_path) as f:
                self.poses = f.readlines()
        except Exception as e:
            raise ValueError(
                "The pose_file_path is not valid or did not lead to a txt file"
            )

        self.detector = feature_detector(threshold=20, nonmaxSuppression=True)
        self.feature_tracker = klt_feature_tracker(true_poses=self.poses)

        self.current_frame = cv2.imread(
            os.path.join(self.img_file_path, str(0).zfill(6) + ".png"), 0
        )
        self.img_id = 0

    def process_frame(self):
        self.img_id += 1
        self.old_frame = self.current_frame
        self.current_frame_full_color = cv2.imread(
            os.path.join(self.img_file_path, str(self.img_id).zfill(6) + ".png")
        )
        self.current_frame = cv2.cvtColor(
            self.current_frame_full_color, cv2.COLOR_RGB2GRAY
        )

        feature_points = self.detector.detect(self.old_frame)
        self.t = self.feature_tracker.track_step(
            self.old_frame, self.current_frame, feature_points, self.img_id
        )

    def get_true_coordinates(self):
        return get_vect_from_pose(
            self.poses[self.img_id].strip().split(),
        )

    def get_mono_coordinates(self):
        diag = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)

        return adj_coord.flatten()

    def get_mse_error(self):
        return np.linalg.norm(self.get_mono_coordinates() - self.get_true_coordinates())


if __name__ == "__main__":
    vo = visual_odometry_monocular(sequence_id=2)
    vo.process_frame()
    print("Current traslation:", vo.t)
    print("MSE error:", vo.get_mse_error())
