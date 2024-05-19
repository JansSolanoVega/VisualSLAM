import cv2
import matplotlib.pyplot as plt
import numpy as np

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from mono_vo.feature_tracker import *
from mono_vo.feature_detector import *

DATA_DIR = "kitti_dataset"


class visual_odometry_monocular:
    def __init__(self, sequence_id=0, camera_id=2):
        self.sequence_id = sequence_id

        self.pose_file_path, self.img_file_path, self.calib_file_path = load_paths(
            DATA_DIR, sequence_id
        )

        self.img_file_path += str(camera_id)

        with open(self.pose_file_path) as f:
            self.true_poses = f.readlines()

        camera_params = load_calib(self.calib_file_path, camera_id=camera_id)
        self.detector = feature_detector(threshold=20, nonmaxSuppression=True)
        self.feature_tracker = klt_feature_tracker(
            camera_params=camera_params, true_poses=self.true_poses
        )

        self.current_frame = cv2.imread(get_path_img(self.img_file_path, 0), 0)
        self.img_id = 0

    def process_frame(self):
        self.img_id += 1
        self.old_frame = self.current_frame
        self.current_frame_full_color = cv2.imread(
            get_path_img(self.img_file_path, self.img_id)
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
            self.true_poses[self.img_id].strip().split(),
        )

    def get_mono_coordinates(self):
        diag = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
        )  # Matching to opencv coordinates (vertical direction is flipped)
        adj_coord = np.matmul(diag, self.t)

        return adj_coord.flatten()

    def get_mse_error(self):
        mono_coordinates = self.get_mono_coordinates()
        true_coordinates = self.get_true_coordinates()
        print("Mono coordinates:", mono_coordinates)
        print("True coordinates:", true_coordinates)
        return np.linalg.norm(mono_coordinates - true_coordinates)


if __name__ == "__main__":
    vo = visual_odometry_monocular(sequence_id=2)
    vo.process_frame()
    print("Current traslation:", vo.t)
    print("MSE error:", vo.get_mse_error())
