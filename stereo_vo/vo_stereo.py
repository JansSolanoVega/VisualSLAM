import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from stereo_vo.disparity_computer import *
from mono_vo.feature_detector import *
from mono_vo.feature_tracker import *

DATA_DIR = "kitti_dataset"


class visual_odometry_stereo:
    def __init__(self, sequence_id=0):
        self.sequence_id = sequence_id
        self.pose_file_path, self.img_file_path, self.calib_file_path = load_paths(
            DATA_DIR, sequence_id
        )

        with open(self.pose_file_path) as f:
            self.poses = f.readlines()

        self.img_l_path = self.img_file_path + "2"
        self.img_r_path = self.img_file_path + "3"

        self.detector = feature_detector(threshold=20, nonmaxSuppression=True)
        self.disp_computer = disparity_computer(numDisparities=64, blockSize=9)

        self.current_frame_l = read_img_gray(self.img_l_path, id=0)
        self.current_frame_r = read_img_gray(self.img_r_path, id=0)
        self.img_id = 0

        self.camera_params_l = load_calib(self.calib_file_path, camera_id=2)
        self.camera_params_r = load_calib(self.calib_file_path, camera_id=3)
        self.feature_tracker_l = klt_feature_tracker(camera_params=self.camera_params_l)
        self.feature_tracker_r = klt_feature_tracker(camera_params=self.camera_params_r)

    def triangulate(self, camera_params_l, feature_pts, disparity, r_camera_shift=0.54):
        points_3d = []
        Q = np.zeros(shape=(4, 4))
        Q[0][0] = 1
        Q[1][1] = 1
        Q[0][3] = -camera_params_l["principal_point"][0]
        Q[1][3] = -camera_params_l["principal_point"][1]
        Q[2][3] = -camera_params_l["focal_length"]
        Q[3][2] = -1 / (r_camera_shift)
        Q[3][3] = 0  # Both cameras are equal

        for feature_pt in feature_pts:
            x, y = int(feature_pt[0]), int(feature_pt[1])
            d = disparity[y][x]
            homo_coord_3d = Q @ np.array([x, y, d, 1])
            points_3d.append(homo_coord_3d[:3])

        return points_3d

    def process_frame(self):
        self.img_id += 1
        self.old_frame_l = self.current_frame_l
        self.curr_frame_l = read_img_gray(self.img_l_path, self.img_id)
        self.curr_frame_r = read_img_gray(self.img_r_path, self.img_id)

        self.curr_disparity = self.disp_computer.compute(
            self.curr_frame_l, self.curr_frame_r
        )

        feature_pts_l = self.detector.detect(self.old_frame_l)

        tracked_pts_old_l, tracked_pts_curr_l = (
            self.feature_tracker_l.find_correspondance_points(
                feature_pts_l, self.old_frame_l, self.curr_frame_l
            )
        )

        points_3d = self.triangulate(
            self.camera_params_l, tracked_pts_curr_l, self.curr_disparity
        )

        return points_3d

    def get_true_coordinates(self):
        return get_vect_from_pose(
            self.poses[self.img_id].strip().split(),
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
    vo = visual_odometry_stereo(sequence_id=2)
    vo.process_frame()
    print("Current traslation:", vo.t)
    print("MSE error:", vo.get_mse_error())
