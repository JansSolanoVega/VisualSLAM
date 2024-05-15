import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from stereo_vo.disparity_computer import *
from mono_vo.feature_detector import *

DATA_DIR = "kitti_dataset"

class visual_odometry_stereo:
    def __init__(self, sequence_id=0):
        self.sequence_id = sequence_id
        self.pose_file_path = os.path.join(
            DATA_DIR, "poses", str(sequence_id).zfill(2) + ".txt"
        )
        self.imgs_file_path = os.path.join(DATA_DIR, "sequences", str(sequence_id).zfill(2)) 
        self.calib_file_path = os.path.join(
            DATA_DIR, "sequences", str(sequence_id).zfill(2), "calib.txt"
        )
        try:
            with open(self.pose_file_path) as f:
                self.poses = f.readlines()
        except Exception as e:
            raise ValueError(
                "The pose_file_path is not valid or did not lead to a txt file"
            )

        self.img_l_path = os.path.join(self.imgs_file_path, "image_2") 
        self.img_r_path = os.path.join(self.imgs_file_path, "image_3") 

        self.detector = feature_detector(threshold=20, nonmaxSuppression=True)
        self.disp_computer = disparity_computer(numDisparities=64, blockSize=9)

        self.current_frame_l = read_img_gray(self.img_l_path, 0)
        self.current_frame_r = read_img_gray(self.img_r_path, 0)
        self.img_id = 0

    def process_frame(self):
        self.img_id += 1
        self.old_frame = self.current_frame
        self.current_frame_l = read_img_gray(self.img_l_path, self.img_id)
        self.current_frame_r = read_img_gray(self.img_r_path, self.img_id)
        
        self.current_disparity = self.disp_computer(self.current_frame_l, self.current_frame_r)

        feature_points = self.detector.detect(self.current_frame_l)
        
        self.t = self.feature_tracker.track_step(
            self.old_frame, self.current_frame, feature_points, self.img_id
        )

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
    vo = visual_odometry_monocular(sequence_id=2)
    vo.process_frame()
    print("Current traslation:", vo.t)
    print("MSE error:", vo.get_mse_error())
