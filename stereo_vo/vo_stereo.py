import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from stereo_vo.disparity_computer import *
from stereo_vo.inlier_detector import *
from mono_vo.feature_detector import *
from mono_vo.feature_tracker import *
from stereo_vo.minimizer import *

DATA_DIR = "kitti_dataset"


def triangulate(camera_params_l, camera_params_r, feature_pts_l, feature_pts_r):
    points_3d = np.zeros((len(feature_pts_l), 3))
    Q = np.zeros(shape=(4, 4))
    proj_l = camera_params_l["proj_matrix"]
    proj_r = camera_params_r["proj_matrix"]

    for i in range(len(feature_pts_l)):
        x_l, y_l = int(feature_pts_l[i][0]), int(feature_pts_l[i][1])
        x_r, y_r = int(feature_pts_r[i][0]), int(feature_pts_r[i][1])
        Q[0, :] = y_l * proj_l[2, :] - proj_l[1, :]
        Q[1, :] = -x_l * proj_l[2, :] + proj_l[0, :]
        Q[2, :] = y_r * proj_r[2, :] - proj_r[1, :]
        Q[3, :] = -x_r * proj_r[2, :] + proj_r[0, :]
        [u, s, v] = np.linalg.svd(Q)
        v = v[-1]  # last row correspond to the minimal singular value

        v /= v[-1]  # Homogoneus
        points_3d[i, :] = v[0:-1]

    return points_3d


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

        self.feature_detector = feature_detector(threshold=20, nonmaxSuppression=True)
        self.disp_computer = disparity_computer(numDisparities=64, blockSize=9)

        self.curr_frame = {
            "l": read_img_gray(self.img_l_path, id=0),
            "r": read_img_gray(self.img_r_path, id=0),
        }

        self.camera_params = {
            "l": load_calib(self.calib_file_path, camera_id=2),
            "r": load_calib(self.calib_file_path, camera_id=3),
        }

        self.feature_tracker = {
            "l": klt_feature_tracker(camera_params=self.camera_params["l"]),
            "r": klt_feature_tracker(camera_params=self.camera_params["r"]),
        }

        self.img_id = 0
        # Rotation and traslation matrix
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 1))

        self.curr_feature_pts = {}

    def process_frame(self):

        self.current_frame_full_color = cv2.imread(
            get_path_img(self.img_l_path, self.img_id)
        )

        self.curr_frame = {
            "l": read_img_gray(self.img_l_path, self.img_id),
            "r": read_img_gray(self.img_r_path, self.img_id),
        }

        self.curr_disparity = self.disp_computer.compute(
            self.curr_frame["l"], self.curr_frame["r"]
        )

        self.curr_feature_pts["l"] = self.feature_detector.detect(self.curr_frame["l"])

        self.curr_feature_pts["l"], self.curr_feature_pts["r"] = compute_pts_with_disp(
            self.curr_feature_pts["l"],
            self.curr_disparity,
            min_thresh=-1.0,
            max_thresh=20.0,
        )

        if self.img_id > 1:
            (track_old_feature_pts_l, track_curr_feature_pts_l) = self.feature_tracker[
                "l"
            ].find_correspondance_points(
                self.old_feature_pts["l"], self.old_frame["l"], self.curr_frame["l"]
            )
            (track_old_feature_pts_r, track_curr_feature_pts_r) = self.feature_tracker[
                "r"
            ].find_correspondance_points(
                self.old_feature_pts["r"], self.old_frame["r"], self.curr_frame["r"]
            )
            old_points_3d = triangulate(
                self.camera_params["l"],
                self.camera_params["r"],
                track_old_feature_pts_l,
                track_old_feature_pts_r,
            )

            curr_points_3d = triangulate(
                self.camera_params["l"],
                self.camera_params["r"],
                track_curr_feature_pts_l,
                track_curr_feature_pts_r,
            )

            # old_points_3d, curr_points_3d = inlier_detector(
            #    old_points_3d, curr_points_3d
            # )
            x0 = np.random.randn(6)
            # self.R, self.t = get_rot_traslation(
            #     least_squares(
            #         function_reprojection_error,
            #         x0,
            #         method="lm",
            #         args=(
            #             track_old_feature_pts_l,
            #             track_curr_feature_pts_l,
            #             old_points_3d,
            #             curr_points_3d,
            #             self.camera_params["l"]["proj_matrix"],
            #         ),
            #     )
            # )

        self.old_feature_pts = self.curr_feature_pts
        self.old_frame = self.curr_frame
        self.img_id += 1
        return self.t

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
