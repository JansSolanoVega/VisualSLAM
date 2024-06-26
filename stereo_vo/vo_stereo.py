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


def triangulate(proj_l, proj_r, feature_pts_l, feature_pts_r):
    points_3d = np.zeros((len(feature_pts_l), 3))
    Q = np.zeros(shape=(4, 4))

    for i in range(len(feature_pts_l)):
        x_l, y_l = int(feature_pts_l[i][0]), int(feature_pts_l[i][1])
        x_r, y_r = int(feature_pts_r[i][0]), int(feature_pts_r[i][1])
        Q[0, :] = y_l * proj_l[2, :] - proj_l[1, :]
        Q[1, :] = -x_l * proj_l[2, :] + proj_l[0, :]
        Q[2, :] = y_r * proj_r[2, :] - proj_r[1, :]
        Q[3, :] = -x_r * proj_r[2, :] + proj_r[0, :]
        [u, s, v] = np.linalg.svd(Q)
        v = v.transpose()
        vSmall = v[:, -1]
        vSmall /= vSmall[-1]
        points_3d[i, :] = vSmall[0:-1]

    return points_3d


class visual_odometry_stereo:
    def __init__(self, sequence_id=0):
        self.sequence_id = sequence_id
        self.pose_file_path, self.img_file_path, self.calib_file_path = load_paths(
            DATA_DIR, sequence_id
        )

        with open(self.pose_file_path) as f:
            self.poses = f.readlines()

        self.img_l_path = self.img_file_path + "0"
        self.img_r_path = self.img_file_path + "1"

        self.feature_detector = feature_detector(threshold=20, nonmaxSuppression=True)
        self.disp_computer = disparity_computer(algorithm="sgbm")

        self.curr_frame = {
            "l": read_img_gray(self.img_l_path, id=0),
            "r": read_img_gray(self.img_r_path, id=0),
        }

        self.camera_params = {
            "l": load_calib(self.calib_file_path, camera_id=0),
            "r": load_calib(self.calib_file_path, camera_id=1),
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

        self.curr_frame = {
            "l": read_img_gray(self.img_l_path, self.img_id),
            "r": read_img_gray(self.img_r_path, self.img_id),
        }

        self.curr_disparity = self.disp_computer.compute(
            self.curr_frame["l"], self.curr_frame["r"]
        )

        self.curr_feature_pts["l"] = self.feature_detector.selective_detect(
            self.curr_frame["l"]
        )

        if self.img_id >= 1:
            (track_old_feature_pts_l, track_curr_feature_pts_l) = self.feature_tracker[
                "l"
            ].find_correspondance_points(
                self.old_feature_pts["l"], self.old_frame["l"], self.curr_frame["l"]
            )

            (
                track_old_feature_pts_l,
                track_old_feature_pts_r,
                track_curr_feature_pts_l,
                track_curr_feature_pts_r,
            ) = compute_pts_with_disp_sequence(
                track_old_feature_pts_l,
                track_curr_feature_pts_l,
                self.old_disparity,
                self.curr_disparity,
                min_thresh=-1.0,
                max_thresh=50.0,
            )
            # imgs = []

            # imgs.append(
            #     show_features(self.old_frame["l"], track_old_feature_pts_l, append=True)
            # )
            # imgs.append(
            #     show_features(self.old_frame["r"], track_old_feature_pts_r, append=True)
            # )
            # imgs.append(
            #     show_features(
            #         self.curr_frame["l"], track_curr_feature_pts_l, append=True
            #     )
            # )
            # imgs.append(
            #     show_features(
            #         self.curr_frame["r"], track_curr_feature_pts_r, append=True
            #     )
            # )

            # show_imgs(imgs)

            old_points_3d = triangulate(
                self.camera_params["l"]["proj_matrix"],
                self.camera_params["r"]["proj_matrix"],
                track_old_feature_pts_l,
                track_old_feature_pts_r,
            )

            curr_points_3d = triangulate(
                self.camera_params["l"]["proj_matrix"],
                self.camera_params["r"]["proj_matrix"],
                track_curr_feature_pts_l,
                track_curr_feature_pts_r,
            )

            clique = inlier_detect_iteration(
                old_points_3d, curr_points_3d, threshold=0.2
            )

            old_points_3d, curr_points_3d = (
                old_points_3d[clique],
                curr_points_3d[clique],
            )

            track_old_feature_pts_l, track_curr_feature_pts_l = (
                track_old_feature_pts_l[clique],
                track_curr_feature_pts_l[clique],
            )
            print(len(clique))

            dSeed = np.zeros(6)
            optRes = least_squares(  # Solve PnP problem: finding the camera pose given a set of 3D points in the world system and their corresponding 2D projections in the image plane
                function_reprojection_error,
                dSeed,
                method="lm",
                max_nfev=200,
                args=(
                    track_old_feature_pts_l,
                    track_curr_feature_pts_l,
                    old_points_3d,
                    curr_points_3d,
                    self.camera_params["l"]["proj_matrix"],
                ),
            )

            optRes = self.remove_points_with_bad_reproj_error(
                old_points_3d,
                curr_points_3d,
                track_old_feature_pts_l,
                track_curr_feature_pts_l,
                optRes=optRes,
                errorThreshold=0.5,
            )

            rot, tras = get_rot_traslation(optRes.x)

            if self.img_id == 1:
                self.R, self.t = rot, tras
            else:
                self.t = self.t + self.R @ tras
                self.R = rot @ self.R

        self.old_feature_pts = self.curr_feature_pts
        self.old_frame = self.curr_frame
        self.old_disparity = self.curr_disparity
        self.img_id += 1
        return self.t

    def remove_points_with_bad_reproj_error(
        self,
        old_points_3d,
        curr_points_3d,
        track_old_feature_pts_l,
        track_curr_feature_pts_l,
        optRes,
        errorThreshold=0.5,
    ):
        lClique = len(curr_points_3d)
        error = optRes.fun
        e = error.reshape((lClique * 2, 3))
        errorThreshold = errorThreshold
        reproj_error_in_x_old_bad = np.where(e[0:lClique, 0] >= errorThreshold)
        reproj_error_in_y_old_bad = np.where(e[0:lClique, 1] >= errorThreshold)
        reproj_error_in_z_old_bad = np.where(e[0:lClique, 2] >= errorThreshold)
        reproj_error_in_x_curr_bad = np.where(
            e[lClique : 2 * lClique, 0] >= errorThreshold
        )
        reproj_error_in_y_curr_bad = np.where(
            e[lClique : 2 * lClique, 1] >= errorThreshold
        )
        reproj_error_in_z_curr_bad = np.where(
            e[lClique : 2 * lClique, 2] >= errorThreshold
        )

        pruneIdx = (  # all 'good' points chosen with clique with bad reproj error
            reproj_error_in_x_old_bad[0].tolist()
            + reproj_error_in_y_old_bad[0].tolist()
            + reproj_error_in_z_old_bad[0].tolist()
            + reproj_error_in_x_curr_bad[0].tolist()
            + reproj_error_in_y_curr_bad[0].tolist()
            + reproj_error_in_z_curr_bad[0].tolist()
        )

        if len(pruneIdx) > 0:
            uPruneIdx = list(set(pruneIdx))
            old_points_3d = np.delete(old_points_3d, uPruneIdx, axis=0)
            curr_points_3d = np.delete(curr_points_3d, uPruneIdx, axis=0)
            track_old_feature_pts_l = np.delete(
                track_old_feature_pts_l, uPruneIdx, axis=0
            )
            track_curr_feature_pts_l = np.delete(
                track_curr_feature_pts_l, uPruneIdx, axis=0
            )
            if len(curr_points_3d) >= 6:
                optRes = least_squares(
                    function_reprojection_error,
                    optRes.x,
                    method="lm",
                    max_nfev=200,
                    args=(
                        track_old_feature_pts_l,
                        track_curr_feature_pts_l,
                        old_points_3d,
                        curr_points_3d,
                        self.camera_params["l"]["proj_matrix"],
                    ),
                )

        return optRes

    def get_true_coordinates(self):
        return get_vect_from_pose(
            self.poses[self.img_id].strip().split(),
        )

    def get_mono_coordinates(self):
        return self.t

    def get_mse_error(self):
        mono_coordinates = self.get_mono_coordinates()
        true_coordinates = self.get_true_coordinates()
        print("Mono coordinates:", mono_coordinates)
        print("True coordinates:", true_coordinates)
        return np.linalg.norm(mono_coordinates - true_coordinates)


if __name__ == "__main__":
    vo = visual_odometry_stereo(sequence_id=2)
    vo.process_frame()
