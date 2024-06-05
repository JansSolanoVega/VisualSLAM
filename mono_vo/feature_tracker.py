import cv2
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *
from mono_vo.feature_detector import *


def filter_tracked_points_error(
    tracked_pts_img1, tracked_pts_img2, errTrackablePoints, error_thresh=4
):
    errThresholdedPoints = np.where(errTrackablePoints < error_thresh, 1, 0).astype(
        bool
    )
    tracked_pts_img1 = tracked_pts_img1[errThresholdedPoints, ...]
    tracked_pts_img2 = tracked_pts_img2[errThresholdedPoints, ...]
    return tracked_pts_img1, tracked_pts_img2


class klt_feature_tracker:
    def __init__(
        self,
        camera_params,
        true_poses=None,
        winSize=(21, 21),
        max_number_iterations=30,
        max_level=3,
        epsilon_or_accuracy=0.01,
    ):
        self.lukas_kanade_params = dict(
            winSize=winSize,
            maxLevel=max_level,
            criteria=(
                cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT,
                max_number_iterations,
                epsilon_or_accuracy,
            ),
        )
        self.focal_length = camera_params["focal_length"]
        self.principal_point = camera_params["principal_point"]

        # Rotation and traslation matrix
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 1))

        self.true_poses = true_poses

    def get_absolute_scale(self, pose1, pose2):

        true_vect = get_vect_from_pose(pose2)
        prev_vect = get_vect_from_pose(pose1)

        return np.linalg.norm(true_vect - prev_vect)

    def get_extrinsic_params(self, tracked_pts_img1, tracked_pts_img2):
        # 5 points are the min number of points to compute the essential matrix
        E, _ = cv2.findEssentialMat(  # RANSAC separates data into inliers and outliers
            # At every iterationit randomly samples five points and estimates E
            # Then check if the other points are inliers when using this E
            # E with the maximum number of points agree is chosen
            tracked_pts_img1,
            tracked_pts_img2,
            focal=self.focal_length,
            pp=self.principal_point,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        _, R, t, _ = cv2.recoverPose(
            E=E,
            points1=tracked_pts_img1,
            points2=tracked_pts_img2,
            focal=self.focal_length,
            pp=self.principal_point,
        )
        return R, t

    def find_correspondance_points(self, feature_pts_img1, img1, img2):
        # status returns 1 if the flow has been found, otherwise, it is set to 0
        # next_points return the calculated new positions in the second image

        feature_pts_img2, status, error = cv2.calcOpticalFlowPyrLK(
            img1,
            img2,
            prevPts=feature_pts_img1,
            nextPts=None,
            flags=cv2.MOTION_AFFINE,  # not only for rotation and translation, but also scaling and shearing
            **self.lukas_kanade_params
        )
        tracked_pts_img1, tracked_pts_img2 = (
            feature_pts_img1[status == 1],
            feature_pts_img2[status == 1],
        )

        errTrackablePoints = error[status == 1]

        tracked_pts_img1, tracked_pts_img2 = filter_tracked_points_error(
            tracked_pts_img1, tracked_pts_img2, errTrackablePoints
        )

        tracked_pts_img1, tracked_pts_img2 = check_inside_image(
            tracked_pts_img1, tracked_pts_img2, img1.shape
        )

        tracked_pts_img1 = tracked_pts_img1.astype("int32")
        tracked_pts_img2 = tracked_pts_img2.astype("int32")

        return tracked_pts_img1, tracked_pts_img2

    def track_step(self, img1, img2, feature_pts_img1, img_id):

        tracked_pts_img1, tracked_pts_img2 = self.find_correspondance_points(
            feature_pts_img1, img1, img2
        )

        R, t = self.get_extrinsic_params(tracked_pts_img1, tracked_pts_img2)

        if img_id < 2:
            self.R = R
            self.t = t
        else:
            # These steps are neccesary because we have only one camera, therefore we cant have measures without scale
            absolute_scale = self.get_absolute_scale(
                pose1=self.true_poses[img_id - 1].strip().split(),
                pose2=self.true_poses[img_id].strip().split(),
            )

            if (
                absolute_scale > 0.1
                and abs(t[2][0]) > abs(t[0][0])
                and abs(t[2][0]) > abs(t[1][0])
            ):  # TODO: CHECK BUG IN CURVE FOR MONOCULAR
                self.t = self.t + absolute_scale * self.R.dot(t)
                self.R = R.dot(self.R)

        return self.t


if __name__ == "__main__":
    data_dir = "kitti_dataset"
    _, img_file_path, calib_file_path = load_paths(data_dir, sequence_id=2)
    camera_params = load_calib(calib_file_path, camera_id=0)
    tracker = klt_feature_tracker(camera_params=camera_params)
    detector = feature_detector(threshold=20, nonmaxSuppression=True)
    img1 = cv2.imread(os.path.join(img_file_path + "0", "000000.png"), 0)
    img2 = cv2.imread(os.path.join(img_file_path + "0", "000001.png"), 0)
    feature_points = detector.selective_detect(img1, max_keypoints_per_patch=2)
    tracked_pts_img1, tracked_pts_img2 = tracker.find_correspondance_points(
        feature_points, img1, img2
    )
    print(tracker.lukas_kanade_params)
    # print(tracker.track_step(img1, img2, feature_points, img_id=0))
    imgs = []

    imgs.append(show_features(img1, tracked_pts_img1, append=True))
    imgs.append(show_features(img2, tracked_pts_img2, append=True))

    show_imgs(imgs)
