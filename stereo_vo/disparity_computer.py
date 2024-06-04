import cv2
import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *
from mono_vo.feature_detector import *


class disparity_computer:
    # Depth of a point is inversely proportional to the difference of image points(disparity)
    def __init__(self, numDisparities=64, blockSize=9, algorithm="sgbm"):
        self.numDisparities = numDisparities
        if (
            algorithm == "bm"
        ):  # disparity is computed by comparing the sum of absolute differences of each 'block' of pixels.
            self.stereo = cv2.StereoBM_create(
                numDisparities=numDisparities, blockSize=blockSize
            )
        elif algorithm == "sgbm":  # forces similar disparity on neighbouring blocks
            P1 = blockSize * blockSize * 8
            P2 = blockSize * blockSize * 32
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=numDisparities,
                blockSize=blockSize,
                P1=P1,
                P2=P2,
            )
        # blockSize: Dimension of the patch to be compared(odd as it is centered at the current pixel)
        # numDisparities: Defines the maximum disparity. For each pixel block, we will find the best disparity (find the best matching patch) from 0 to max

    def compute(self, img_l, img_r, show=False):
        disparity = self.stereo.compute(img_l, img_r)
        if show:
            disparity_norm = cv2.normalize(
                disparity, disparity, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX
            )
            disparity = np.uint8(disparity_norm)
        return np.divide(disparity, 16.0)


def compute_pts_with_disp(pts_l, disp, min_thresh=0.0, max_thresh=100.0):
    pts_r = np.copy(pts_l)
    selected_points = np.zeros(pts_l.shape[0])
    for i in range(pts_l.shape[0]):
        disp_value = disp[int(pts_l[i, 0, 1]), int(pts_l[i, 0, 0])]  # y,x cuz opencv
        if disp_value > min_thresh and disp_value < max_thresh:
            pts_r[i, 0, 0] = pts_l[i, 0, 0] - disp_value
            selected_points[i] = 1
    selected_points = selected_points.astype(bool)
    return pts_l[selected_points], pts_r[selected_points]


def compute_pts_with_disp_sequence(
    pts_l_1, pts_l_2, disp_1, disp_2, min_thresh=0.0, max_thresh=100.0
):
    pts_r_1 = np.copy(pts_l_1)
    pts_r_2 = np.copy(pts_l_2)
    selected_points = np.zeros(pts_l_1.shape[0])
    for i in range(pts_l_1.shape[0]):
        disp_value_1 = disp_1[pts_l_1[i, 1], pts_l_1[i, 0]]  # y,x cuz opencv
        disp_value_2 = disp_2[pts_l_2[i, 1], pts_l_2[i, 0]]  # y,x cuz opencv
        if (
            disp_value_1 > min_thresh
            and disp_value_1 < max_thresh
            and disp_value_2 > min_thresh
            and disp_value_2 < max_thresh
        ):
            pts_r_1[i, 0] = pts_l_1[i, 0] - disp_value_1
            pts_r_2[i, 0] = pts_l_2[i, 0] - disp_value_2
            selected_points[i] = 1
    selected_points = selected_points.astype(bool)
    return (
        pts_l_1[selected_points],
        pts_r_1[selected_points],
        pts_l_2[selected_points],
        pts_r_2[selected_points],
    )


if __name__ == "__main__":
    data_dir = "kitti_dataset/sequences/00"
    disparity = disparity_computer(algorithm="sgbm")
    detector = feature_detector(threshold=20, nonmaxSuppression=True)
    img_l = cv2.imread(os.path.join(data_dir, "image_2", "000000.png"), 0)
    img_r = cv2.imread(os.path.join(data_dir, "image_3", "000000.png"), 0)
    disp = disparity.compute(img_l, img_r, show=False)

    ft_pts_l = detector.detect(img_l, show=False)
    ft_pts_l, ft_pts_r = compute_pts_with_disp(
        ft_pts_l, disp, min_thresh=-1.0, max_thresh=100.0
    )

    selection = list(range(10))
    ft_pts_l = ft_pts_l[selection]
    ft_pts_r = ft_pts_r[selection]

    imgs = []
    disparity_norm = cv2.normalize(
        disp, disp, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX
    )
    disp = np.uint8(disparity_norm)
    imgs.append(show_image(disp, append=True))
    imgs.append(show_features(img_l, ft_pts_l, append=True))
    imgs.append(show_features(img_r, ft_pts_r, append=True))

    show_imgs(imgs)
