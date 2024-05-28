import cv2
import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *
from mono_vo.feature_detector import *


class disparity_computer:
    # Depth of a point is inversely proportional to the difference of image points(disparity)
    def __init__(self, numDisparities=64, blockSize=9):
        self.numDisparities = numDisparities
        self.stereo = cv2.StereoBM_create(
            numDisparities=numDisparities, blockSize=blockSize
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


if __name__ == "__main__":
    data_dir = "kitti_dataset/sequences/00"
    disparity = disparity_computer()
    detector = feature_detector(threshold=20, nonmaxSuppression=True)
    img_l = cv2.imread(os.path.join(data_dir, "image_2", "000000.png"), 0)
    img_r = cv2.imread(os.path.join(data_dir, "image_3", "000000.png"), 0)
    disp = disparity.compute(img_l, img_r, show=False)

    ft_pts_l = detector.detect(img_l, show=False)
    ft_pts_l, ft_pts_r = compute_pts_with_disp(
        ft_pts_l, disp, min_thresh=58.0, max_thresh=100.0
    )

    imgs = []
    disparity_norm = cv2.normalize(
        disp, disp, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX
    )
    disp = np.uint8(disparity_norm)
    imgs.append(show_image(disp, append=True))
    imgs.append(show_features(img_l, ft_pts_l, append=True))
    imgs.append(show_features(img_r, ft_pts_r, append=True))

    show_imgs(imgs)
