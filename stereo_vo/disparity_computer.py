import cv2
import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


class disparity_computer:
    # Depth of a point is inversely proportional to the difference of image points(disparity)
    def __init__(self, numDisparities=64, blockSize=9):
        self.stereo = cv2.StereoBM_create(
            numDisparities=numDisparities, blockSize=blockSize
        )
        # blockSize: Dimension of the patch to be compared(odd as it is centered at the current pixel)
        # numDisparities: Defines the maximum disparity. For each pixel block size, we will find the best disparity (find the best matching patch) from 0 to max

    def compute(self, img_l, img_r, show=False):
        disparity = self.stereo.compute(img_l, img_r)
        if show:
            disparity_norm = cv2.normalize(
                disparity, disparity, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX
            )
            disparity = np.uint8(disparity_norm)
        return disparity


if __name__ == "__main__":
    data_dir = "kitti_dataset/sequences/00"
    disparity = disparity_computer()
    img1 = cv2.imread(os.path.join(data_dir, "image_2", "000000.png"), 0)
    img2 = cv2.imread(os.path.join(data_dir, "image_3", "000000.png"), 0)
    disp = disparity.compute(img1, img2)
    show_image(disp)
