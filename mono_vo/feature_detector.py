import cv2
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


class feature_detector:
    def __init__(self, threshold=20, nonmaxSuppression=True):
        self.detector = cv2.FastFeatureDetector_create(
            threshold=threshold, nonmaxSuppression=nonmaxSuppression
        )

    def detect(self, img, show=False):
        detected_keypoints = self.detector.detect(img)

        detected_pts = np.array(
            [kp.pt for kp in detected_keypoints], dtype=np.float32
        ).reshape(-1, 1, 2)

        if show:
            show_features(img, detected_pts)

        return detected_pts


def show_features(img, fts, append=False):
    if fts.shape[1] == 1:
        fts = fts[:, 0, :]
    keypoints = [cv2.KeyPoint(float(x), float(y), 10) for x, y in fts]
    return show_image(cv2.drawKeypoints(img, keypoints, None), append=append)


if __name__ == "__main__":
    data_dir = "kitti_dataset"
    detector = feature_detector(threshold=20, nonmaxSuppression=True)
    img1 = cv2.imread(
        os.path.join(data_dir, "sequences", "00", "image_2", "000002.png")
    )
    print(detector.detect(img1, show=True))
