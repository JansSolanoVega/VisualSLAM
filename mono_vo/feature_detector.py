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

    def selective_detect(
        self, img, show=False, tile_h=10, tile_w=20, max_keypoints_per_patch=10
    ):
        # Loop in patches from image, and no more than 10 keypoints per patch
        H, W = img.shape
        detected_keypoints = []
        for y in range(0, H, tile_h):
            for x in range(0, W, tile_w):
                imPatch = img[y : y + tile_h, x : x + tile_w]
                keypoints = self.detector.detect(imPatch)
                for pt in keypoints:
                    pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

                if len(keypoints) > max_keypoints_per_patch:
                    keypoints = sorted(
                        keypoints, key=lambda x: -x.response
                    )  # response is the strength of a keypoint: how good it is
                    for kpt in keypoints[0:max_keypoints_per_patch]:
                        detected_keypoints.append(kpt)
                else:
                    for kpt in keypoints:
                        detected_keypoints.append(kpt)

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
        os.path.join(data_dir, "sequences", "00", "image_2", "000002.png"), 0
    )
    print(detector.selective_detect(img1, show=True))
