import cv2
import os
import numpy as np


class feature_detector:
    def __init__(self, threshold=20, nonmaxSuppression=True):
        self.detector = cv2.FastFeatureDetector_create(
            threshold=threshold, nonmaxSuppression=nonmaxSuppression
        )

    def detect(self, img):
        detected_keypoints = self.detector.detect(img)

        return np.array([kp.pt for kp in detected_keypoints], dtype=np.float32).reshape(
            -1, 1, 2
        )


if __name__ == "__main__":
    data_dir = "KITTI_sequence_1"
    detector = feature_detector(threshold=20, nonmaxSuppression=True)
    img1 = cv2.imread(os.path.join(data_dir, "image_r", "000002.png"))
    print(detector.detect(img1))
