import cv2
import os
import matplotlib.pyplot as plt
from classes.feature_matcher import *
from utils import *
import numpy as np


class visualOdometry:
    def __init__(self, data_dir, imgs):
        self.orb = cv2.ORB_create(500)  # Number of features
        self.fm = feature_matcher(type="bf")
        self.images = imgs
        self.IntrinsicMatrix, self.ProjectionMatrix = load_calib(
            os.path.join(data_dir, "calib.txt")
        )

    def get_matches_between_imgs(self, i):
        keypoints1, descriptors1 = self.orb.detectAndCompute(self.images[i - 1], None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(self.images[i], None)
        matches = self.fm.get_matches(descriptors1, descriptors2)
        self.fm.draw_matches(
            self.images[i - 1], keypoints1, self.images[i], keypoints2, matches
        )

        good_keypoints_1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        good_keypoints_2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        return good_keypoints_1, good_keypoints_2

    def get_transformation_matrix(self, kp1, kp2):
        essential_matrix, _ = cv2.findEssentialMat(kp1, kp2, self.K)
        R1, R2, t = cv2.decomposeEssentialMat(essential_matrix)

        # Four possible transformations
        T1 = form_transformation(R1, np.ndarray.flatten(t))
        T2 = form_transformation(R2, np.ndarray.flatten(t))
        T3 = form_transformation(R1, np.ndarray.flatten(-t))
        T4 = form_transformation(R2, np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]

        # Projections
        projections = [
            self.ProjectionMatrix @ T1,
            self.ProjectionMatrix @ T2,
            self.ProjectionMatrix @ T3,
            self.ProjectionMatrix @ T4,
        ]

        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.IntrinsicMatrix, P, kp1, kp2)
            hom_Q2 = T @ hom_Q1

    def find_features_img(self, img):
        keypoints = self.orb.detect(img, None)  # Performing FAST feature detector
        keypoints, _ = self.orb.compute(
            img, keypoints
        )  # Performing BRIEF descriptor method
        img = cv2.drawKeypoints(
            img,
            keypoints,
            outImage=None,
            flags=0,  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        show_image(img)


if __name__ == "__main__":
    data_dir = "KITTI_sequence_1"
    img1 = cv2.imread(os.path.join(data_dir, "image_r", "000000.png"))
    img2 = cv2.imread(os.path.join(data_dir, "image_r", "000001.png"))
    vo = visualOdometry(data_dir, imgs=[img1, img2])
    kp1, kp2 = vo.get_matches_between_imgs(i=0)
    print(len(kp1[0]))
