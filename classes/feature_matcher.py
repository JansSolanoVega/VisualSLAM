import cv2
from utils import *


class feature_matcher:
    def __init__(self, type="bf"):
        self.type_algo = type
        if self.type_algo == "bf":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif (
            self.type_algo == "flann"
        ):  # Much faster than brute force, but might not find best matches
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1,
            )
            search_params = dict(checks=50)  # or pass empty dictionary
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def get_matches(self, des1, des2):
        if self.type_algo == "bf":
            matches = self.matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good = matches[:100]
        elif self.type_algo == "flann":
            matches = self.matcher.knnMatch(
                des1, des2, k=2
            )  # For each point it finds the best two matches
            # Apply ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])  # The match with smaller distance, better
        return good

    def draw_matches(self, img1, kp1, img2, kp2, matches):
        draw_matches_func = (
            cv2.drawMatches if self.type_algo == "bf" else cv2.drawMatchesKnn
        )
        img = draw_matches_func(  # Sort them in the order of their distance
            img1,
            kp1,
            img2,
            kp2,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        show_image(img)
