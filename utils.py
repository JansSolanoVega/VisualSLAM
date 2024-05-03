import cv2
import numpy as np


def form_transformation(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_calib(filepath):
    with open(filepath, "r") as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=" ")
        P = np.reshape(params, (3, 4))  # Projection matrix: A_(int,f)*Pf
        K = P[0:3, 0:3]  # Intrinsic matrix
    return K, P


def get_vect_from_pose(pose):
    x = float(pose[3])
    y = float(pose[7])
    z = float(pose[11])
    return np.array([[x], [y], [z]])


def show_image(img, shape=(1080, 720)):
    img = cv2.resize(img, shape)
    cv2.imshow("fig", img), cv2.waitKey(0)
    cv2.destroyAllWindows()


class plotter:
    def __init__(self, shape=(1440, 720)):
        self.shape = shape
        self.traj = np.zeros(shape=(shape[0] // 2, shape[1], 3))

    def get_trajectory_step(self, true_x, true_z, draw_x, draw_z):
        self.traj = cv2.circle(
            self.traj, (true_x + 400, true_z + 100), 1, list((0, 0, 255)), 4
        )
        self.traj = cv2.circle(
            self.traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4
        )

        cv2.putText(
            self.traj,
            "Actual Position:",
            (140, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            self.traj, "Red", (270, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
        cv2.putText(
            self.traj,
            "Estimated Odometry Position:",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            self.traj,
            "Green",
            (270, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        return np.asarray(self.traj, np.uint8)

    def plot_step(self, current_trajectory, current_frame):
        current_frame = cv2.resize(current_frame, (self.shape[1], self.shape[0] // 2))

        horizontal_concatenation = np.hstack([current_trajectory, current_frame])
        cv2.imshow("Figure", horizontal_concatenation)
        cv2.waitKey(1000 // 60)
