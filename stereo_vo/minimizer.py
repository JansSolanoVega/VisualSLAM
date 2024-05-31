from scipy.optimize import least_squares
import numpy as np
import sys, os
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *

DATA_DIR = "kitti_dataset"


def get_rot_traslation(x):
    r, tras = x[:3], x[3:]
    rot = R.from_rotvec(r).as_matrix()
    return rot, tras.reshape(3, 1)


def function_reprojection_error(x, ft1, ft2, w1, w2, proj):
    """
    For a pair of features 'ft1' and 'ft2' at time 't' and 't+1', with corresponding triangulated 3d coordinates 'w1' and 'w2'. These 3d coordinates
    are reprojected into image to estimate 2d image points ('w1' estimates the feature for time 't+1' ('ft2') and 'w2' estimates 'ft1'). Then we calculate the error between the reprojections
    and the original features coordinates.
    Returns:
        The reprojection error depending on x, the parameter to find
    Params:
        proj: Projection matrix of left camera
        w1,w2: 3d coordinates of the features triangulated
        ft1,ft2: 2d coordinates of features in Il_t, Il_t+1
        T: Transformation matrix
        x: Variable param containing r and t
    """
    n_features = ft1.shape[0]
    reproj1_error, reproj2_error = np.zeros(shape=(n_features, 3)), np.zeros(
        shape=(n_features, 3)
    )
    f_curr, f_next = np.zeros(shape=(3)), np.zeros(shape=(3))
    w_curr, w_next = np.zeros(shape=(4)), np.zeros(shape=(4))
    r, t = get_rot_traslation(x)
    T = np.concatenate((r, t), axis=1)
    T = np.vstack((T, [0, 0, 0, 1]))

    forward_proj = proj @ T
    backward_proj = proj @ np.linalg.inv(T)

    for i in range(n_features):
        f_curr[:2], f_next[:2] = ft1[i], ft2[i]
        f_curr[2], f_next[2] = 1, 1  # Adding homogeneous coord

        w_curr[:3], w_next[:3] = w1[i], w2[i]
        w_curr[3], w_next[3] = 1, 1  # Adding homogeneous coord

        reproj1_error[i] = forward_proj @ w_next
        reproj1_error[i] = f_curr - (
            reproj1_error[i] / reproj1_error[i][2]
        )  # Removing homogeneous coord

        reproj2_error[i] = backward_proj @ w_curr
        reproj2_error[i] = f_next - (
            reproj2_error[i] / reproj2_error[i][2]
        )  # Removing homogeneous coord
    #np.vstack((reproj1_error, reproj2_error)).shape=(2*n_features, 3)
    return np.vstack((reproj1_error, reproj2_error)).flatten()


if __name__ == "__main__":
    _, _, calib_file_path = load_paths(DATA_DIR, sequence_id=0)
    p = load_calib(calib_file_path, camera_id=2)["proj_matrix"]

    n_features = 10
    w1, w2 = np.random.rand(n_features, 3), np.random.rand(n_features, 3)
    ft1, ft2 = np.random.rand(n_features, 2), np.random.rand(n_features, 2)

    x0 = np.random.randn(6)
    res_1 = least_squares(
        function_reprojection_error, x0, method="lm", args=(ft1, ft2, w1, w2, p)
    )
    print("Param minimizing reproj error:", res_1.x) #Dim: (len(x), 1)
    print("Reprojection errors:", res_1.fun) #Dim: (n_features x reproj_dim x 2, 1)
