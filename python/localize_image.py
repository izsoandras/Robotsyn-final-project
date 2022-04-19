import numpy as np
import cv2
from matlab_inspired_interface import *
from common import *
import scipy.optimize


class LocalizationOptimizer:
    def __init__(self, K, distCoeff, R0, t0, worldPoints, imPoints, Sigma_r=None):
        self.K = K
        self.distCoeff = distCoeff
        self.T0 = np.vstack((np.hstack((R0,t0)), np.array([0, 0, 0, 1])))
        self.worldPoints = worldPoints
        self.imPoints = imPoints

        if Sigma_r is None:
            Sigma_r = np.eye(2*worldPoints.shape[1])

        self.Sigma_r = Sigma_r
        self.sqrt_Sigma_r = np.linalg.inv(np.linalg.cholesky(Sigma_r))

    def residuals(self, params):
        roll = params[0]
        pitch = params[1]
        yaw = params[2]
        x = params[3]
        y = params[4]
        z = params[5]

        T = rotate_z(yaw) @ rotate_y(pitch) @ rotate_x(roll)
        T[0:3,-1] = np.array([x, y, z])

        T = T @ self.T0

        camPoints = T @ self.worldPoints
        uv_hat = project(self.K, camPoints, self.distCoeff)

        diff = self.imPoints - uv_hat
        return self.sqrt_Sigma_r @ diff.flatten()


def localize_image(img, K=None, sigma_u=1, sigma_v=1):
    if K is None:
        K = np.loadtxt('../data_hw5_ext/calibration/K.txt').astype(np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dc = np.loadtxt('../data_hw5_ext/calibration/dc.txt').astype(np.float32)
    base_points = np.loadtxt('./points.txt').astype(np.float32)
    des1 = np.loadtxt('./des1.txt').astype(np.float32)
    des2 = np.loadtxt('./des2.txt').astype(np.float32)
    des_base = np.vstack((des1, des2))

    img = cv2.undistort(img, K, dc)
    sift = cv2.SIFT_create(nfeatures=30000)
    kp, des = sift.detectAndCompute(img, None)

    index_pairs, match_metric = match_features(des2, des, max_ratio=0.65, unique=False)

    world_points = base_points[:,index_pairs[:,0]]
    image_points = np.array([kp[idx].pt for idx in index_pairs[:,1]]).T.astype(np.float32)

    retval, rvec_w2c, tvec_w2c, inliers = cv2.solvePnPRansac((world_points[0:3,:]/world_points[3,:]).T, image_points.T, K, dc)

    Sigma_r = np.diag(np.array([sigma_u**2,]*world_points.shape[1] + [sigma_v**2,]*world_points.shape[1]))

    optimizer = LocalizationOptimizer(K, dc, cv2.Rodrigues(rvec_w2c)[0], tvec_w2c, world_points, image_points, Sigma_r)
    opt_res = scipy.optimize.least_squares(optimizer.residuals, np.array([0, ]*6), verbose=2)

    opt_params = opt_res.x
    T = rotate_z(opt_params[2]) @ rotate_y(opt_params[1]) @ rotate_x(opt_params[0])
    T[0:3,-1] = opt_params[3:]
    T = T @ optimizer.T0

    J = opt_res.jac

    Sigma_p = np.linalg.inv(J.T @ Sigma_r @ J)

    return optimizer.T0, Sigma_p, opt_params


if __name__ == "__main__":
    img = cv2.imread('../data_hw5_ext/IMG_8207.jpg')
    localize_image(img)
