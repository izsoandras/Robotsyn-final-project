#
# This script uses example localization results to show
# what the figure should look like. You need to modify
# this script to work with your data.
#

import matplotlib.pyplot as plt
import numpy as np
from draw_point_cloud import *
import localize_image
from os.path import join
import cv2


def run():
    model = 'points.txt'
    K = np.loadtxt('../data_hw5_ext/calibration/K.txt').astype(np.float32)
    std_int = np.loadtxt('../data_hw5_ext/calibration/std_int.txt')
    fx_std, fy_std, cx_std, cy_std, k1_std, k2_std, p1_std, p2_std, k3_std, k4_std, k5_std, k6_std, s1_std, s2_std, s3_std, s4_std, taux_std, tauy_std = std_int
    folder = '../data_hw5_ext/'
    m = 10

    queries = ['../data_hw5_ext/IMG_8207.jpg','../data_hw5_ext/IMG_8217.jpg','../data_hw5_ext/IMG_8227.jpg']
    # queries = ['../data_hw5_ext/IMG_8224.jpg', '../data_hw5_ext/IMG_8228.jpg']
    # queries = ['../data_hw5_ext/IMG_8227.jpg']
    X = np.loadtxt(model)

    paramses = [[],[],[]]
    for i in range(m):
        rands = np.random.normal([0, 0, 0, 0], [fx_std, fy_std, cx_std, cy_std])

        K_disturbed = K + np.array([[rands[0], 0, rands[2]],[0, rands[1], rands[3]],[0, 0, 0]])

        for idx, query in enumerate(queries):
            img = cv2.imread(query)
            _, _, params = localize_image.localize_image(img, K_disturbed)
            paramses[idx].append(params)

    covs = []
    for params in paramses:
        params = np.array(params).T
        covs.append(np.cov(params))

    for cov, query in zip(covs, queries):
        print('-----------------------')
        print(f'Covariance for {query}:')
        print(f'Order: yaw [rad], pitch [rad], roll [rad], x [m], y [m], z [m]')
        print(cov)

if __name__ == "__main__":
    run()

