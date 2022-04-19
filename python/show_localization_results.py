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
    folder = '../data_hw5_ext/'

    queries = ['../data_hw5_ext/IMG_8207.jpg','../data_hw5_ext/IMG_8217.jpg','../data_hw5_ext/IMG_8227.jpg']
    # queries = ['../data_hw5_ext/IMG_8224.jpg', '../data_hw5_ext/IMG_8228.jpg']
    # queries = ['../data_hw5_ext/IMG_8227.jpg']
    # queries = []
    # img_ids = []
    # for i in range(3):
    #     rnd = 8
    #     while rnd in img_ids or rnd in [8, 22]:
    #         rnd = np.random.randint(7, 31)
    #
    #     img_ids.append(rnd)
    #     path = join(folder, f'IMG_82{rnd:02d}.jpg')
    #     queries.append(path)

    # 3D points [4 x num_points].
    X = np.loadtxt(model)

    # Model-to-query transformation.
    T_m2qs = []
    Sigmas = []
    for query in queries:
        img = cv2.imread(query)
        T, Sigma = localize_image.localize_image(img)
        T_m2qs.append(T)
        Sigmas.append(Sigma)

    # If you have colors for your point cloud model...
    colors = np.loadtxt('colors.txt') # RGB colors [num_points x 3].
    # ...otherwise...
    # colors = np.zeros((X.shape[1], 3))

    # These control the visible volume in the 3D point cloud plot.
    # You may need to adjust these if your model does not show up.
    xlim = [-1,+2]
    ylim = [-1,+1]
    zlim = [0,+3.5]

    frame_size = 0.2
    marker_size = 5

    # plt.figure('3D point cloud', figsize=(6,6))
    # ax = draw_point_cloud(X, T_m2qs[0], xlim, ylim, zlim, colors=colors, marker_size=marker_size, frame_size=frame_size)
    for T_m2q, q in zip(T_m2qs,queries):
        plt.figure(f'3D point cloud, {q}', figsize=(6, 6))
        draw_point_cloud(X, T_m2q, xlim, ylim, zlim, colors=colors, marker_size=marker_size, frame_size=frame_size)
        # draw_frame(ax, T_m2q, frame_size)

    for Sigma, q in zip(Sigmas,queries):
        yaw_dev = np.rad2deg(np.sqrt(Sigma[0,0]))
        pitch_dev = np.rad2deg(np.sqrt(Sigma[1, 1]))
        roll_dev = np.rad2deg(np.sqrt(Sigma[2, 2]))
        x_dev = np.sqrt(Sigma[3, 3]) * 1000
        y_dev = np.sqrt(Sigma[4, 4]) * 1000
        z_dev = np.sqrt(Sigma[5, 5]) * 1000

        print('====================')
        print(f'Standard deviations for: {q}')
        print(f'\tYaw: {yaw_dev:10.4f}')
        print(f'\tPitch: {pitch_dev:10.4f}')
        print(f'\tRoll: {roll_dev:10.4f}')
        print('--------------------')
        print(f'\tX: {x_dev:10.4f}')
        print(f'\tY: {y_dev:10.4f}')
        print(f'\tZ: {z_dev:10.4f}')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()

