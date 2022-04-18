import numpy as np
import cv2
import matplotlib.pyplot as plt
from os.path import join
from matlab_inspired_interface import *
from estimate_E_ransac import estimate_E_ransac
from F_from_E import F_from_E
from figures import *
from triangulate_many import *
from decompose_E import *


def run(img_name1='IMG_8224.jpg', img_name2='IMG_8228.jpg'):
    folder = '../data_hw5_ext/calibration'
    K = np.loadtxt(join(folder, 'K.txt'))
    dc = np.loadtxt(join(folder, 'dc.txt'))

    img1_color = cv2.imread(f'../data_hw5_ext/{img_name1}')
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(f'../data_hw5_ext/{img_name2}', cv.IMREAD_GRAYSCALE)
    img1_color = img1_color/255.0

    img1 = cv2.undistort(img1, K, dc)
    img2 = cv2.undistort(img2, K, dc)

    # Initiate SIFT detector
    sift = cv2.SIFT_create(nfeatures=30000)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # NB! You will want to experiment with different options for the ratio test and
    # "unique" (cross-check).
    index_pairs, match_metric = match_features(des1, des2, max_ratio=0.9, unique=False)
    print('Found %d matches' % index_pairs.shape[0])

    kp1 = np.array([kp.pt for kp in kp1])
    kp2 = np.array([kp.pt for kp in kp2])

    kp1_match = np.array([kp1[idx] for idx in index_pairs[:,0]])
    kp2_match = np.array([kp2[idx] for idx in index_pairs[:,1]])
    des1_match = np.array([des1[idx] for idx in index_pairs[:,0]])
    des2_match = np.array([des2[idx] for idx in index_pairs[:,1]])

    uv1 = np.vstack([kp1_match.T, np.ones(kp1_match.shape[0])])
    uv2 = np.vstack([kp2_match.T, np.ones(kp2_match.shape[0])])

    xy1 = np.linalg.inv(K) @ uv1
    xy2 = np.linalg.inv(K) @ uv2
    E, errors, inliers_mask = estimate_E_ransac(xy1, xy2, K, 4, 2000)

    Ts = decompose_E(E)

    for T in Ts:
        X = triangulate_many(xy1[:, inliers_mask], xy2[:, inliers_mask], np.hstack((np.eye(3), np.zeros((3, 1)))),
                             T[0:3, :])

        if X[2, 0] > 0:
            X2 = T @ X[:, 0]
            if X2[2] > 0:
                break

    X = triangulate_many(xy1[:, inliers_mask], xy2[:, inliers_mask], np.hstack((np.eye(3), np.zeros((3, 1)))),
                         T[0:3, :])

    des1_inl = des1_match[inliers_mask]
    des2_inl = des2_match[inliers_mask]

    np.savetxt('./points.txt', X)
    np.savetxt('./des1.txt', des1_inl)
    np.savetxt('./des2.txt', des2_inl)

    # Plot the 50 best matches in two ways befure RANSAC
    best_index_pairs = index_pairs[np.argsort(match_metric)[:50]]
    best_kp1 = kp1[best_index_pairs[:, 0]]
    best_kp2 = kp2[best_index_pairs[:, 1]]
    plt.figure('Best 50/1')
    show_matched_features(img1, img2, best_kp1, best_kp2, method='falsecolor')
    plt.figure('Best 50/2')
    show_matched_features(img1, img2, best_kp1, best_kp2, method='montage')

    # Plot epipolar lines
    np.random.seed(123)  # Leave as commented out to get a random selection each time
    draw_correspondences(img1, img2, uv1[:, inliers_mask], uv2[:, inliers_mask], F_from_E(E, K), sample_size=8)

    plt.figure()
    print(uv1[:, inliers_mask].shape)
    draw_point_cloud(X, img1_color, uv1[:, inliers_mask], xlim=[-1, +1], ylim=[-1, +1], zlim=[1, 3])
    plt.show()


if __name__ == "__main__":
    run()
