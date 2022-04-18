import numpy as np
from estimate_E import *
from epipolar_distance import *
from F_from_E import *
from triangulate_many import *

def estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials):

    # Tip: The following snippet extracts a random subset of 8
    # correspondences (w/o replacement) and estimates E using them.

    max_cnt = 0
    e_best = (None,None)
    inliers_mask = (None,None)

    for i in range(num_trials):
        sample = np.random.choice(xy1.shape[1], size=8, replace=False)
        E = estimate_E(xy1[:,sample], xy2[:,sample])

        F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

        e1, e2 = epipolar_distance(F, K@xy1, K@xy2)
        e = (e1 + e2)/2

        inl_mask = np.abs(e) < distance_threshold
        cnt = inl_mask.sum()

        if cnt > max_cnt:
                e_best = e
                inliers_mask = inl_mask
                max_cnt = cnt


    E_best = estimate_E(xy1[:,inliers_mask], xy2[:,inliers_mask])

    return E_best, e_best, inliers_mask
