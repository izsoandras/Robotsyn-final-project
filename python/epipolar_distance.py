import numpy as np

def epipolar_distance(F, uv1, uv2):
    """
    F should be the fundamental matrix (use F_from_E)
    uv1, uv2 should be 3 x n homogeneous pixel coordinates
    """
    n = uv1.shape[1]

    e1 = np.sum(uv2 * (F @ uv1),axis=0) / np.sqrt(np.power((F@uv1)[0,:],2) + np.power((F@uv1)[1,:],2))
    e2 = np.sum(uv1 * (F.T @ uv2),axis=0) / np.sqrt(np.power((F.T @ uv2)[0, :], 2) + np.power((F.T @ uv2)[1, :], 2))

    return e1,e2
