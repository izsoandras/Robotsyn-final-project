import numpy as np

def estimate_E(xy1, xy2):
    n = xy1.shape[1]
    A = np.vstack((xy1[0:1,:] * xy2[0:1,:], xy1[1:2,:] * xy2[0:1,:], xy2[0:1,:], xy1[0:1,:]*xy2[1:2], xy1[1:2,:]*xy2[1:2,:], xy2[1:2,:], xy1[0:1,:], xy1[1:2,:], xy1[2:3,:])).T

    [_, S, VT] = np.linalg.svd(A)
    h = VT[np.where(S == np.min(S[S > np.finfo(float).eps]))[0][0], :].T

    E = np.reshape(h, (3, 3))

    return E
